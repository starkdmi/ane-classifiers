"""Convert MediaPipe BlazeFace (PyTorch port by M. Hollemans, Apache-2.0) to Core ML.

The network definition below (BlazeBlock, FinalBlazeBlock, BlazeFace.forward and the anchor decoding) is copied
from https://github.com/hollance/BlazeFace-PyTorch (blazeface.py): Copyright (c) 2019 Matthijs Hollemans, under
the same terms as MediaPipe, Apache License 2.0 (LICENSES/Apache-2.0.txt). The rest of this file is MIT.
Anchor decoding is baked into the model, so Core ML returns final boxes + scores; the app only
thresholds and runs MediaPipe's weighted NMS.

Input  : `image`   SxS RGB (S = 128 front / 256 back), letterboxed with black; normalized in-model to [-1, 1]
Outputs: `boxes`   (1, 896, 4) x1, y1, x2, y2 normalized to the input square
         `scores`  (1, 896)    sigmoid face confidence

Usage: .venv/bin/python -I tools/convert_blazeface.py {front|back} weights/blazeface App/Models/BlazeFace.mlpackage
"""
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x
        if self.channel_pad > 0:
            # zero channels appended (same as F.pad on dim 1, but friendlier to converters);
            # BlazeFace never pads more channels than it has, so a slice of zeros always suffices
            x = torch.cat([x, torch.zeros_like(x[:, : self.channel_pad])], dim=1)
        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 2, 0, groups=channels, bias=True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convs(F.pad(x, (0, 2, 0, 2), "constant", 0)))


class BlazeFace(nn.Module):
    def __init__(self, back_model=False):
        super().__init__()
        self.back_model = back_model
        if back_model:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 24, 5, 2, 0, bias=True), nn.ReLU(inplace=True),
                *[BlazeBlock(24, 24) for _ in range(7)],
                BlazeBlock(24, 24, stride=2), *[BlazeBlock(24, 24) for _ in range(7)],
                BlazeBlock(24, 48, stride=2), *[BlazeBlock(48, 48) for _ in range(7)],
                BlazeBlock(48, 96, stride=2), *[BlazeBlock(96, 96) for _ in range(7)],
            )
            self.final = FinalBlazeBlock(96)
            self.classifier_8 = nn.Conv2d(96, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)
            self.regressor_8 = nn.Conv2d(96, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)
        else:
            self.backbone1 = nn.Sequential(
                nn.Conv2d(3, 24, 5, 2, 0, bias=True), nn.ReLU(inplace=True),
                BlazeBlock(24, 24), BlazeBlock(24, 28), BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36), BlazeBlock(36, 42), BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56), BlazeBlock(56, 64), BlazeBlock(64, 72), BlazeBlock(72, 80), BlazeBlock(80, 88),
            )
            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2), *[BlazeBlock(96, 96) for _ in range(4)],
            )
            self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)
            self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

    def forward(self, x):
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        b = x.shape[0]
        if self.back_model:
            x = self.backbone(x)
            h = self.final(x)
        else:
            x = self.backbone1(x)
            h = self.backbone2(x)
        c = torch.cat((self.classifier_8(x).permute(0, 2, 3, 1).reshape(b, -1, 1),
                       self.classifier_16(h).permute(0, 2, 3, 1).reshape(b, -1, 1)), dim=1)
        r = torch.cat((self.regressor_8(x).permute(0, 2, 3, 1).reshape(b, -1, 16),
                       self.regressor_16(h).permute(0, 2, 3, 1).reshape(b, -1, 16)), dim=1)
        return r, c


class Decoded(nn.Module):
    """BlazeFace + MediaPipe anchor decoding (boxes only, keypoints dropped)."""

    def __init__(self, net: BlazeFace, anchors: torch.Tensor, scale: float):
        super().__init__()
        self.net = net
        self.scale = scale
        self.register_buffer("anchors", anchors)  # (896, 4): x_center, y_center, w, h (normalized)

    def forward(self, x):
        r, c = self.net(x)
        a, s = self.anchors, self.scale
        xc = r[..., 0] / s * a[:, 2] + a[:, 0]
        yc = r[..., 1] / s * a[:, 3] + a[:, 1]
        w = r[..., 2] / s * a[:, 2]
        h = r[..., 3] / s * a[:, 3]
        boxes = torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=-1)
        scores = torch.sigmoid(c.clamp(-100.0, 100.0)).squeeze(-1)
        return boxes, scores


def build(variant: str, weights_dir: str) -> tuple[Decoded, int]:
    back = variant == "back"
    size = 256 if back else 128
    net = BlazeFace(back_model=back)
    state = torch.load(f"{weights_dir}/{'blazefaceback' if back else 'blazeface'}.pth", map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    anchors = torch.tensor(np.load(f"{weights_dir}/{'anchorsback' if back else 'anchors'}.npy"), dtype=torch.float32)
    return Decoded(net.eval(), anchors, float(size)).eval(), size


def main(variant: str, weights_dir: str, dst: str) -> None:
    model, size = build(variant, weights_dir)
    example = torch.rand(1, 3, size, size) * 2 - 1
    with torch.no_grad():
        ref_boxes, ref_scores = model(example)
        traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1 / 127.5, bias=[-1.0, -1.0, -1.0],
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="boxes", dtype=np.float32), ct.TensorType(name="scores", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = f"MediaPipe BlazeFace ({variant}) with anchor decoding; boxes x1y1x2y2 normalized"
    mlmodel.author = "Google MediaPipe; PyTorch port by M. Hollemans; converted for Core ML"
    mlmodel.license = "Apache-2.0"
    mlmodel.user_defined_metadata["variant"] = variant
    mlmodel.user_defined_metadata["min_score"] = "0.65" if variant == "back" else "0.75"
    mlmodel.save(dst)
    print("saved", dst)

    from PIL import Image

    img = Image.fromarray(((example[0].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8))
    out = mlmodel.predict({"image": img})
    print("max |box diff|:", float(np.abs(out["boxes"] - ref_boxes.numpy()).max()),
          "max |score diff|:", float(np.abs(out["scores"] - ref_scores.numpy()).max()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
