"""Convert EmotiEffLib enet_b0_8_va_mtl (timm EfficientNet-B0) to a Core ML ML Program.

Input  : `image`  224x224 RGB face crop (Core ML scales 0-255 -> 0-1; ImageNet
         normalization is folded into the network).
Outputs: `probs`  (1, 8) softmax over Anger, Contempt, Disgust, Fear, Happiness,
                  Neutral, Sadness, Surprise
         `va`     (1, 2) valence, arousal (roughly -1..1)

Usage: .venv/bin/python -I tools/convert_emotion.py weights/emotiefflib/enet_b0_8_va_mtl.pt App/Models/EmotionNet.mlpackage
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coremltools as ct
import numpy as np
import torch

from safe_load import safe_torch_load

CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class EmotionWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):  # x: (1,3,224,224) in 0..1
        scores = self.net((x - self.mean) / self.std)
        return torch.softmax(scores[:, :8], dim=1), scores[:, 8:]


def make_same_padding_static(model: torch.nn.Module, example: torch.Tensor) -> int:
    """Swap timm's Conv2dSame (pads from runtime shapes) for ZeroPad2d + Conv2d with the
    TF 'SAME' padding precomputed for this fixed input size. Numerically identical."""
    import math

    from timm.layers.conv2d_same import Conv2dSame

    input_hw: dict[str, tuple[int, int]] = {}
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, Conv2dSame):
            hooks.append(m.register_forward_pre_hook(lambda mod, inp, name=name: input_hw.__setitem__(name, inp[0].shape[-2:])))
    with torch.no_grad():
        model(example)
    for h in hooks:
        h.remove()

    for name, (ih, iw) in input_hw.items():
        conv: Conv2dSame = model.get_submodule(name)
        (kh, kw), (sh, sw), (dh, dw) = conv.kernel_size, conv.stride, conv.dilation
        ph = max((math.ceil(ih / sh) - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pw = max((math.ceil(iw / sw) - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        plain = torch.nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, 0,
                                conv.dilation, conv.groups, conv.bias is not None)
        plain.load_state_dict(conv.state_dict())
        replacement = torch.nn.Sequential(torch.nn.ZeroPad2d((pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)), plain)
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child, replacement)
    return len(input_hw)


def main(src: str, dst: str) -> None:
    legacy = safe_torch_load(src)  # pickled with timm 0.9; its module objects don't run on timm 1.x
    print("legacy classifier:", legacy.classifier)
    import timm

    # Conv2dSame layers => TF-ported EfficientNet-B0; rebuild it and copy the weights over strictly.
    net = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=legacy.classifier.out_features)
    net.load_state_dict(legacy.state_dict(), strict=True)
    net.eval()
    model = EmotionWrapper(net).eval()

    example = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        ref_probs, ref_va = model(example)
        print("static-padded convs:", make_same_padding_static(net, (example - model.mean) / model.std))
        probs2, va2 = model(example)
        print("padding swap max diff:", float((probs2 - ref_probs).abs().max()), float((va2 - ref_va).abs().max()))
        traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="probs"), ct.TensorType(name="va")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = "EmotiEffLib enet_b0_8_va_mtl: 8 AffectNet expressions + valence/arousal"
    mlmodel.author = "EmotiEffLib (sb-ai-lab), converted for Core ML"
    mlmodel.license = "Apache-2.0 code; weights trained on AffectNet (research use)"
    mlmodel.user_defined_metadata["classes"] = ",".join(CLASSES)
    mlmodel.save(dst)
    print("saved", dst)

    # Sanity check: same random image through Core ML vs PyTorch.
    from PIL import Image

    img = Image.fromarray((example[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    out = mlmodel.predict({"image": img})
    print("max |probs diff|:", float(np.abs(out["probs"] - ref_probs.numpy()).max()))
    print("max |va diff|   :", float(np.abs(out["va"] - ref_va.numpy()).max()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
