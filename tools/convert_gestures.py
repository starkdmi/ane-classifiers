"""Convert HaGRIDv2 YOLOv10n gesture detector (Ultralytics checkpoint) to Core ML.

We keep YOLOv10's NMS-free one-to-one head but stop *before* its top-k step, so the
app gets every anchor with all class scores (lets the UI show full per-class bars).

Input  : `image`       640x384 (WxH) RGB letterboxed frame, 0-255 (scaled in-model to 0-1)
Output : `detections`  (1, anchors, 4 + nc) float32 = x1, y1, x2, y2 (input pixels),
                       then one sigmoid score per gesture class

Usage: .venv/bin/python -I tools/convert_gestures.py weights/hagrid/YOLOv10n_gestures.pt App/Models/GestureNet.mlpackage
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coremltools as ct
import numpy as np
import torch

from safe_load import ALLOWED_PREFIXES

H, W = 384, 640


def main(src: str, dst: str) -> None:
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(src)
    bad = [g for g in unsafe if not g.startswith(ALLOWED_PREFIXES)]
    if bad:
        raise RuntimeError(f"refusing to load, unexpected globals: {bad}")

    from ultralytics import YOLO

    yolo = YOLO(src)  # Ultralytics' loader maps the legacy YOLOv10DetectionModel class name
    names = [yolo.names[i] for i in range(len(yolo.names))]
    print(f"{len(names)} classes:", names)

    model = yolo.model.float().eval()
    # Legacy v10 checkpoints store `end2end` as a plain attribute that the newer head's property
    # shadows; without restoring it, fuse() would drop the NMS-free one-to-one branch.
    model.model[-1].end2end = True
    model = model.fuse()
    head = model.model[-1]
    print("head:", type(head).__name__, "end2end:", head.end2end, "stride:", head.stride.tolist())
    head.export = True
    head.format = "coreml"
    head.dynamic = False
    head.postprocess = lambda preds: preds  # skip top-k, keep (B, anchors, 4+nc)
    for p in model.parameters():
        p.requires_grad_(False)

    example = torch.rand(1, 3, H, W)
    with torch.no_grad():
        ref = model(example)
        print("torch output:", tuple(ref.shape))
        traced = torch.jit.trace(model, example, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="detections", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = "HaGRIDv2 YOLOv10n gesture detector, raw one-to-one head (no top-k)"
    mlmodel.author = "HaGRID (hukenovs), Ultralytics YOLOv10; converted for Core ML"
    mlmodel.license = "HaGRID license (CC BY-SA 4.0 variant); Ultralytics AGPL-3.0"
    mlmodel.user_defined_metadata["classes"] = ",".join(names)
    mlmodel.user_defined_metadata["input_hw"] = f"{H},{W}"
    mlmodel.save(dst)
    print("saved", dst)

    from PIL import Image

    img = Image.fromarray((example[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    out = mlmodel.predict({"image": img})["detections"]
    diff = np.abs(out - ref.numpy())
    print("shape:", out.shape, "max |box diff| px:", float(diff[..., :4].max()), "max |score diff|:", float(diff[..., 4:].max()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
