"""Compare BlazeFace front vs back on webcam-like 16:9 frames (face at several sizes).

Usage: .venv/bin/python -I tools/check_blazeface.py weights/blazeface/1face.png
"""
import sys
import time

import coremltools as ct
import numpy as np
from PIL import Image


def letterbox_square(frame: Image.Image, size: int):
    s = min(size / frame.width, size / frame.height)
    nw, nh = round(frame.width * s), round(frame.height * s)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(frame.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas, (s, px, py)


def detect(model, img, min_score):
    out = model.predict({"image": img})
    boxes, scores = out["boxes"][0], out["scores"][0]
    keep = scores >= min_score
    return boxes[keep], scores[keep]


def main(face_path: str) -> None:
    face = Image.open(face_path).convert("RGB")
    models = {v: ct.models.MLModel(f"App/Models/BlazeFace_{v}.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
              for v in ("front", "back")}
    sizes = {"front": 128, "back": 256}
    min_scores = {"front": 0.75, "back": 0.65}
    print(f"source face image {face.size}")

    for frac in (0.6, 0.4, 0.25, 0.15):  # face image height as a fraction of a 1920x1080 frame
        frame = Image.new("RGB", (1920, 1080), (90, 100, 110))
        h = int(1080 * frac)
        w = int(face.width * h / face.height)
        frame.paste(face.resize((w, h), Image.BILINEAR), ((1920 - w) // 2, (1080 - h) // 3))
        row = []
        for v, model in models.items():
            img, _ = letterbox_square(frame, sizes[v])
            boxes, scores = detect(model, img, min_scores[v])
            best = f"{scores.max():.2f}" if len(scores) else "-"
            row.append(f"{v}: {len(scores):2d} raw boxes, best {best}")
        print(f"face img {frac:4.0%} of frame height -> " + " | ".join(row))

    img, _ = letterbox_square(Image.new("RGB", (1920, 1080), (90, 100, 110)), 256)
    for v, model in models.items():
        feed = {"image": letterbox_square(Image.new("RGB", (1920, 1080)), sizes[v])[0]}
        model.predict(feed)
        t = time.perf_counter()
        for _ in range(10):
            model.predict(feed)
        print(f"{v}: ~{(time.perf_counter() - t) / 10 * 1000:.2f} ms/predict via coremltools (CPU+ANE)")


if __name__ == "__main__":
    main(sys.argv[1])
