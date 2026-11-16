"""Quick sanity check of the converted gesture and expression models on tiles of HaGRID's README mosaic
(a 10x5 grid of 128 px sample photos; HaGRID license, not included here).

Usage:
  curl -L --create-dirs -o weights/testdata/example.jpeg \
      https://raw.githubusercontent.com/hukenovs/hagrid/master/images/example.jpeg
  .venv/bin/python -I tools/check_models.py weights/testdata/example.jpeg
"""
import sys
import time

import coremltools as ct
import numpy as np
from PIL import Image

TILE = 128
# (row, col) tiles in the 10x5 mosaic with a clearly visible gesture
GESTURE_TILES = {"thumbs up": (0, 8), "thumbs down": (4, 4), "mute": (2, 9), "peace-ish": (3, 4)}
# rough face box inside tile (row 1, col 8), in tile-local pixels
FACE_TILE, FACE_BOX = (1, 8), (10, 10, 85, 95)


def tile(img: Image.Image, rc):
    r, c = rc
    return img.crop((c * TILE, r * TILE, (c + 1) * TILE, (r + 1) * TILE))


def letterbox(img: Image.Image, w=640, h=384):
    s = min(w / img.width, h / img.height)
    nw, nh = round(img.width * s), round(img.height * s)
    canvas = Image.new("RGB", (w, h), (114, 114, 114))
    px, py = (w - nw) // 2, (h - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas


def main(path: str) -> None:
    img = Image.open(path).convert("RGB")
    gest = ct.models.MLModel("App/Models/GestureNet.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
    emo = ct.models.MLModel("App/Models/EmotionNet.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
    g_names = gest.user_defined_metadata["classes"].split(",")
    e_names = emo.user_defined_metadata["classes"].split(",")

    for label, rc in GESTURE_TILES.items():
        det = gest.predict({"image": letterbox(tile(img, rc))})["detections"][0]  # (anchors, 4+nc)
        scores = det[:, 4:]
        best_anchor = int(scores.max(1).argmax())
        top = np.argsort(-scores[best_anchor])[:3]
        box = det[best_anchor, :4].round(1)
        print(f"{label:12s} -> " + ", ".join(f"{g_names[i]} {scores[best_anchor, i]:.2f}" for i in top) + f"  box(x1y1x2y2)={box.tolist()}")

    x0, y0, x1, y1 = FACE_BOX
    face = tile(img, FACE_TILE).crop((x0, y0, x1, y1)).resize((224, 224), Image.BILINEAR)
    out = emo.predict({"image": face})
    probs, va = out["probs"][0], out["va"][0]
    top = np.argsort(-probs)[:3]
    print("face        -> " + ", ".join(f"{e_names[i]} {probs[i]:.2f}" for i in top) + f"  valence={va[0]:.2f} arousal={va[1]:.2f}")

    # A few warm timings (not a benchmark): Python + coremltools overhead included.
    lb = letterbox(tile(img, GESTURE_TILES["thumbs up"]))
    for name, model, feed in (("gesture", gest, {"image": lb}), ("expression", emo, {"image": face})):
        model.predict(feed)
        t = time.perf_counter()
        for _ in range(10):
            model.predict(feed)
        print(f"{name:10s} ~{(time.perf_counter() - t) / 10 * 1000:.1f} ms/predict via coremltools (CPU+ANE)")


if __name__ == "__main__":
    main(sys.argv[1])
