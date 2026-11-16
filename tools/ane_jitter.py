"""Measure ANE call-latency jitter for the three app models with no camera running.

Calls BlazeFace -> YOLOv10n -> EfficientNet back to back, paced at ~30 Hz like the app, and prints
p50/p95/max per model. Usage: .venv/bin/python -I tools/ane_jitter.py [seconds]
"""
import sys
import time

import coremltools as ct
import numpy as np
from PIL import Image


def stats(name, values):
    v = np.sort(np.array(values))
    p = lambda q: v[min(len(v) - 1, int((len(v) - 1) * q + 0.5))]
    print(f"{name:10s} p50 {p(0.5):5.1f}  p95 {p(0.95):5.1f}  max {v[-1]:5.1f} ms  (n={len(v)}, >8 ms: {(v > 8).mean():.0%})")


def main(seconds: float) -> None:
    unit = ct.ComputeUnit.CPU_AND_NE
    face = ct.models.MLModel("App/Models/BlazeFace.mlpackage", compute_units=unit)
    gesture = ct.models.MLModel("App/Models/GestureNet.mlpackage", compute_units=unit)
    expr = ct.models.MLModel("App/Models/EmotionNet.mlpackage", compute_units=unit)
    rng = np.random.default_rng(0)
    img = lambda w, h: Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    feeds = [(face, {"image": img(256, 256)}), (gesture, {"image": img(640, 384)}), (expr, {"image": img(224, 224)})]
    for m, f in feeds:  # warm-up
        m.predict(f)

    times = {"face": [], "gesture": [], "expression": []}
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        frame_start = time.perf_counter()
        for (m, f), name in zip(feeds, times):
            t = time.perf_counter()
            m.predict(f)
            times[name].append((time.perf_counter() - t) * 1000)
        time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - frame_start)))
    for name, values in times.items():
        stats(name, values)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10)
