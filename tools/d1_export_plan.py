"""Compile a fixed d1 question set over a fixed image size into a "plan" for the Swift port, and dump
reference tensors for one test image.

The Swift side then needs no tokenizer or prompt code: with the questions fixed and every frame resized to
the same WxH, the packed token row is constant; only the image features change per frame.

Plan (JSON): token ids (image placeholders included), the state length, each token's question group, and per
question its answer slot(s), the readout tables (option -> its token forms), the calibration temperature, and
for boxes the presence calibration knots. The plan depends on the questions and the image size, not on the
image's content. Reference (safetensors, optional): pixel patches, vision+projector features, hidden rows at
the answer slots, and the final answers for one test image, for checking the Swift port with `d1check`.

The prompt construction and packing come from mlx-vlm's d1 code (LFM Open License v1.0); this script only
calls it. The questions mirror github.com/ivanfioravanti/mlx-d1-gestures (gesture, expression, two boxes).

Usage:
  .venv-d1/bin/python -I tools/d1_export_plan.py weights/d1-2B-MLX-8bit App/Models/d1_plan.json \
      [TEST_IMAGE weights/d1_reference.safetensors]
"""
import json
import sys
import time

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm import load
from mlx_vlm.models.d1_lfm3_vl import d1 as d1mod
from mlx_vlm.models.d1_lfm3_vl.media import media_list
from mlx_vlm.models.d1_lfm3_vl.prompt import (
    as_question,
    prefix_text,
    question_header,
    question_slots,
    readout_table,
    suffix_layout,
    typed,
)

# Multiples of patch (16) x merge (2), >= the 65,536-pixel minimum. 640x352 is what a 640x360 frame (the
# mlx-d1-gestures demo sends 640 px wide frames) becomes in the LFM3-VL processor: 40x22 patches, 220 image tokens.
WIDTH, HEIGHT = 640, 352

# Hand gesture + facial expression, as in the mlx-d1-gestures demo (wording and descriptions our own).
QUESTIONS = {
    "hand": {
        "type": "choice",
        "instructions": "Which hand gesture is the person making?",
        "criteria": {
            "thumbs_up": "Thumbs up",
            "thumbs_down": "Thumbs down",
            "ok": "OK sign",
            "peace": "Peace / victory sign",
            "open_palm": "Open palm",
            "pointing": "Pointing with one finger",
            "heart": "Heart shape made with hands or fingers",
            "fist": "Closed fist",
            "no_gesture": "No hand gesture, or no hand visible",
        },
    },
    "face": {
        "type": "choice",
        "instructions": "Which facial expression does the person have?",
        "criteria": {
            "smile": "Smiling",
            "sad": "Sad",
            "surprised": "Surprised",
            "thinking": "Thinking",
            "neutral": "Neutral",
            "no_face": "No face visible",
        },
    },
    # Where (for the overlay boxes and zoom tiles): a presence + four edges each.
    "face_box": {"type": "box", "instructions": "Where is the person's face?"},
    # One box: ask for the gesturing hand, so two visible hands don't blend into one wide box.
    "hand_box": {"type": "box", "instructions": "Where is the hand making the gesture?"},
}


def main(model_path: str, plan_path: str, image_path: str | None = None, reference_path: str | None = None) -> None:
    model, processor = load(model_path)
    engine = model.engine
    tok = engine.tokenizer
    if image_path:
        image = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT), Image.BICUBIC)
    else:  # the plan only needs the size
        image = Image.new("RGB", (WIDTH, HEIGHT), (128, 128, 128))

    names = list(QUESTIONS)
    qs = [as_question(QUESTIONS[n]) for n in names]
    if len(qs) > engine.set_size:
        raise SystemExit("more questions than one set; the plan assumes a single packed row")

    # --- the engine's _request, step by step (one group) ---
    markup, parts = engine.media.encode(media_list([image]))
    prefix = prefix_text(tok, None, engine.bos, engine.state_style, engine.system, markup)
    layouts = [suffix_layout(tok, q, engine.lead) for q in qs]
    placed = [engine.media.place(text, q) for (text, _), q in zip(layouts, qs)]
    header = question_header(tok, None)
    suffixes = [text for text, _ in placed]
    first_ids = tok.encode(prefix + suffixes[0], add_special_tokens=False)
    ids, state_len, question, ends = d1mod.pack_set(
        tok, prefix, suffixes, header, [offsets for _, offsets in layouts], first_ids
    )

    part = parts[0]
    grid = np.asarray(part["image_grid_thw"]).reshape(-1).tolist()
    image_tokens = int(np.prod(grid) // 4)
    placeholders = int(np.sum(np.asarray(ids) == model.config.image_token_id))
    assert placeholders == image_tokens, (placeholders, image_tokens)

    # Readout tables (tokens + each option's forms), shared between slots that read the same table.
    tables, table_index = [], {}

    def table_id(readout):
        tokens, forms, _ = readout_table(tok, readout)
        key = (tuple(tokens), np.asarray(forms).tobytes())
        if key not in table_index:
            table_index[key] = len(tables)
            tables.append({"tokens": [int(t) for t in tokens], "forms": np.asarray(forms).astype(int).tolist()})
        return table_index[key]

    calibration = engine.calibration
    plan_questions = []
    for name, q, slots in zip(names, qs, ends):
        readouts = list(question_slots(q, tok))
        assert len(readouts) == len(slots)
        tempered = calibration is not None and not typed(q)
        entry = {
            "name": name,
            "kind": QUESTIONS[name]["type"],
            "slots": [
                {"at": int(at), "table": table_id(r),
                 "temperature": float(calibration.temperature(r)) if tempered else 1.0}
                for r, at in zip(readouts, slots)
            ],
        }
        if entry["kind"] == "choice":
            entry["labels"] = list(q.criteria.keys())
            entry["titles"] = [QUESTIONS[name]["criteria"][k] for k in q.criteria.keys()]
        else:  # box: presence (yes/no) then top, bottom, left, right over the thousandths grid
            entry["found_knots"] = (calibration.tables.get("found:box") if calibration else None) or []
            entry["grid"] = {"low": 0.0, "high": 0.999, "size": 1000}
            entry["edges"] = getattr(tok, "d1_box_edges", None) or "median"
            assert len(tables[entry["slots"][1]["table"]]["forms"]) == 1000
        plan_questions.append(entry)

    plan = {
        "model": "LiquidAI/d1-2B (MLX 8-bit)",
        "image": {
            "width": WIDTH, "height": HEIGHT, "grid_thw": grid,
            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
            "patch": 16, "temporal": 2, "merge": 2,
        },
        "image_token_id": int(model.config.image_token_id),
        "sliding_window": int(model.config.text_config.sliding_window),
        "ids": [int(i) for i in ids],
        "state_len": int(state_len),
        "question": [int(g) for g in question],
        "tables": tables,
        "questions": plan_questions,
    }
    with open(plan_path, "w") as f:
        json.dump(plan, f)
    print(f"plan: {len(ids)} tokens ({image_tokens} image), state {state_len}, grid {grid}, {len(tables)} tables, "
          f"questions {[(q['name'], q['kind'], len(q['slots'])) for q in plan_questions]}")

    if not (image_path and reference_path):
        return

    # --- reference tensors ---
    pixel_values = mx.array(part["pixel_values"])
    token, features = model.d1_features(part)
    hidden = model.d1_hidden(ids, state_len, question, [(token, features)])
    slots = mx.array([s["at"] for q in plan_questions for s in q["slots"]])
    tensors = {
        "pixel_values": pixel_values.astype(mx.float32),
        "features": features.astype(mx.float32),
        "hidden_slots": hidden[slots].astype(mx.float32),
        "image_rgb": mx.array(np.asarray(image, dtype=np.uint8)),
    }
    result = model.system_one(None, QUESTIONS, media=[image])
    for name in names:
        answer = result["answers"][name]
        if QUESTIONS[name]["type"] == "choice":
            probs = answer["probabilities"]
            tensors[f"probs_{name}"] = mx.array([probs[k] for k in QUESTIONS[name]["criteria"]], dtype=mx.float32)
            top = max(probs, key=probs.get)
            print(f"{name}: {top} {probs[top]:.3f}  " + ", ".join(f"{k} {v:.3f}" for k, v in probs.items()))
        else:
            box = answer["box"] or {}
            tensors[f"box_{name}"] = mx.array(
                [answer["found"]] + [box.get(e, -1.0) for e in ("top", "bottom", "left", "right")], dtype=mx.float32)
            print(f"{name}: found {answer['found']:.3f} box {box}")
    mx.save_safetensors(reference_path, tensors)
    print("usage:", result["usage"])

    # --- d1 speed on this Mac (warm, whole system_one incl. image preprocessing) ---
    times = []
    for _ in range(6):
        start = time.perf_counter()
        model.system_one(None, QUESTIONS, media=[image])
        times.append((time.perf_counter() - start) * 1000)
    times = sorted(times[1:])
    print(f"system_one (mlx-vlm, Python): median {times[len(times) // 2]:.0f} ms, min {times[0]:.0f} ms")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        raise SystemExit(__doc__)
    main(*sys.argv[1:5])
