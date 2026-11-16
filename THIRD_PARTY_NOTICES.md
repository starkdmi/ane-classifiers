# Third-party notices

This repository's own code is MIT (see `LICENSE`). It builds on the work below, which keeps its own license.
**No model weights are stored in the repository**: `tools/setup.sh` downloads them from their authors and converts
them locally. Check each license before using the models, especially for anything commercial.

## Models

| Model | Used for | Source | License | Notes |
|---|---|---|---|---|
| EmotiEffLib `enet_b0_8_va_mtl` (EfficientNet-B0) | Classic: facial expression + valence/arousal | [sb-ai-lab/EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) | Apache-2.0 | Trained on AffectNet, a dataset licensed for research use; review AffectNet's terms before commercial use. |
| HaGRIDv2 `YOLOv10n_gestures` | Classic: hand boxes + 33 gestures | [hukenovs/hagrid](https://github.com/hukenovs/hagrid) | HaGRID license (a variant of CC BY-SA 4.0, see the repo's `license/`) | Trained with Ultralytics YOLOv10 (AGPL-3.0). Converted locally only; not redistributed. |
| MediaPipe BlazeFace (back-camera model) | Classic: face detection | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe), PyTorch port [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch) | Apache-2.0 | The port's network definition is copied into `tools/convert_blazeface.py`. |
| LiquidAI d1-2B, `LiquidAI/d1-2B-MLX-8bit` | d1 mode (MLX on the GPU) | [huggingface.co/LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) | LFM Open License v1.0 | Commercial use only below USD 10M annual revenue. `App/Models/d1_plan.json` is derived from this model (tokenizer ids, calibration) and is under the same license. |

## Code

| Component | Where | License |
|---|---|---|
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) `models/lfm3`, `models/lfm3_vl` (Copyright 2026 The HuggingFace Team, adapted from the LFM3 reference implementations) | Ported to Swift in `App/ANEDemo/D1/D1Model.swift` | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) `models/d1_lfm3_vl` (adapted from the system_one code shipped with the LiquidAI d1 checkpoints) | Ported to Swift in `App/ANEDemo/D1/D1Model.swift` and `D1Runner.swift`; called by `tools/d1_export_plan.py` | LFM Open License v1.0 (`App/ANEDemo/D1/LICENSE`) |
| [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch) network definition, Copyright (c) 2019 Matthijs Hollemans | Copied into `tools/convert_blazeface.py` | Apache-2.0 |
| [MLX Swift](https://github.com/ml-explore/mlx-swift) | Swift package dependency (d1 mode) | MIT |

## Tools (installed by `tools/setup.sh` into local virtual environments, not redistributed)

| Tool | License |
|---|---|
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), [MLX](https://github.com/ml-explore/mlx) | MIT |
| [coremltools](https://github.com/apple/coremltools) | BSD-3-Clause |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) (loads the HaGRID YOLOv10 checkpoint for conversion) | AGPL-3.0 |

## Credits

- The d1 comparison follows [ivanfioravanti/mlx-d1-gestures](https://github.com/ivanfioravanti/mlx-d1-gestures)
  (webcam → d1-2B with mlx-vlm): same model build, question types and input size. No code was copied from it.
- Apple's Vision, Core ML and AVFoundation frameworks (system components).
