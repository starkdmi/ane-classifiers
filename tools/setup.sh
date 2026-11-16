#!/bin/bash
# Fetches the original model checkpoints from their authors and converts them to Core ML for the app.
# Nothing converted is stored in git: each model keeps its own license (see THIRD_PARTY_NOTICES.md).
#
#   tools/setup.sh          classic pipeline: BlazeFace, EmotiEffLib, HaGRIDv2 YOLOv10n -> App/Models/*.mlpackage
#   tools/setup.sh --d1     also d1-2B (8-bit MLX, 2.2 GB) -> weights/d1-2B-MLX-8bit, and App/Models/d1_plan.json
#
# Needs Apple Silicon, macOS, and uv (https://docs.astral.sh/uv/). Safe to re-run: downloads are skipped
# when the files are already there.
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_D1=0
for arg in "$@"; do
  case "$arg" in
    --d1) WITH_D1=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)"; exit 2 ;;
  esac
done
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }

fetch() {  # fetch URL DEST: download once, atomically
  [ -s "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  echo "downloading $1"
  curl -fL --retry 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

echo "== conversion environment (.venv)"
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
# NumPy < 2.3: coremltools 9 relies on int() of 1-element arrays, which newer NumPy rejects.
VIRTUAL_ENV=.venv uv pip install -q torch torchvision coremltools timm ultralytics pillow "numpy<2.3"
PY=".venv/bin/python -I"

echo "== original checkpoints"
fetch "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/enet_b0_8_va_mtl.pt" \
      weights/emotiefflib/enet_b0_8_va_mtl.pt
fetch "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_gestures.pt" \
      weights/hagrid/YOLOv10n_gestures.pt
for f in blazefaceback.pth anchorsback.npy LICENSE; do
  fetch "https://raw.githubusercontent.com/hollance/BlazeFace-PyTorch/master/$f" "weights/blazeface/$f"
done

echo "== safety check: every global the pickled checkpoints reference (expect torch/timm/ultralytics only)"
$PY tools/inspect_pickle.py weights/emotiefflib/enet_b0_8_va_mtl.pt weights/hagrid/YOLOv10n_gestures.pt

echo "== converting to Core ML"
mkdir -p App/Models
$PY tools/convert_emotion.py weights/emotiefflib/enet_b0_8_va_mtl.pt App/Models/EmotionNet.mlpackage
$PY tools/convert_gestures.py weights/hagrid/YOLOv10n_gestures.pt App/Models/GestureNet.mlpackage
$PY tools/convert_blazeface.py back weights/blazeface App/Models/BlazeFace.mlpackage

if [ "$WITH_D1" = 1 ]; then
  echo "== d1-2B (mlx-vlm environment .venv-d1, 2.2 GB of weights)"
  [ -x .venv-d1/bin/python ] || uv venv --python 3.12 .venv-d1
  VIRTUAL_ENV=.venv-d1 uv pip install -q "mlx-vlm>=0.7.11" huggingface_hub
  # Plain HTTPS: the Xet transfer protocol can stall on some networks (set HF_HUB_DISABLE_XET=0 to use it).
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    .venv-d1/bin/hf download LiquidAI/d1-2B-MLX-8bit --local-dir weights/d1-2B-MLX-8bit
  .venv-d1/bin/python -I tools/d1_export_plan.py weights/d1-2B-MLX-8bit App/Models/d1_plan.json
fi

echo "done. Open App/ANEDemo.xcodeproj and run the ANEDemo scheme (or see README.md)."
