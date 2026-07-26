#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 STYLE_NAME SOURCE_OBJ"
  echo "Example: $0 bunny data/style_sources/bunny.obj"
  exit 2
fi

STYLE="$1"
SOURCE_OBJ="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT}/scripts/build_cxx_generator.sh"
"${ROOT}/scripts/generate_style_dataset.sh" "${STYLE}" "${SOURCE_OBJ}" 200 10 500 2
python "${ROOT}/scripts/make_style_pkls.py" "${STYLE}" --num-train 200 --num-valid 10
python "${ROOT}/scripts/write_style_hyperparams.py" "${STYLE}" --epochs 700 --device cpu
"${ROOT}/scripts/train_style.sh" "${STYLE}"

