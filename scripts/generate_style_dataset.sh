#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 STYLE_NAME SOURCE_OBJ [NUM_TRAIN=200] [NUM_VALID=10] [TARGET_FACES=500] [NUM_SUBD=2]"
  echo "Example: $0 bunny data/style_sources/bunny.obj 200 10 500 2"
  exit 2
fi

STYLE="$1"
SOURCE_OBJ="$2"
NUM_TRAIN="${3:-200}"
NUM_VALID="${4:-10}"
TARGET_FACES="${5:-500}"
NUM_SUBD="${6:-2}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURAL="${ROOT}/external/neuralSubdiv"
GEN_DIR="${ROOT}/external/surface_multigrid_code/09_random_subdiv_remesh"
BIN="${GEN_DIR}/build/random_subdiv_remesh_bin"

if [[ ! -x "${BIN}" ]]; then
  echo "Generator binary not found. Run scripts/build_cxx_generator.sh first."
  exit 3
fi

SOURCE_ABS="$(cd "$(dirname "${SOURCE_OBJ}")" && pwd)/$(basename "${SOURCE_OBJ}")"
if [[ ! -f "${SOURCE_ABS}" ]]; then
  echo "Missing source OBJ: ${SOURCE_OBJ}"
  exit 4
fi

make_split() {
  local split_name="$1"
  local count="$2"
  local seed_offset="$3"
  local out_root="${NEURAL}/data_meshes/style_${STYLE}_${split_name}_${count}"

  if [[ "${split_name}" == "train" ]]; then
    out_root="${NEURAL}/data_meshes/style_${STYLE}_${count}"
  fi

  mkdir -p "${out_root}"
  for level in $(seq 0 "${NUM_SUBD}"); do
    mkdir -p "${out_root}/subd${level}"
  done

  echo
  echo "Generating ${split_name}: ${count} chains -> ${out_root}"
  for i in $(seq 1 "${count}"); do
    idx="$(printf "%03d" "${i}")"
    seed=$((seed_offset + i))
    echo "[${STYLE} ${split_name} ${idx}/${count}] seed=${seed}"
    (
      cd "${GEN_DIR}/build"
      ./random_subdiv_remesh_bin "${SOURCE_ABS}" "${TARGET_FACES}" "${NUM_SUBD}" "${seed}"
    )
    for level in $(seq 0 "${NUM_SUBD}"); do
      cp "${GEN_DIR}/output_s${level}.obj" "${out_root}/subd${level}/${idx}.obj"
    done
  done
}

echo "============================================================"
echo "Generating normalized Neural Subdivision style dataset"
echo "Style:        ${STYLE}"
echo "Source OBJ:   ${SOURCE_ABS}"
echo "Train/valid:  ${NUM_TRAIN}/${NUM_VALID}"
echo "Target faces: ${TARGET_FACES}"
echo "Subdivisions: ${NUM_SUBD}"
echo "============================================================"

make_split train "${NUM_TRAIN}" 0
make_split valid "${NUM_VALID}" 100000

echo
echo "Done. Generated folders:"
echo "- ${NEURAL}/data_meshes/style_${STYLE}_${NUM_TRAIN}"
echo "- ${NEURAL}/data_meshes/style_${STYLE}_valid_${NUM_VALID}"

