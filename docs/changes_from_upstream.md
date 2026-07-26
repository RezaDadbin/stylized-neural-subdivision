# Changes From Upstream

This repository is mostly a cleaned packaging of the released Neural Subdivision code and its
official linked C++ remeshing generator.

## Added experiment scripts

- `scripts/build_cxx_generator.sh`
- `scripts/generate_style_dataset.sh`
- `scripts/make_style_pkls.py`
- `scripts/write_style_hyperparams.py`
- `scripts/train_style.sh`
- `scripts/test_style.sh`
- `scripts/full_style_pipeline.sh`

These scripts organize the reproduction pipeline. They do not replace the Neural Subdivision model
architecture.

## Added training reliability wrapper

- `external/neuralSubdiv/train_resume.py`

This keeps the original training logic but adds checkpoint/resume behavior and final validation
output writing.

## C++ generator normalization fix

- `external/surface_multigrid_code/09_random_subdiv_remesh/main.cpp`

The copied C++ generator now normalizes the loaded mesh before decimation, matching
`external/neuralSubdiv/utils_matlab/normalizeUnitBox.m`.

## Modern CMake compatibility

- `scripts/build_cxx_generator.sh` passes `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
- `external/surface_multigrid_code/libigl/cmake/DownloadProject.cmake` passes the same flag to
  nested dependency CMake calls.

This is only to make the older upstream CMake files build with current CMake versions.

## Intentionally not included

- `scaled_test.py`
- generated training meshes
- PKL datasets
- trained checkpoints
- rendered figures/results

The cleaned repository is meant to regenerate these artifacts from local dataset meshes.

