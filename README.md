# Stylized Neural Subdivision Experiments

This repository is a cleaned, GitHub-ready code package for my stylized Neural Subdivision
experiments.

The goal is to follow the released Neural Subdivision implementation closely, train several
independent models on different style sources, and test the same coarse input object with each
trained model.

## Main Idea

Different training sources produce different learned subdivision behavior. In the experiment I used
four style sources:

- `bunny`: smooth organic style
- `balloon`: smooth round / sphere-like style
- `textured_earth`: rough bumpy style
- `gear_16t`: mechanical sharp-edged style

After training, the same coarse test object can be sent to all four models and the outputs can be
compared.

## Code Organization

```text
.
├── configs/
│   ├── styles.json
│   └── test_objects.json
├── data/
│   ├── style_sources/      # local OBJ style sources, not committed
│   └── test_objects/       # local OBJ test meshes, not committed
├── docs/
│   ├── normalization_fix.md
│   └── reproduction_notes.md
├── external/
│   ├── neuralSubdiv/       # released Neural Subdivision implementation
│   └── surface_multigrid_code/
│       └── 09_random_subdiv_remesh/
└── scripts/
    ├── build_cxx_generator.sh
    ├── generate_style_dataset.sh
    ├── make_style_pkls.py
    ├── write_style_hyperparams.py
    ├── train_style.sh
    ├── test_style.sh
    └── full_style_pipeline.sh
```

Generated datasets, PKLs, checkpoints, and rendered outputs are intentionally excluded from Git.

## Upstream Code

This package is based on:

- Neural Subdivision: <https://github.com/HTDerekLiu/neuralSubdiv>
- Surface multigrid / random subdivision remeshing generator:
  <https://github.com/HTDerekLiu/surface_multigrid_code>

The copied C++ generator includes one practical fix: it normalizes the input mesh to a unit bounding
box before remeshing, matching the MATLAB `normalizeUnitBox.m` behavior from the Neural Subdivision
data-generation path. See `docs/normalization_fix.md` and `docs/changes_from_upstream.md`.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch separately for your machine from the official PyTorch instructions. CUDA, CPU, and
Apple Silicon/MPS builds require different install commands.

Build the C++ generator:

```bash
./scripts/build_cxx_generator.sh
```

The build was checked on macOS with Apple Clang and current CMake. The first build may download
Eigen/GLFW/GLAD through libigl; those downloaded/build artifacts are ignored by Git.

## Train One Style Model

Place a style source OBJ, for example:

```text
data/style_sources/bunny.obj
```

Then run:

```bash
./scripts/full_style_pipeline.sh bunny data/style_sources/bunny.obj
```

This runs:

1. C++ generator build
2. normalized dataset generation
3. PKL conversion
4. hyperparameter writing
5. resumable training

For separate steps:

```bash
./scripts/generate_style_dataset.sh bunny data/style_sources/bunny.obj 200 10 500 2
python scripts/make_style_pkls.py bunny --num-train 200 --num-valid 10
python scripts/write_style_hyperparams.py bunny --epochs 700 --device cpu
./scripts/train_style.sh bunny
```

## Test One Object

After a model is trained:

```bash
./scripts/test_style.sh bunny data/test_objects/egg_chair.obj
```

Outputs are written to:

```text
external/neuralSubdiv/jobs/net_style_bunny/
```

## Notes

The repository does not include raw datasets or trained checkpoints. I keep those out of Git because
they are large generated artifacts. The codebase and generated experiment artifacts can be shared
separately when needed.
