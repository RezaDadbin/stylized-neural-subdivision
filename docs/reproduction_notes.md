# Reproduction Notes

This repository is organized to reproduce the stylized Neural Subdivision experiment without
committing generated data.

## What is included

- Official Neural Subdivision code under `external/neuralSubdiv`.
- Official linked surface multigrid / random subdivision remeshing code under
  `external/surface_multigrid_code`.
- A small training wrapper, `train_resume.py`, for checkpoint/resume reliability.
- Scripts for dataset generation, PKL conversion, hyperparameter writing, training, and testing.
- A C++ normalization fix in the copied generator path.

## What is not included

- Raw mesh datasets.
- Generated `data_meshes`.
- PKL files.
- Trained checkpoints.
- Rendered experiment results.

These are intentionally ignored because they are generated artifacts or dataset files.

## Style sources used in the visual experiment

- Bunny: smooth organic style.
- Balloon: smooth sphere-like style.
- Textured Earth: rough/bumpy style.
- Gear16: mechanical sharp-edged style.

## Test objects used in the visual experiment

Egg chair, dragon, sphere, elephant, fish, bunny, gear16, balloon, crane hook,
unicorn head, rock frog, and brain slug.

