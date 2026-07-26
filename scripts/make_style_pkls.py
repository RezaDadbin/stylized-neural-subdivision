#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create NeuralSubdiv train/valid PKLs for one style.")
    parser.add_argument("style", help="Style name, e.g. bunny or gear_16t")
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-valid", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    neural = root / "external/neuralSubdiv"
    os.chdir(neural)
    sys.path.insert(0, str(neural))

    from include import TrainMeshes

    train_folder = f"./data_meshes/style_{args.style}_{args.num_train}/"
    valid_folder = f"./data_meshes/style_{args.style}_valid_{args.num_valid}/"
    out_dir = Path("./data_PKL")
    out_dir.mkdir(exist_ok=True)

    print("============================================================")
    print("Creating PKLs")
    print("style:", args.style)
    print("train folder:", train_folder)
    print("valid folder:", valid_folder)
    print("============================================================")

    train = TrainMeshes([train_folder])
    valid = TrainMeshes([valid_folder])

    train_pkl = out_dir / f"style_{args.style}_train.pkl"
    valid_pkl = out_dir / f"style_{args.style}_valid.pkl"
    pickle.dump(train, open(train_pkl, "wb"))
    pickle.dump(valid, open(valid_pkl, "wb"))

    print("Wrote", train_pkl)
    print("Wrote", valid_pkl)


if __name__ == "__main__":
    main()
