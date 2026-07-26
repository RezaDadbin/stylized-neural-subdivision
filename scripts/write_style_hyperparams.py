#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write NeuralSubdiv hyperparameters.json for one style.")
    parser.add_argument("style", help="Style name, e.g. bunny or gear_16t")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-subd", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    neural = root / "external/neuralSubdiv"
    job = neural / "jobs" / f"net_style_{args.style}"
    job.mkdir(parents=True, exist_ok=True)

    data = {
        "train_pkl": f"./data_PKL/style_{args.style}_train.pkl",
        "valid_pkl": f"./data_PKL/style_{args.style}_valid.pkl",
        "output_path": f"./jobs/net_style_{args.style}/",
        "epochs": args.epochs,
        "lr": args.lr,
        "device": args.device,
        "Din": 6,
        "Dout": 32,
        "h_initNet": [32, 32],
        "h_edgeNet": [32, 32],
        "h_vertexNet": [32, 32],
        "numSubd": args.num_subd,
    }

    path = job / "hyperparameters.json"
    path.write_text(json.dumps(data, indent=2))
    print("Wrote", path)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

