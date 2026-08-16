from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features.synthetic import generate_synthetic_dataset
from src.inference.predict import predict_feature_file
from src.training.evaluate import evaluate_checkpoint
from src.training.train import train_one_run
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    generate_synthetic_dataset(
        config["data"]["processed_dir"],
        num_classes=int(config["model"]["num_classes"]),
        sequence_length=int(config["features"]["sequence_length"]),
        seed=int(config.get("seed", 7)),
    )
    train_result = train_one_run(config)
    checkpoint = train_result["best_checkpoint"]
    metrics = evaluate_checkpoint(checkpoint, split="test", output_dir=Path(config["training"]["run_dir"]) / "eval_test")
    first_npz = sorted((Path(config["data"]["processed_dir"]) / "test").glob("*/*.npz"))[0]
    preds = predict_feature_file(checkpoint, first_npz, top_k=3)
    print({"checkpoint": checkpoint, "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"], "sample": str(first_npz), "predictions": preds})


if __name__ == "__main__":
    main()
