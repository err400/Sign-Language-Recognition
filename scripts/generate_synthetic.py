from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features.synthetic import generate_synthetic_dataset
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


if __name__ == "__main__":
    main()
