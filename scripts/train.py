from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.training.train import train_one_run
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    setup_logging()
    result = train_one_run(load_config(args.config))
    print(result)


if __name__ == "__main__":
    main()
