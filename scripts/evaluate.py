from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.evaluate import evaluate_checkpoint
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    setup_logging()
    metrics = evaluate_checkpoint(args.checkpoint, args.split, args.output_dir, args.device)
    print(metrics)


if __name__ == "__main__":
    main()
