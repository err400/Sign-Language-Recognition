from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.webcam import run_webcam
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    setup_logging()
    run_webcam(args.checkpoint, args.camera_index, args.headless, args.device)


if __name__ == "__main__":
    main()
