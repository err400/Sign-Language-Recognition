from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predict import predict_feature_file, predict_video_file
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--features-npz", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    setup_logging()
    if bool(args.video) == bool(args.features_npz):
        raise SystemExit("Provide exactly one of --video or --features-npz.")
    if args.features_npz:
        preds = predict_feature_file(args.checkpoint, args.features_npz, args.top_k, args.device)
    else:
        preds = predict_video_file(args.checkpoint, args.video, args.top_k, args.device)
    print(preds)


if __name__ == "__main__":
    main()
