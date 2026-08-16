from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features.extraction import MediaPipeHolisticExtractor, cached_feature_path, extract_video_to_npz
from src.utils import save_json, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from a video manifest.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True, help="JSON list with path, label and split fields.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    extractor = MediaPipeHolisticExtractor(
        sequence_length=int(config["features"]["sequence_length"]),
        min_detection_confidence=float(config["features"].get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config["features"].get("min_tracking_confidence", 0.5)),
        mirror=bool(config["features"].get("mirror", False)),
    )
    processed = []
    failed = []
    for row in manifest:
        target = cached_feature_path(row["path"], config["data"]["processed_dir"], row["label"], row["split"])
        result = extract_video_to_npz(row["path"], target, row["label"], row["split"], extractor, args.overwrite)
        if result is None:
            failed.append(row)
        else:
            processed.append(str(result))
    save_json({"processed": processed, "failed": failed}, Path(config["data"]["processed_dir"]) / "preprocess_report.json")


if __name__ == "__main__":
    main()
