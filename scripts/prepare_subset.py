from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import prepare_generic_manifest, prepare_wlasl_subset_manifest
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a WLASL or generic video manifest.")
    sub = parser.add_subparsers(dest="mode", required=True)
    wlasl = sub.add_parser("wlasl")
    wlasl.add_argument("--json", required=True)
    wlasl.add_argument("--videos", required=True)
    wlasl.add_argument("--out", default="data/interim/wlasl_subset_manifest.json")
    wlasl.add_argument("--num-classes", type=int, default=20)
    wlasl.add_argument("--max-samples-per-class", type=int, default=None)
    wlasl.add_argument("--classes", nargs="*", default=None)

    generic = sub.add_parser("generic")
    generic.add_argument("--videos", required=True)
    generic.add_argument("--out", default="data/interim/generic_manifest.json")
    generic.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup_logging()
    if args.mode == "wlasl":
        prepare_wlasl_subset_manifest(
            args.json,
            args.videos,
            args.out,
            num_classes=args.num_classes,
            max_samples_per_class=args.max_samples_per_class,
            class_names=args.classes,
        )
    else:
        prepare_generic_manifest(args.videos, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
