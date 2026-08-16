from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from src.utils import ensure_dir, save_json

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def prepare_generic_manifest(
    video_dir: str | Path,
    output_manifest: str | Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> list[dict[str, str]]:
    """Create a stratified manifest from class-named video directories."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for class_dir in sorted(Path(video_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        files = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTENSIONS]
        rng.shuffle(files)
        n_train = max(1, int(len(files) * train_ratio)) if files else 0
        n_val = int(len(files) * val_ratio)
        for idx, path in enumerate(files):
            split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            rows.append({"path": str(path), "label": class_dir.name, "split": split})
    save_json(rows, output_manifest)
    return rows


def prepare_wlasl_subset_manifest(
    wlasl_json: str | Path,
    video_dir: str | Path,
    output_manifest: str | Path,
    num_classes: int = 20,
    max_samples_per_class: int | None = None,
    class_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """Create a WLASL subset manifest while preserving official split labels."""
    with Path(wlasl_json).open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    selected = set(class_names or [entry["gloss"] for entry in entries[:num_classes]])
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    video_root = Path(video_dir)
    for entry in entries:
        gloss = entry["gloss"]
        if gloss not in selected:
            continue
        for inst in entry.get("instances", []):
            video_id = inst.get("video_id")
            split = inst.get("split", "train")
            matches = list(video_root.glob(f"{video_id}.*"))
            if matches:
                by_class[gloss].append({"path": str(matches[0]), "label": gloss, "split": split})

    rows: list[dict[str, str]] = []
    for gloss in sorted(by_class):
        class_rows = by_class[gloss]
        if max_samples_per_class is not None:
            class_rows = class_rows[:max_samples_per_class]
        rows.extend(class_rows)
    save_json(rows, output_manifest)
    return rows


def copy_manifest_videos_to_split_dirs(manifest: list[dict[str, str]], output_dir: str | Path) -> None:
    """Optional helper for making a generic class/split video directory."""
    out = Path(output_dir)
    for row in manifest:
        src = Path(row["path"])
        dst = out / row["split"] / row["label"] / src.name
        ensure_dir(dst.parent)
        if not dst.exists():
            shutil.copy2(src, dst)

