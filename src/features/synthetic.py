from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.normalization import FEATURE_DIM, FEATURE_LANDMARKS
from src.utils import ensure_dir, save_json


def generate_synthetic_dataset(
    output_dir: str | Path,
    num_classes: int = 3,
    samples_per_split: dict[str, int] | None = None,
    sequence_length: int = 24,
    seed: int = 7,
) -> None:
    """Create a tiny separable landmark dataset for smoke tests."""
    samples_per_split = samples_per_split or {"train": 18, "val": 6, "test": 6}
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    class_names = [f"synthetic_{idx}" for idx in range(num_classes)]

    for split, per_class in samples_per_split.items():
        for class_idx, class_name in enumerate(class_names):
            split_dir = ensure_dir(out / split / class_name)
            for sample_idx in range(per_class):
                length = int(rng.integers(max(8, sequence_length - 6), sequence_length + 1))
                t = np.linspace(0.0, 1.0, length, dtype=np.float32)
                features = rng.normal(0.0, 0.015, size=(length, FEATURE_DIM)).astype(np.float32)
                base = class_idx * 0.7
                phase = class_idx + 1
                features[:, 0] = base + np.sin(t * np.pi * phase)
                features[:, 1] = base + np.cos(t * np.pi * phase)
                features[:, 4] = base + t * (class_idx + 1)
                features[:, 8] = base - t * (class_idx + 1)
                # Duplicate the class-specific trajectory in pose columns so
                # samples with intentionally missing hands remain separable.
                features[:, 100] = class_idx * 2.0 + t
                features[:, 104] = class_idx * 2.0 - t
                features[:, 108] = class_idx * 2.0 + np.sin(t * np.pi)
                presence = np.ones((length, FEATURE_LANDMARKS), dtype=bool)
                if sample_idx % 5 == 0:
                    presence[:, :21] = False
                    features[:, :84] = 0.0
                np.savez_compressed(
                    split_dir / f"{class_name}_{sample_idx:03d}.npz",
                    features=features,
                    presence_mask=presence,
                    label=class_name,
                    split=split,
                    length=np.array(length, dtype=np.int64),
                    source="synthetic",
                )

    save_json({"classes": class_names}, out / "metadata.json")
