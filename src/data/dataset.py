from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.augmentation import TemporalAugmenter


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    split: str


def discover_npz_samples(root: str | Path, split: str) -> list[Sample]:
    base = Path(root) / split
    samples: list[Sample] = []
    if not base.exists():
        return samples
    for path in sorted(base.glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            label = str(data["label"].item() if data["label"].shape == () else data["label"])
        samples.append(Sample(path=path, label=label, split=split))
    return samples


def build_class_mapping(samples: list[Sample]) -> dict[str, int]:
    labels = sorted({sample.label for sample in samples})
    return {label: idx for idx, label in enumerate(labels)}


class LandmarkSequenceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        class_to_idx: dict[str, int] | None = None,
        augmenter: TemporalAugmenter | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.samples = discover_npz_samples(root, split)
        if not self.samples:
            raise ValueError(f"No .npz samples found under {self.root / split}")
        self.class_to_idx = class_to_idx or build_class_mapping(self.samples)
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        with np.load(sample.path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32)
            presence = data["presence_mask"].astype(bool)
            label = str(data["label"].item() if data["label"].shape == () else data["label"])
        if self.augmenter is not None:
            features, presence = self.augmenter(features, presence)
        return {
            "features": torch.from_numpy(features),
            "presence_mask": torch.from_numpy(presence),
            "length": int(features.shape[0]),
            "label": torch.tensor(self.class_to_idx[label], dtype=torch.long),
            "label_name": label,
            "path": str(sample.path),
        }


def collate_landmark_sequences(batch: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feature_dim = int(batch[0]["features"].shape[1])  # type: ignore[index, union-attr]
    landmark_dim = int(batch[0]["presence_mask"].shape[1])  # type: ignore[index, union-attr]
    features = torch.zeros((len(batch), max_len, feature_dim), dtype=torch.float32)
    presence = torch.zeros((len(batch), max_len, landmark_dim), dtype=torch.bool)
    padding_mask = torch.ones((len(batch), max_len), dtype=torch.bool)
    labels = torch.stack([item["label"] for item in batch])  # type: ignore[list-item]

    for row, item in enumerate(batch):
        length = int(item["length"])
        features[row, :length] = item["features"]  # type: ignore[index]
        presence[row, :length] = item["presence_mask"]  # type: ignore[index]
        padding_mask[row, :length] = False

    return {
        "features": features,
        "presence_mask": presence,
        "lengths": lengths,
        "padding_mask": padding_mask,
        "labels": labels,
        "paths": [str(item["path"]) for item in batch],
        "label_names": [str(item["label_name"]) for item in batch],
    }


def make_weighted_sampler(dataset: LandmarkSequenceDataset) -> WeightedRandomSampler:
    labels = [dataset.class_to_idx[sample.label] for sample in dataset.samples]
    counts = np.bincount(labels, minlength=len(dataset.class_to_idx)).astype(np.float32)
    weights = [1.0 / max(counts[label], 1.0) for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

