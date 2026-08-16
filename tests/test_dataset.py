from __future__ import annotations

import torch

from src.data.dataset import collate_landmark_sequences
from src.features.normalization import FEATURE_DIM, FEATURE_LANDMARKS


def test_padding_and_masks_for_unequal_lengths() -> None:
    batch = [
        {
            "features": torch.ones(3, FEATURE_DIM),
            "presence_mask": torch.ones(3, FEATURE_LANDMARKS, dtype=torch.bool),
            "length": 3,
            "label": torch.tensor(0),
            "label_name": "a",
            "path": "a.npz",
        },
        {
            "features": torch.ones(1, FEATURE_DIM) * 2,
            "presence_mask": torch.ones(1, FEATURE_LANDMARKS, dtype=torch.bool),
            "length": 1,
            "label": torch.tensor(1),
            "label_name": "b",
            "path": "b.npz",
        },
    ]
    collated = collate_landmark_sequences(batch)
    assert collated["features"].shape == (2, 3, FEATURE_DIM)
    assert collated["padding_mask"].tolist() == [[False, False, False], [False, True, True]]
    assert torch.all(collated["features"][1, 1:] == 0)

