from __future__ import annotations

import numpy as np

from src.data.augmentation import TemporalAugmenter
from src.features.normalization import FEATURE_DIM, FEATURE_LANDMARKS


def test_augmentation_preserves_feature_dimensions() -> None:
    aug = TemporalAugmenter(frame_drop_prob=0.1, crop_ratio=0.8, jitter_std=0.01, resample_min=0.9, resample_max=1.1)
    x = np.zeros((12, FEATURE_DIM), dtype=np.float32)
    p = np.ones((12, FEATURE_LANDMARKS), dtype=bool)
    y, q = aug(x, p)
    assert y.ndim == 2
    assert q.ndim == 2
    assert y.shape[1] == FEATURE_DIM
    assert q.shape[1] == FEATURE_LANDMARKS
    assert y.shape[0] == q.shape[0]

