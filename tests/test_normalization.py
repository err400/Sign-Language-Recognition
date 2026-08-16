from __future__ import annotations

import numpy as np

from src.features.normalization import FEATURE_DIM, FEATURE_LANDMARKS, normalize_landmark_frame


def test_landmark_normalization_uses_shoulder_midpoint_and_scale() -> None:
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[11, :3] = [0.25, 0.5, 0.0]
    pose[12, :3] = [0.75, 0.5, 0.0]
    left = np.zeros((21, 4), dtype=np.float32)
    left[:, 3] = 1.0
    left[0, :3] = [0.75, 0.5, 0.0]
    result = normalize_landmark_frame(pose, left, None)
    assert result.features.shape == (FEATURE_DIM,)
    assert result.presence.shape == (FEATURE_LANDMARKS,)
    assert np.isclose(result.features[0], 0.5)
    assert np.isclose(result.features[1], 0.0)


def test_missing_landmarks_are_zero_and_absent() -> None:
    result = normalize_landmark_frame(None, None, None)
    assert np.all(result.features == 0.0)
    assert not result.presence.any()

