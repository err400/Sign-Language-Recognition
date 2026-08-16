from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
UPPER_BODY_POSE_INDICES = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
FEATURE_LANDMARKS = HAND_LANDMARK_COUNT * 2 + len(UPPER_BODY_POSE_INDICES)
FEATURE_DIM = FEATURE_LANDMARKS * 4
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
EPS = 1e-6


@dataclass(frozen=True)
class NormalizedFrame:
    features: np.ndarray
    presence: np.ndarray


def empty_landmarks(count: int) -> np.ndarray:
    """Return zero-filled x, y, z, visibility landmarks."""
    return np.zeros((count, 4), dtype=np.float32)


def _coerce_landmarks(values: np.ndarray | None, expected: int) -> tuple[np.ndarray, np.ndarray]:
    if values is None:
        return empty_landmarks(expected), np.zeros(expected, dtype=bool)
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape == (expected, 3):
        arr = np.concatenate([arr, np.ones((expected, 1), dtype=np.float32)], axis=1)
    if arr.shape != (expected, 4):
        raise ValueError(f"Expected landmarks with shape ({expected}, 3|4), got {arr.shape}")
    finite = np.isfinite(arr).all(axis=1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, finite


def normalize_landmark_frame(
    pose: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    mirror: bool = False,
) -> NormalizedFrame:
    """Normalize one frame relative to the shoulder midpoint and shoulder width.

    Coordinates are translated by the midpoint between left and right shoulders
    and divided by shoulder width. If either shoulder is unavailable or the
    distance is too small, a scale of 1.0 is used to avoid numeric explosions.
    Missing hands are represented as zero vectors and marked absent in the mask.
    """
    pose_arr, pose_present = _coerce_landmarks(pose, POSE_LANDMARK_COUNT)
    left_arr, left_present = _coerce_landmarks(left_hand, HAND_LANDMARK_COUNT)
    right_arr, right_present = _coerce_landmarks(right_hand, HAND_LANDMARK_COUNT)

    shoulders_ok = pose_present[LEFT_SHOULDER] and pose_present[RIGHT_SHOULDER]
    if shoulders_ok:
        left_shoulder = pose_arr[LEFT_SHOULDER, :3]
        right_shoulder = pose_arr[RIGHT_SHOULDER, :3]
        origin = (left_shoulder + right_shoulder) / 2.0
        scale = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
        scale = scale if scale > EPS else 1.0
    else:
        origin = np.zeros(3, dtype=np.float32)
        scale = 1.0

    pose_subset = pose_arr[UPPER_BODY_POSE_INDICES]
    pose_subset_present = pose_present[UPPER_BODY_POSE_INDICES]

    landmark_blocks = [left_arr, right_arr, pose_subset]
    presence_blocks = [left_present, right_present, pose_subset_present]
    landmarks = np.concatenate(landmark_blocks, axis=0).astype(np.float32)
    presence = np.concatenate(presence_blocks, axis=0)

    coords = (landmarks[:, :3] - origin.reshape(1, 3)) / scale
    vis = landmarks[:, 3:4]
    coords[~presence] = 0.0
    vis[~presence] = 0.0

    if mirror:
        coords[:, 0] *= -1.0
        left = coords[:HAND_LANDMARK_COUNT].copy()
        right = coords[HAND_LANDMARK_COUNT : HAND_LANDMARK_COUNT * 2].copy()
        coords[:HAND_LANDMARK_COUNT] = right
        coords[HAND_LANDMARK_COUNT : HAND_LANDMARK_COUNT * 2] = left
        left_v = vis[:HAND_LANDMARK_COUNT].copy()
        right_v = vis[HAND_LANDMARK_COUNT : HAND_LANDMARK_COUNT * 2].copy()
        vis[:HAND_LANDMARK_COUNT] = right_v
        vis[HAND_LANDMARK_COUNT : HAND_LANDMARK_COUNT * 2] = left_v
        left_p = presence[:HAND_LANDMARK_COUNT].copy()
        right_p = presence[HAND_LANDMARK_COUNT : HAND_LANDMARK_COUNT * 2].copy()
        presence[:HAND_LANDMARK_COUNT] = right_p
        presence[HAND_LANDMARK_COUNT : HAND_LANDMARK_COUNT * 2] = left_p

    features = np.concatenate([coords, vis], axis=1).reshape(-1).astype(np.float32)
    return NormalizedFrame(features=features, presence=presence.astype(bool))

