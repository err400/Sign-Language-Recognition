from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from src.features.normalization import FEATURE_DIM, FEATURE_LANDMARKS, normalize_landmark_frame
from src.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def sample_frame_indices(total_frames: int, target_frames: int) -> np.ndarray:
    """Evenly sample frame indices from a video."""
    if total_frames <= 0:
        return np.array([], dtype=np.int64)
    target = min(total_frames, target_frames)
    return np.linspace(0, total_frames - 1, num=target, dtype=np.int64)


def _landmark_list_to_array(landmarks: object | None, expected: int) -> np.ndarray | None:
    if landmarks is None:
        return None
    values = getattr(landmarks, "landmark", None)
    if values is None:
        return None
    arr = np.zeros((expected, 4), dtype=np.float32)
    for idx, lm in enumerate(values[:expected]):
        arr[idx, 0] = float(getattr(lm, "x", 0.0))
        arr[idx, 1] = float(getattr(lm, "y", 0.0))
        arr[idx, 2] = float(getattr(lm, "z", 0.0))
        arr[idx, 3] = float(getattr(lm, "visibility", 1.0))
    return arr


class MediaPipeHolisticExtractor:
    """Extract normalized holistic landmarks from a video file."""

    def __init__(
        self,
        sequence_length: int,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mirror: bool = False,
    ) -> None:
        self.sequence_length = sequence_length
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mirror = mirror

    def extract(self, video_path: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Video extraction requires opencv-python and mediapipe.") from exc

        path = Path(video_path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        wanted = set(sample_frame_indices(total, self.sequence_length).tolist())
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        with holistic:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in wanted:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = holistic.process(rgb)
                    norm = normalize_landmark_frame(
                        pose=_landmark_list_to_array(result.pose_landmarks, 33),
                        left_hand=_landmark_list_to_array(result.left_hand_landmarks, 21),
                        right_hand=_landmark_list_to_array(result.right_hand_landmarks, 21),
                        mirror=self.mirror,
                    )
                    frames.append(norm.features)
                    masks.append(norm.presence)
                idx += 1
        cap.release()

        if not frames:
            raise ValueError(f"No readable frames extracted from video: {path}")
        features = np.stack(frames).astype(np.float32)
        presence = np.stack(masks).astype(bool)
        return features, presence, int(features.shape[0])


def cached_feature_path(video_path: str | Path, output_dir: str | Path, label: str, split: str) -> Path:
    digest = hashlib.sha1(str(Path(video_path).resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(output_dir) / split / label / f"{Path(video_path).stem}_{digest}.npz"


def extract_video_to_npz(
    video_path: str | Path,
    output_path: str | Path,
    label: str,
    split: str,
    extractor: MediaPipeHolisticExtractor,
    overwrite: bool = False,
) -> Path | None:
    """Extract one video to a compressed NumPy feature file."""
    target = Path(output_path)
    if target.exists() and not overwrite:
        return target
    try:
        features, presence, length = extractor.extract(video_path)
        if features.shape[1] != FEATURE_DIM or presence.shape[1] != FEATURE_LANDMARKS:
            raise ValueError("Unexpected feature or presence dimensionality.")
        ensure_dir(target.parent)
        np.savez_compressed(
            target,
            features=features,
            presence_mask=presence,
            label=label,
            split=split,
            length=np.array(length, dtype=np.int64),
            source=str(video_path),
        )
        return target
    except Exception as exc:  # noqa: BLE001 - preprocessing should continue over bad videos.
        LOGGER.warning("Failed to extract %s: %s", video_path, exc)
        return None

