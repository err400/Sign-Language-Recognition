from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from src.features.extraction import _landmark_list_to_array
from src.features.normalization import normalize_landmark_frame
from src.inference.predict import PredictionSmoother, load_model_for_inference, predict_sequence


def run_webcam(
    checkpoint_path: str | Path,
    camera_index: int = 0,
    headless: bool = False,
    device_name: str = "auto",
) -> None:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Webcam inference requires opencv-python and mediapipe.") from exc

    model, idx_to_class, config, device = load_model_for_inference(checkpoint_path, device_name)
    sequence_length = int(config["features"]["sequence_length"])
    threshold = float(config["inference"].get("confidence_threshold", 0.45))
    top_k = int(config["evaluation"].get("top_k", 3))
    smoother = PredictionSmoother(int(config["inference"].get("smoothing_window", 5)))
    buffer: deque[np.ndarray] = deque(maxlen=sequence_length)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError(f"Could not open webcam index {camera_index}")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )
    with holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            norm = normalize_landmark_frame(
                pose=_landmark_list_to_array(result.pose_landmarks, 33),
                left_hand=_landmark_list_to_array(result.left_hand_landmarks, 21),
                right_hand=_landmark_list_to_array(result.right_hand_landmarks, 21),
            )
            buffer.append(norm.features)
            status = "Collecting frames"
            if len(buffer) == sequence_length:
                predictions = predict_sequence(model, np.stack(buffer), idx_to_class, device, top_k=top_k)
                best = predictions[0]
                status = "Unknown"
                if float(best["confidence"]) >= threshold:
                    status = smoother.update(str(best["label"]))
                if not headless:
                    for row, pred in enumerate(predictions[:3]):
                        cv2.putText(
                            frame,
                            f'{pred["label"]}: {float(pred["confidence"]):.2f}',
                            (10, 60 + row * 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
            if headless:
                if len(buffer) == sequence_length:
                    break
                continue
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Sign recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if not headless:
        cv2.destroyAllWindows()

