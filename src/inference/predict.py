from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.features.extraction import MediaPipeHolisticExtractor
from src.models.bilstm import BiLSTMSignClassifier
from src.training.checkpoint import load_checkpoint
from src.utils import select_device


def load_model_for_inference(checkpoint_path: str | Path, device_name: str = "auto") -> tuple[BiLSTMSignClassifier, list[str], dict[str, Any], torch.device]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    model_cfg = dict(config["model"])
    model_cfg["num_classes"] = len(class_to_idx)
    device = select_device(device_name)
    model = BiLSTMSignClassifier(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, idx_to_class, config, device


@torch.no_grad()
def predict_sequence(
    model: BiLSTMSignClassifier,
    features: np.ndarray,
    idx_to_class: list[str],
    device: torch.device,
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    padding_mask = torch.zeros((1, features.shape[0]), dtype=torch.bool, device=device)
    probs = torch.softmax(model(x, lengths, padding_mask), dim=1).squeeze(0).detach().cpu().numpy()
    order = np.argsort(probs)[::-1][: min(top_k, len(idx_to_class))]
    return [{"label": idx_to_class[idx], "confidence": float(probs[idx])} for idx in order]


def predict_feature_file(checkpoint_path: str | Path, feature_npz: str | Path, top_k: int = 3, device_name: str = "auto") -> list[dict[str, float | str]]:
    model, idx_to_class, _, device = load_model_for_inference(checkpoint_path, device_name)
    with np.load(feature_npz, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
    return predict_sequence(model, features, idx_to_class, device, top_k=top_k)


def predict_video_file(checkpoint_path: str | Path, video_path: str | Path, top_k: int = 3, device_name: str = "auto") -> list[dict[str, float | str]]:
    model, idx_to_class, config, device = load_model_for_inference(checkpoint_path, device_name)
    feat_cfg = config["features"]
    extractor = MediaPipeHolisticExtractor(
        sequence_length=int(feat_cfg["sequence_length"]),
        min_detection_confidence=float(feat_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(feat_cfg.get("min_tracking_confidence", 0.5)),
        mirror=bool(feat_cfg.get("mirror", False)),
    )
    features, _, _ = extractor.extract(video_path)
    return predict_sequence(model, features, idx_to_class, device, top_k=top_k)


class PredictionSmoother:
    def __init__(self, window: int = 5) -> None:
        self.labels: deque[str] = deque(maxlen=window)

    def update(self, label: str) -> str:
        self.labels.append(label)
        counts = {item: list(self.labels).count(item) for item in set(self.labels)}
        return max(counts, key=counts.get)

