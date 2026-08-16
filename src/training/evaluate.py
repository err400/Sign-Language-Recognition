from __future__ import annotations

from pathlib import Path
from typing import Any

import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib-cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LandmarkSequenceDataset, collate_landmark_sequences
from src.models.bilstm import BiLSTMSignClassifier
from src.training.checkpoint import load_checkpoint
from src.training.metrics import compute_metrics, predict_loader
from src.utils import ensure_dir, save_json, select_device


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str = "test",
    output_dir: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    dataset = LandmarkSequenceDataset(config["data"]["processed_dir"], split, class_to_idx, augmenter=None)
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collate_landmark_sequences,
    )
    device = select_device(device_name)
    model_cfg = dict(config["model"])
    model_cfg["num_classes"] = len(class_to_idx)
    model = BiLSTMSignClassifier(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    pred = predict_loader(model, loader, device)
    metrics = compute_metrics(pred["logits"], pred["labels"], idx_to_class, int(config["evaluation"].get("top_k", 3)))
    metrics["average_latency_ms"] = pred["average_latency_ms"]

    out = ensure_dir(output_dir or Path(config["training"]["run_dir"]) / f"eval_{split}")
    save_json(metrics, out / "metrics.json")
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.float32)
    plt.figure(figsize=(max(5, len(idx_to_class) * 0.4), max(4, len(idx_to_class) * 0.4)))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(idx_to_class)), idx_to_class, rotation=90)
    plt.yticks(range(len(idx_to_class)), idx_to_class)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png")
    plt.close()
    return metrics
