from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    latencies: list[float] = []
    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        start = time.perf_counter()
        logits = model(features, lengths, padding_mask)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) / max(1, features.shape[0]))
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(batch["labels"].detach().cpu().numpy())
    return {
        "logits": np.concatenate(logits_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "average_latency_ms": float(np.mean(latencies) * 1000.0) if latencies else 0.0,
    }


def compute_metrics(logits: np.ndarray, labels: np.ndarray, class_names: list[str], top_k: int = 3) -> dict[str, Any]:
    preds = logits.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    macro = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    top_k = min(top_k, logits.shape[1])
    top_indices = np.argsort(logits, axis=1)[:, -top_k:]
    top_k_acc = float(np.mean([label in top_indices[row] for row, label in enumerate(labels)]))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "top_k_accuracy": top_k_acc,
        "per_class": {
            class_names[idx]: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx in range(len(class_names))
        },
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
    }

