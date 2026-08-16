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

from src.data.augmentation import TemporalAugmenter
from src.data.dataset import (
    LandmarkSequenceDataset,
    build_class_mapping,
    collate_landmark_sequences,
    discover_npz_samples,
    make_weighted_sampler,
)
from src.models.bilstm import BiLSTMSignClassifier
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.metrics import compute_metrics, predict_loader
from src.utils import ensure_dir, save_json, seed_everything, select_device


def _make_augmenter(config: dict[str, Any]) -> TemporalAugmenter | None:
    aug = config.get("augmentation", {})
    if not aug.get("enabled", False):
        return None
    return TemporalAugmenter(
        frame_drop_prob=float(aug.get("frame_drop_prob", 0.0)),
        crop_ratio=float(aug.get("crop_ratio", 1.0)),
        jitter_std=float(aug.get("jitter_std", 0.0)),
        resample_min=float(aug.get("resample_min", 1.0)),
        resample_max=float(aug.get("resample_max", 1.0)),
        mirror_prob=float(aug.get("mirror_prob", 0.0)),
    )


def make_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    data_cfg = config["data"]
    root = data_cfg["processed_dir"]
    train_samples = discover_npz_samples(root, "train")
    val_samples = discover_npz_samples(root, "val")
    class_to_idx = build_class_mapping(train_samples + val_samples)
    train_ds = LandmarkSequenceDataset(root, "train", class_to_idx, augmenter=_make_augmenter(config))
    val_ds = LandmarkSequenceDataset(root, "val", class_to_idx, augmenter=None)
    sampler = make_weighted_sampler(train_ds) if data_cfg.get("weighted_sampling", False) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_landmark_sequences,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_landmark_sequences,
    )
    return train_loader, val_loader, class_to_idx


def build_model(config: dict[str, Any], num_classes: int) -> BiLSTMSignClassifier:
    model_cfg = dict(config["model"])
    model_cfg["num_classes"] = num_classes
    return BiLSTMSignClassifier(**model_cfg)


def _class_weights(loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = [loader.dataset.class_to_idx[sample.label] for sample in loader.dataset.samples]  # type: ignore[attr-defined]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_run(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))
    train_loader, val_loader, class_to_idx = make_loaders(config)
    idx_to_class = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    model = build_model(config, len(class_to_idx)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    if config["training"].get("class_weighted_loss", False):
        criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(train_loader, len(class_to_idx), device))
    else:
        criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    resume = config["training"].get("resume")
    if resume:
        checkpoint = load_checkpoint(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    best_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    run_dir = ensure_dir(config["training"]["run_dir"])
    checkpoint_dir = ensure_dir(config["training"]["checkpoint_dir"])
    best_path = checkpoint_dir / "best.pt"

    for epoch in range(start_epoch, int(config["training"].get("epochs", 1))):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["features"].to(device),
                batch["lengths"].to(device),
                batch["padding_mask"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        pred = predict_loader(model, val_loader, device)
        metrics = compute_metrics(pred["logits"], pred["labels"], idx_to_class, int(config["evaluation"].get("top_k", 3)))
        scheduler.step(metrics["macro_f1"])
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_accuracy": float(metrics["accuracy"]),
            "val_macro_f1": float(metrics["macro_f1"]),
        }
        history.append(row)
        save_json({"history": history}, run_dir / "history.json")

        if metrics["macro_f1"] > best_f1:
            best_f1 = float(metrics["macro_f1"])
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, class_to_idx, config, metrics)
        else:
            stale_epochs += 1
        if stale_epochs >= int(config["training"].get("patience", 5)):
            break

    if history:
        plt.figure(figsize=(6, 4))
        plt.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train_loss")
        plt.plot([h["epoch"] for h in history], [h["val_macro_f1"] for h in history], label="val_macro_f1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "training_curves.png")
        plt.close()

    return {"best_checkpoint": str(best_path), "history": history, "class_to_idx": class_to_idx}
