from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils import ensure_dir


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    class_to_idx: dict[str, int],
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "class_to_idx": class_to_idx,
            "config": config,
            "metrics": metrics or {},
        },
        target,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)

