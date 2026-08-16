from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.features.normalization import FEATURE_DIM, FEATURE_LANDMARKS
from src.inference.predict import predict_feature_file
from src.models.bilstm import BiLSTMSignClassifier
from src.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_save_load_and_feature_inference(tmp_path: Path) -> None:
    config = {
        "model": {
            "input_dim": FEATURE_DIM,
            "projection_dim": 8,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "num_classes": 2,
        },
        "features": {"sequence_length": 4},
        "evaluation": {"top_k": 2},
    }
    model = BiLSTMSignClassifier(FEATURE_DIM, 8, 8, 1, 0.0, 2)
    ckpt = tmp_path / "model.pt"
    save_checkpoint(ckpt, model, None, 0, {"hello": 0, "thanks": 1}, config)
    loaded = load_checkpoint(ckpt)
    assert loaded["class_to_idx"] == {"hello": 0, "thanks": 1}

    feature_file = tmp_path / "sample.npz"
    np.savez_compressed(
        feature_file,
        features=np.zeros((4, FEATURE_DIM), dtype=np.float32),
        presence_mask=np.ones((4, FEATURE_LANDMARKS), dtype=bool),
        label="hello",
        split="test",
        length=np.array(4),
        source="synthetic",
    )
    preds = predict_feature_file(ckpt, feature_file, top_k=2, device_name="cpu")
    assert len(preds) == 2
    assert {"hello", "thanks"} == {str(item["label"]) for item in preds}

