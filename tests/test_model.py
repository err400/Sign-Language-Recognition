from __future__ import annotations

import torch

from src.features.normalization import FEATURE_DIM
from src.models.bilstm import BiLSTMSignClassifier


def test_bilstm_forward_unequal_lengths() -> None:
    model = BiLSTMSignClassifier(FEATURE_DIM, 16, 12, 1, 0.0, 4)
    x = torch.randn(2, 5, FEATURE_DIM)
    lengths = torch.tensor([5, 3])
    mask = torch.tensor([[False, False, False, False, False], [False, False, False, True, True]])
    logits = model(x, lengths, mask)
    assert logits.shape == (2, 4)


def test_padded_timesteps_do_not_change_prediction_for_valid_sequence() -> None:
    torch.manual_seed(0)
    model = BiLSTMSignClassifier(FEATURE_DIM, 16, 12, 1, 0.0, 4)
    model.eval()
    valid = torch.randn(1, 3, FEATURE_DIM)
    padded_a = torch.cat([valid, torch.zeros(1, 2, FEATURE_DIM)], dim=1)
    padded_b = padded_a.clone()
    padded_b[:, 3:] = 1000.0
    lengths = torch.tensor([3])
    mask = torch.tensor([[False, False, False, True, True]])
    with torch.no_grad():
        out_a = model(padded_a, lengths, mask)
        out_b = model(padded_b, lengths, mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)

