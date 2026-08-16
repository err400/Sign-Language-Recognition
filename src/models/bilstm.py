from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMSignClassifier(nn.Module):
    """Bidirectional LSTM classifier for variable-length landmark sequences."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        pooled_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengths_cpu = lengths.detach().to("cpu").clamp_min(1)
        projected = self.projection(features)
        packed = pack_padded_sequence(projected, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=features.shape[1])

        if padding_mask is None:
            time = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
            padding_mask = time >= lengths.to(features.device).unsqueeze(1)

        valid = ~padding_mask
        valid_float = valid.unsqueeze(-1).to(output.dtype)
        denom = valid_float.sum(dim=1).clamp_min(1.0)
        mean_pool = (output * valid_float).sum(dim=1) / denom
        masked = output.masked_fill(~valid.unsqueeze(-1), torch.finfo(output.dtype).min)
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(pooled)

