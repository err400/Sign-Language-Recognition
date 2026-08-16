from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalAugmenter:
    frame_drop_prob: float = 0.0
    crop_ratio: float = 1.0
    jitter_std: float = 0.0
    resample_min: float = 1.0
    resample_max: float = 1.0
    mirror_prob: float = 0.0

    def __call__(self, features: np.ndarray, presence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = features.copy()
        p = presence.copy()
        if len(x) == 0:
            return x, p

        if self.frame_drop_prob > 0 and len(x) > 2:
            keep = np.random.random(len(x)) > self.frame_drop_prob
            if keep.sum() >= 2:
                x, p = x[keep], p[keep]

        if 0 < self.crop_ratio < 1.0 and len(x) > 2:
            target = max(2, int(round(len(x) * self.crop_ratio)))
            if target < len(x):
                start = np.random.randint(0, len(x) - target + 1)
                x, p = x[start : start + target], p[start : start + target]

        if self.resample_min > 0 and self.resample_max > 0 and len(x) > 2:
            factor = float(np.random.uniform(self.resample_min, self.resample_max))
            target = max(2, int(round(len(x) * factor)))
            if target != len(x):
                old = np.linspace(0.0, 1.0, len(x))
                new = np.linspace(0.0, 1.0, target)
                x = np.stack([np.interp(new, old, x[:, dim]) for dim in range(x.shape[1])], axis=1)
                indices = np.clip(np.round(new * (len(p) - 1)).astype(int), 0, len(p) - 1)
                p = p[indices]

        if self.jitter_std > 0:
            noise = np.random.normal(0.0, self.jitter_std, size=x.shape).astype(np.float32)
            # Coordinates are the first three values of every landmark tuple; visibility is left unchanged.
            coord_cols = np.ones(x.shape[1], dtype=bool)
            coord_cols[3::4] = False
            x[:, coord_cols] += noise[:, coord_cols]

        if self.mirror_prob > 0 and np.random.random() < self.mirror_prob:
            x = x.copy()
            x[:, 0::4] *= -1.0

        return x.astype(np.float32), p.astype(bool)

