"""Sequence dataset for TCN training and inference."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class WellSequence:
    well_id: str
    X: "torch.Tensor | np.ndarray"   # (F, T) — post-PS sequence features (torch for train, can be numpy before scaler)
    y: "torch.Tensor | np.ndarray"   # (T,) — delta target values
    baseline: float                  # last_tvt_input for reconstruction
    row_indices: np.ndarray          # (T,) — absolute row indices in the well
    X_abs: np.ndarray | None = None  # (T, 4) — raw X,Y,Z,MD for global StandardScaler


class WellSequenceDataset(torch.utils.data.Dataset):
    """Lazy sliding-window dataset with configurable stride.

    Stores only (seq_idx, start_t) tuples. Slices from WellSequence.X
    on-the-fly in __getitem__. With X as (F,T) torch.Tensor, slicing is zero-copy.
    """

    def __init__(self, sequences: list[WellSequence], window_size: int = 64, stride: int = 1):
        self.window_size = window_size
        self._seqs = sequences
        self._sample_map: list[tuple[int, int]] = []  # (seq_idx, start_t)
        for seq_idx, seq in enumerate(sequences):
            T = seq.X.shape[1] if seq.X.ndim == 2 else len(seq.X)
            n_windows = max(0, T - window_size + 1)
            for t in range(0, n_windows, stride):
                self._sample_map.append((seq_idx, t))

    def __len__(self) -> int:
        return len(self._sample_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx, t = self._sample_map[idx]
        seq = self._seqs[seq_idx]
        w = self.window_size
        X = seq.X
        y_arr = seq.y
        if isinstance(X, np.ndarray):
            window = X[t : t + w]                          # (W, F)
            x = torch.from_numpy(window.copy()).T.contiguous()  # (F, W)
        else:
            x = X[:, t : t + w]                            # (F, W) — zero-copy slice
        target = y_arr[t + w - 1]
        if isinstance(target, torch.Tensor):
            y = target.clone().detach().unsqueeze(0).float()
        elif isinstance(target, np.ndarray):
            y = torch.tensor(float(target), dtype=torch.float32).unsqueeze(0)
        else:
            y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return x, y
