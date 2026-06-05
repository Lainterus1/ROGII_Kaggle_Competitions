"""Causal GR DWT (Discrete Wavelet Transform) features.

All features use trailing/expanding windows only. Feature value at row i
must not depend on GR values after i.
"""

import numpy as np
import pandas as pd
import pywt


def build_gr_dwt_features(
    horizontal: pd.DataFrame,
    wavelet: str = "db4",
    window: int = 256,
    min_window: int = 16,
) -> pd.DataFrame:
    """Build causal GR DWT features: approx and detail energy.

    For each row i, a trailing window of up to `window` GR values
    (GR[max(0, i - window + 1) : i + 1]) is decomposed via pywt.wavedec
    at level 1. The last value of the reconstructed approximation gives
    `gr_dwt_approx`; the mean squared detail coefficient gives
    `gr_dwt_detail_energy`.

    Early rows (i < min_window) fall back to the raw GR value and 0
    detail energy.
    """
    n = len(horizontal)
    gr = horizontal["GR"].astype(float).values
    gr_filled = np.where(np.isnan(gr), 0.0, gr)

    approx = np.zeros(n, dtype=float)
    detail_energy = np.zeros(n, dtype=float)

    for i in range(n):
        lo = max(0, i - window + 1)
        segment = gr_filled[lo : i + 1]
        seg_len = len(segment)

        if seg_len < min_window:
            approx[i] = float(gr_filled[i])
            detail_energy[i] = 0.0
            continue

        coeffs = pywt.wavedec(segment, wavelet, level=1)
        cA, cD = coeffs[0], coeffs[1]

        recon = pywt.upcoef("a", cA, wavelet, level=1, take=seg_len)
        approx[i] = float(recon[-1])

        detail_energy[i] = float(np.mean(np.square(cD)))

    feats = pd.DataFrame(index=horizontal.index)
    feats["gr_dwt_approx"] = approx
    feats["gr_dwt_detail_energy"] = detail_energy
    return feats
