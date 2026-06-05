"""Target transforms for the ROGII baseline.

Options:
- signed_log: sign(x) * log1p(abs(x)) for residual target
- derivative: predict d(TVT)/d(MD) and integrate
"""

import numpy as np
import pandas as pd


def signed_log_transform(y: np.ndarray) -> np.ndarray:
    """Transform residuals: sign(x) * log1p(abs(x))."""
    return np.sign(y) * np.log1p(np.abs(y))


def signed_log_inverse(y_trans: np.ndarray) -> np.ndarray:
    """Inverse transform: sign(x) * (exp(abs(x)) - 1)."""
    return np.sign(y_trans) * (np.exp(np.abs(y_trans)) - 1.0)


def build_derivative_target(
    horizontal: pd.DataFrame,
    post_ps_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d(TVT)/d(MD) target for post-PS rows.

    Returns (derivative_values, md_values) for post-PS rows.
    Derivative is computed as finite difference TVT[i+1] - TVT[i] / (MD[i+1] - MD[i]).
    Last row uses the preceding derivative.
    """
    tv = horizontal["TVT"].astype(float).values
    md = horizontal["MD"].astype(float).values
    deriv = np.zeros(len(tv))
    for i in range(len(tv) - 1):
        dmd = md[i + 1] - md[i]
        if dmd > 1e-9:
            deriv[i] = (tv[i + 1] - tv[i]) / dmd
    deriv[-1] = deriv[-2] if len(deriv) > 1 else 0.0
    return deriv[post_ps_mask], md[post_ps_mask]


def integrate_derivative_predictions(
    y_pred_deriv: np.ndarray,
    md_values: np.ndarray,
    initial_tvt: float,
) -> np.ndarray:
    """Integrate predicted derivatives to reconstruct TVT.

    TVT[i] = initial_tvt + sum_{j=0}^{i-1} deriv[j] * (md[j+1] - md[j])
    """
    if len(y_pred_deriv) == 0:
        return np.array([], dtype=float)
    tvt = np.zeros(len(y_pred_deriv))
    tvt[0] = initial_tvt + y_pred_deriv[0] * (md_values[1] - md_values[0]) if len(md_values) > 1 else initial_tvt
    for i in range(1, len(y_pred_deriv)):
        dmd = md_values[i] - md_values[i - 1]
        tvt[i] = tvt[i - 1] + y_pred_deriv[i] * dmd
    return tvt
