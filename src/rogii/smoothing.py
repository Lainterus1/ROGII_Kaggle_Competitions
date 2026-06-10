"""Per-well Savitzky-Golay smoothing and TVT clipping of predicted sequences."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rogii.metrics import rmse
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# PostprocConfig — immutable train/predict contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostprocConfig:
    """Immutable post-processing configuration for train/predict handoff.

    Stored in model payload as ``payload["postproc"]`` so predict can
    reproduce the exact post-processing chain without CLI flags.

    Args:
        savgol_window: Savgol filter window length (odd int, or None to skip).
        savgol_polyorder: Savgol filter polynomial order.
        clip_lower: Minimum TVT bound (None = skip clipping).
        clip_upper: Maximum TVT bound (None = skip clipping).
        apply_order: Processing order: clip→smooth or smooth→clip.
    """

    savgol_window: int | None = 31
    savgol_polyorder: int = 2
    clip_lower: float | None = None
    clip_upper: float | None = None
    apply_order: Literal["clip_smooth", "smooth_clip"] = "clip_smooth"

    def validate(self) -> None:
        if self.savgol_window is not None:
            if self.savgol_window <= 0:
                raise ValueError(f"savgol_window must be positive, got {self.savgol_window}")
            if self.savgol_window % 2 == 0:
                raise ValueError(f"savgol_window must be odd, got {self.savgol_window}")
            if self.savgol_window <= self.savgol_polyorder:
                raise ValueError(
                    f"savgol_window ({self.savgol_window}) must be > "
                    f"savgol_polyorder ({self.savgol_polyorder})"
                )
        if self.clip_lower is not None and self.clip_upper is not None:
            if self.clip_lower >= self.clip_upper:
                raise ValueError(
                    f"clip_lower ({self.clip_lower}) must be < "
                    f"clip_upper ({self.clip_upper})"
                )
        if self.apply_order not in ("clip_smooth", "smooth_clip"):
            raise ValueError(
                f"apply_order must be 'clip_smooth' or 'smooth_clip', "
                f"got '{self.apply_order}'"
            )

    def to_dict(self) -> dict:
        return {
            "savgol_window": self.savgol_window,
            "savgol_polyorder": self.savgol_polyorder,
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
            "apply_order": self.apply_order,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PostprocConfig":
        return cls(
            savgol_window=d.get("savgol_window"),
            savgol_polyorder=d.get("savgol_polyorder", 2),
            clip_lower=d.get("clip_lower"),
            clip_upper=d.get("clip_upper"),
            apply_order=d.get("apply_order", "clip_smooth"),
        )


# ---------------------------------------------------------------------------
# PostprocParamGrid — search specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostprocParamGrid:
    """Grid specification for post-processing parameter search.

    ``clip_percentiles`` are **search specs** (percentile pairs).
    Actual numerical bounds are computed from train data at search time
    and stored in ``PostprocConfig`` / payload as concrete values.
    """

    savgol_windows: tuple[int, ...] = (5, 11, 17, 25, 31, 41, 51)
    savgol_polyorders: tuple[int, ...] = (2, 3)
    clip_percentiles: tuple[tuple[float, float], ...] = (
        (0.1, 99.9),
        (0.5, 99.5),
        (1.0, 99.0),
    )


# ---------------------------------------------------------------------------
# PostprocResult — evaluation of a single config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostprocResult:
    """Evaluation result for one post-processing configuration."""

    config: PostprocConfig
    rmse: float
    delta_vs_raw: float | None = None
    delta_vs_current: float | None = None


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------

def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers.

    ``"5,11,17"`` → ``[5, 11, 17]``.

    Raises:
        ValueError: empty string or non-integer values.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty string is not a valid int list")
    parts = s.split(",")
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            raise ValueError(f"Empty element in int list: '{s}'")
        try:
            result.append(int(p))
        except ValueError:
            raise ValueError(f"Invalid integer '{p}' in list: '{s}'")
    return result


def parse_percentile_pairs(s: str) -> list[tuple[float, float]]:
    """Parse comma-separated percentile pairs.

    ``"0.1:99.9,0.5:99.5"`` → ``[(0.1, 99.9), (0.5, 99.5)]``.

    Raises:
        ValueError: empty string, bad format, lower >= upper,
                    or values outside [0, 100].
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty string is not a valid percentile pair list")
    pairs = s.split(",")
    result = []
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            raise ValueError(f"Empty element in percentile pairs: '{s}'")
        if ":" not in pair:
            raise ValueError(
                f"Percentile pair must use ':' separator, got '{pair}'"
            )
        try:
            lower_str, upper_str = pair.split(":", 1)
            lower = float(lower_str.strip())
            upper = float(upper_str.strip())
        except ValueError:
            raise ValueError(f"Invalid percentile pair: '{pair}'")
        if lower < 0 or lower > 100 or upper < 0 or upper > 100:
            raise ValueError(
                f"Percentiles must be in [0, 100], got {lower}, {upper}"
            )
        if lower >= upper:
            raise ValueError(
                f"Lower percentile ({lower}) must be < upper ({upper})"
            )
        result.append((lower, upper))
    return result


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_postproc_configs(
    grid: PostprocParamGrid,
    clip_bounds_map: dict[str, tuple[float, float]],
) -> list[PostprocConfig]:
    """Generate all post-processing configs for grid search.

    Includes:
    - Raw / no postproc
    - Current Savgol w=31 p=2 without clipping
    - All Savgol variants without clipping
    - All clip-only variants
    - All Savgol + clipping combinations

    Args:
        grid: Search grid specification.
        clip_bounds_map: Mapping ``"p{low}-p{high}"`` → ``(clip_lower, clip_upper)``
            pre-computed from train data.

    Returns:
        List of PostprocConfig to evaluate.
    """
    configs: list[PostprocConfig] = []

    # 1. Raw (no postproc)
    raw = PostprocConfig(savgol_window=None, savgol_polyorder=0)
    try:
        raw.validate()
    except ValueError:
        raw = PostprocConfig(savgol_window=None, savgol_polyorder=1)
    configs.append(raw)

    # 2. Current default Savgol w=31 p=2 without clipping
    current = PostprocConfig(savgol_window=31, savgol_polyorder=2)
    current.validate()
    configs.append(current)

    # 3. Savgol variants without clipping
    for w in grid.savgol_windows:
        for p in grid.savgol_polyorders:
            if w <= p:
                continue
            cfg = PostprocConfig(savgol_window=w, savgol_polyorder=p)
            cfg.validate()
            if cfg not in configs:
                configs.append(cfg)

    # 4. Clip-only variants
    for pct_label, (lo, hi) in clip_bounds_map.items():
        cfg = PostprocConfig(savgol_window=None, savgol_polyorder=0,
                             clip_lower=lo, clip_upper=hi)
        try:
            cfg.validate()
        except ValueError:
            continue
        if cfg not in configs:
            configs.append(cfg)

    # 5. Savgol + clipping combinations
    for w in grid.savgol_windows:
        for p in grid.savgol_polyorders:
            if w <= p:
                continue
            for pct_label, (lo, hi) in clip_bounds_map.items():
                cfg = PostprocConfig(savgol_window=w, savgol_polyorder=p,
                                     clip_lower=lo, clip_upper=hi)
                try:
                    cfg.validate()
                except ValueError:
                    continue
                if cfg not in configs:
                    configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_postproc_config(
    oof_df: pd.DataFrame,
    config: PostprocConfig,
    y_true_full: np.ndarray | None = None,
    y_pred_full: np.ndarray | None = None,
) -> PostprocResult:
    """Evaluate a single post-processing config on OOF predictions.

    Works in full TVT space: reconstructs delta→full, applies the config
    chain (clip→smooth or smooth→clip), computes RMSE.

    Args:
        oof_df: OOF DataFrame with columns:
            well_id, row_idx, fold, y_true, y_pred, baseline.
        config: Post-processing configuration to evaluate.
        y_true_full: Pre-computed full TVT true values (optional).
        y_pred_full: Pre-computed full TVT pred values (optional).

    Returns:
        PostprocResult with rmse in full TVT space.
    """
    oof_copy = oof_df.copy()

    if y_true_full is None:
        oof_copy["y_true_full"] = oof_copy["y_true"] + oof_copy["baseline"]
        y_true_full = oof_copy["y_true_full"].to_numpy(dtype=float)
    else:
        oof_copy["y_true_full"] = y_true_full

    if y_pred_full is None:
        oof_copy["y_pred_full"] = oof_copy["y_pred"] + oof_copy["baseline"]
        y_pred_full = oof_copy["y_pred_full"].to_numpy(dtype=float)
    else:
        oof_copy["y_pred_full"] = y_pred_full

    oof_copy = oof_copy.sort_values(["well_id", "row_idx"]).reset_index(drop=True)

    y_pred_pp = np.zeros(len(oof_copy), dtype=float)
    for wid, group in oof_copy.groupby("well_id", sort=False):
        preds = group["y_pred_full"].to_numpy(dtype=float)

        if config.apply_order == "clip_smooth":
            if config.clip_lower is not None and config.clip_upper is not None:
                preds = preds.clip(config.clip_lower, config.clip_upper)
            if config.savgol_window is not None and len(preds) > config.savgol_window:
                preds = savgol_filter(preds, config.savgol_window, config.savgol_polyorder)
        else:
            if config.savgol_window is not None and len(preds) > config.savgol_window:
                preds = savgol_filter(preds, config.savgol_window, config.savgol_polyorder)
            if config.clip_lower is not None and config.clip_upper is not None:
                preds = preds.clip(config.clip_lower, config.clip_upper)

        y_pred_pp[group.index] = preds

    y_true_full = oof_copy["y_true_full"].to_numpy(dtype=float)
    score = rmse(y_true_full, y_pred_pp)

    raw_rmse = rmse(
        oof_copy["y_true_full"].to_numpy(dtype=float),
        oof_copy["y_pred_full"].to_numpy(dtype=float),
    )

    return PostprocResult(
        config=config,
        rmse=score,
        delta_vs_raw=score - raw_rmse,
    )


def grid_search_postprocessing(
    oof_df: pd.DataFrame,
    grid: PostprocParamGrid,
    clip_bounds_map: dict[str, tuple[float, float]],
) -> list[PostprocResult]:
    """Run full grid search over post-processing configs.

    Args:
        oof_df: OOF DataFrame (as built by ``_build_oof_per_well``).
        grid: Search grid specification.
        clip_bounds_map: Pre-computed clip bounds by percentile label.

    Returns:
        List of PostprocResult sorted by rmse ascending.
    """
    configs = generate_postproc_configs(grid, clip_bounds_map)

    y_true_full = oof_df["y_true"].to_numpy(dtype=float) + oof_df["baseline"].to_numpy(dtype=float)
    y_pred_full = oof_df["y_pred"].to_numpy(dtype=float) + oof_df["baseline"].to_numpy(dtype=float)
    raw_rmse = rmse(y_true_full, y_pred_full)

    results: list[PostprocResult] = []

    # Raw baseline
    results.append(PostprocResult(
        config=configs[0],
        rmse=raw_rmse,
        delta_vs_raw=0.0,
    ))

    for cfg in configs[1:]:
        r = evaluate_postproc_config(oof_df, cfg, y_true_full, y_pred_full)
        results.append(r)

    results.sort(key=lambda r: r.rmse)
    return results


def select_best_postproc(
    results: list[PostprocResult],
    baseline_config: PostprocConfig,
    min_delta: float = 0.01,
) -> PostprocConfig:
    """Select the best post-processing config, guarding against noise.

    If the best config improves RMSE by less than ``min_delta`` vs the
    baseline, return the baseline config instead (prefer simplicity).

    Args:
        results: Sorted list of PostprocResult (best first).
        baseline_config: Config to fall back to if improvement is marginal.
        min_delta: Minimum RMSE improvement to accept a new config.

    Returns:
        The selected PostprocConfig.
    """
    if not results:
        return baseline_config

    best = results[0]
    # Find the baseline result for comparison
    baseline_rmse: float | None = None
    for r in results:
        if r.config == baseline_config:
            baseline_rmse = r.rmse
            break

    if baseline_rmse is None:
        return best.config

    improvement = baseline_rmse - best.rmse
    if improvement < min_delta:
        return baseline_config

    return best.config


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def smooth_predictions_per_well(
    submission_df: pd.DataFrame,
    window: int = 31,
    polyorder: int = 2,
) -> pd.DataFrame:
    """Apply Savgol smoothing to TVT predictions per well.

    Parses well_id from submission id and smooths each well's TVT sequence
    independently. Short sequences (length <= window) are passed through
    unchanged.

    Args:
        submission_df: DataFrame with "id" and "tvt" columns.
        window: Savgol filter window length (must be odd, default 31).
        polyorder: Savgol filter polynomial order (default 2).

    Returns:
        New DataFrame with smoothed "tvt" column.
    """
    result = submission_df.copy()
    ids = result["id"].astype(str)
    last_underscore = ids.str.rsplit("_", n=1)
    well_ids = last_underscore.str[0]
    row_indices = last_underscore.str[1].astype(int)

    result["_well_id"] = well_ids
    result["_row_idx"] = row_indices
    result = result.sort_values(["_well_id", "_row_idx"]).reset_index(drop=True)

    smoothed = np.zeros(len(result), dtype=float)
    for well_id, group in result.groupby("_well_id", sort=False):
        values = group["tvt"].to_numpy(dtype=float)
        if len(values) > window:
            smoothed[group.index] = savgol_filter(values, window, polyorder)
        else:
            smoothed[group.index] = values

    result["tvt"] = smoothed
    result = result.drop(columns=["_well_id", "_row_idx"])
    return result


def clip_predictions(
    submission_df: pd.DataFrame,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """Clamp TVT predictions to [lower, upper].

    Args:
        submission_df: DataFrame with "id" and "tvt" columns.
        lower: Minimum allowed TVT value.
        upper: Maximum allowed TVT value.

    Returns:
        New DataFrame with clipped "tvt" column.
    """
    result = submission_df.copy()
    result["tvt"] = result["tvt"].clip(lower, upper)
    return result


def compute_tvt_clip_bounds(
    data_dir: str | Path,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
) -> tuple[float, float]:
    """Scan train TVT values and return (lower, upper) clip bounds.

    Args:
        data_dir: Competition data directory.
        lower_percentile: Percentile for lower clipping bound.
        upper_percentile: Percentile for upper clipping bound.

    Returns:
        (lower, upper) tuple of float bounds.
    """
    from rogii.data_loading import list_well_ids, read_horizontal_well

    well_ids = list_well_ids(data_dir, "train")
    all_tvt: list[float] = []

    for wid in well_ids:
        df = read_horizontal_well(data_dir, "train", wid)
        if "TVT" not in df.columns:
            continue
        tvt = df["TVT"].dropna().to_numpy(dtype=float)
        all_tvt.extend(tvt.tolist())

    if not all_tvt:
        raise ValueError("No TVT values found in train data")

    arr = np.array(all_tvt, dtype=float)
    lower = float(np.percentile(arr, lower_percentile))
    upper = float(np.percentile(arr, upper_percentile))
    return lower, upper


def apply_postprocessing(
    submission_df: pd.DataFrame,
    savgol_window: int | None = None,
    savgol_polyorder: int = 2,
    clip_lower: float | None = None,
    clip_upper: float | None = None,
    apply_order: str = "clip_smooth",
) -> pd.DataFrame:
    """Apply post-processing (clipping + Savgol smoothing) to predictions.

    Args:
        submission_df: DataFrame with "id" and "tvt" columns.
        savgol_window: Savgol window length (None = skip smoothing).
        savgol_polyorder: Savgol polynomial order.
        clip_lower: Minimum TVT bound (None = skip clipping).
        clip_upper: Maximum TVT bound (None = skip clipping).
        apply_order: Order: "clip_smooth" or "smooth_clip".

    Returns:
        New DataFrame with post-processed "tvt" column.
    """
    result = submission_df.copy()

    if apply_order == "clip_smooth":
        if clip_lower is not None and clip_upper is not None:
            result = clip_predictions(result, clip_lower, clip_upper)
        if savgol_window is not None:
            result = smooth_predictions_per_well(result, window=savgol_window, polyorder=savgol_polyorder)
    else:
        if savgol_window is not None:
            result = smooth_predictions_per_well(result, window=savgol_window, polyorder=savgol_polyorder)
        if clip_lower is not None and clip_upper is not None:
            result = clip_predictions(result, clip_lower, clip_upper)

    return result
