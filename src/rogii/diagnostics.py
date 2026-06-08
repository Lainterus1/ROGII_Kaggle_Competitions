"""TCN Phase 0 diagnostics: OOF analysis by fold, position, well length, dispersion and error correlation."""

from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from rogii.data_loading import read_horizontal_well
from rogii.features import post_ps_mask
from rogii.metrics import rmse
from rogii.oof import load_oof


@lru_cache(maxsize=1024)
def _get_well_ps_metadata(data_dir: str, well_id: str) -> tuple[int | None, int]:
    horizontal = read_horizontal_well(data_dir, "train", well_id)
    mask = post_ps_mask(horizontal)
    if not mask.any():
        return None, 0
    ps_start = int(np.where(mask)[0][0])
    post_ps_length = int(mask.sum())
    return ps_start, post_ps_length


def load_oof_with_metadata(oof_path: str | Path, data_dir: str | Path) -> pd.DataFrame:
    """Load OOF parquet and enrich with per-row position/well metadata from raw CSVs.

    Adds columns: frac_after_ps, rows_since_ps, well_post_ps_length.
    """
    oof = load_oof(oof_path).copy()
    data_dir = str(data_dir)

    well_ids = oof["well_id"].unique()
    meta_cache: dict[str, tuple[int | None, int]] = {}
    for wid in well_ids:
        meta_cache[wid] = _get_well_ps_metadata(data_dir, wid)

    frac_after_ps = np.empty(len(oof), dtype=float)
    rows_since_ps = np.empty(len(oof), dtype=int)
    well_lengths = np.empty(len(oof), dtype=int)

    for i, (_, row) in enumerate(oof.iterrows()):
        wid = row["well_id"]
        ps_start, post_ps_len = meta_cache[wid]
        if ps_start is None:
            frac_after_ps[i] = np.nan
            rows_since_ps[i] = -1
            well_lengths[i] = 0
        else:
            r_idx = row["row_idx"]
            rs = r_idx - ps_start
            frac = rs / max(post_ps_len - 1, 1)
            frac_after_ps[i] = frac
            rows_since_ps[i] = rs
            well_lengths[i] = post_ps_len

    oof["frac_after_ps"] = frac_after_ps
    oof["rows_since_ps"] = rows_since_ps
    oof["well_post_ps_length"] = well_lengths
    return oof


def compute_rmse_by_fold(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold RMSE in delta and full-TVT space.

    Returns empty DataFrame if fold column is missing or all-NaN.
    """
    if "fold" not in oof_df.columns or oof_df["fold"].isna().all():
        return pd.DataFrame()

    rows = []
    for fold_val in sorted(oof_df["fold"].dropna().unique()):
        fold_mask = oof_df["fold"] == fold_val
        fold_data = oof_df.loc[fold_mask]
        delta_rmse = rmse(
            fold_data["y_true"].to_numpy(dtype=float),
            fold_data["y_pred"].to_numpy(dtype=float),
        )
        full_true = fold_data["y_true"].to_numpy(dtype=float) + fold_data["baseline"].to_numpy(dtype=float)
        full_pred = fold_data["y_pred"].to_numpy(dtype=float) + fold_data["baseline"].to_numpy(dtype=float)
        full_rmse = rmse(full_true, full_pred)
        rows.append({
            "fold": int(fold_val),
            "n_rows": len(fold_data),
            "n_wells": fold_data["well_id"].nunique(),
            "rmse_delta": round(delta_rmse, 4),
            "rmse_full": round(full_rmse, 4),
        })
    return pd.DataFrame(rows)


def compute_rmse_by_position(oof_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """RMSE by frac_after_ps bin.

    Also returns RMSE for first/last N rows as special entries.
    """
    if "frac_after_ps" not in oof_df.columns:
        raise ValueError("oof_df must contain frac_after_ps column (run load_oof_with_metadata first)")

    valid = oof_df["frac_after_ps"].notna()
    if not valid.any():
        return pd.DataFrame()

    data = oof_df.loc[valid].copy()
    data["bin"] = pd.cut(data["frac_after_ps"], bins=n_bins, labels=False, include_lowest=True)

    rows = []
    for b in range(n_bins):
        mask = data["bin"] == b
        if not mask.any():
            continue
        chunk = data.loc[mask]
        r = rmse(chunk["y_true"].to_numpy(dtype=float), chunk["y_pred"].to_numpy(dtype=float))
        frac_min = chunk["frac_after_ps"].min()
        frac_max = chunk["frac_after_ps"].max()
        rows.append({
            "frac_bin": f"[{frac_min:.2f}-{frac_max:.2f})",
            "n_rows": len(chunk),
            "rmse_delta": round(r, 4),
        })

    # First / last N rows
    for label, mask_fn in [
        ("first_5", lambda d: d["rows_since_ps"] < 5),
        ("first_10", lambda d: d["rows_since_ps"] < 10),
        ("last_5", lambda d: d["frac_after_ps"] > 1.0 - 5.0 / d["well_post_ps_length"].clip(lower=1)),
        ("last_10", lambda d: d["frac_after_ps"] > 1.0 - 10.0 / d["well_post_ps_length"].clip(lower=1)),
    ]:
        mask = mask_fn(data)
        if mask.any():
            chunk = data.loc[mask]
            r = rmse(chunk["y_true"].to_numpy(dtype=float), chunk["y_pred"].to_numpy(dtype=float))
            rows.append({
                "frac_bin": f"* {label}",
                "n_rows": len(chunk),
                "rmse_delta": round(r, 4),
            })

    return pd.DataFrame(rows)


def compute_rmse_by_well_length(oof_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """RMSE grouped by well post-PS length bins."""
    if "well_post_ps_length" not in oof_df.columns:
        raise ValueError("oof_df must contain well_post_ps_length column (run load_oof_with_metadata first)")

    valid = oof_df["well_post_ps_length"] > 0
    if not valid.any():
        return pd.DataFrame()

    data = oof_df.loc[valid].copy()

    # Bin by well length
    well_lengths = data.groupby("well_id")["well_post_ps_length"].first()
    length_bins = pd.qcut(well_lengths, q=n_bins, duplicates="drop")
    bin_map = dict(zip(well_lengths.index, length_bins))

    rows = []
    for bin_label, well_group in data.groupby(data["well_id"].map(bin_map), observed=False):
        r = rmse(well_group["y_true"].to_numpy(dtype=float), well_group["y_pred"].to_numpy(dtype=float))
        rows.append({
            "length_bin": str(bin_label),
            "n_wells": well_group["well_id"].nunique(),
            "n_rows": len(well_group),
            "rmse_delta": round(r, 4),
        })
    return pd.DataFrame(rows)


def compute_prediction_dispersion(oof_df: pd.DataFrame) -> dict:
    """Check if TCN predictions collapse toward the mean (flattening).

    Returns dict with target_std, pred_std, std_ratio, and per-well ratios.
    """
    y_true = oof_df["y_true"].to_numpy(dtype=float)
    y_pred = oof_df["y_pred"].to_numpy(dtype=float)

    target_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    std_ratio = pred_std / target_std if target_std > 1e-8 else 0.0

    per_well_ratios = []
    flattening_wells = []
    for wid, group in oof_df.groupby("well_id"):
        t_std = float(np.std(group["y_true"].to_numpy(dtype=float)))
        p_std = float(np.std(group["y_pred"].to_numpy(dtype=float)))
        ratio = p_std / t_std if t_std > 1e-8 else 0.0
        per_well_ratios.append(ratio)
        if ratio < 0.5 and len(group) > 10:
            flattening_wells.append((wid, ratio))

    return {
        "target_std": round(target_std, 2),
        "pred_std": round(pred_std, 2),
        "std_ratio": round(std_ratio, 4),
        "target_mean": round(float(np.mean(y_true)), 2),
        "pred_mean": round(float(np.mean(y_pred)), 2),
        "per_well_ratio_median": round(float(np.median(per_well_ratios)), 4),
        "per_well_ratio_std": round(float(np.std(per_well_ratios)), 4),
        "n_flattening_wells": len(flattening_wells),
        "flattening_wells": flattening_wells[:10],
    }


def compute_error_correlation(oof_tcn: pd.DataFrame, oof_lgbm: pd.DataFrame) -> dict:
    """Compare TCN and LGBM errors to assess blend potential.

    Joins on (well_id, row_idx). Computes global and per-well error correlation, plus mean-blend RMSE.
    """
    merged = oof_tcn.merge(
        oof_lgbm,
        on=["well_id", "row_idx"],
        suffixes=("_tcn", "_lgbm"),
        how="inner",
    )
    if len(merged) == 0:
        return {"error": "No overlapping rows between TCN and LGBM OOF"}

    err_tcn = merged["y_pred_tcn"].to_numpy(dtype=float) - merged["y_true_tcn"].to_numpy(dtype=float)
    err_lgbm = merged["y_pred_lgbm"].to_numpy(dtype=float) - merged["y_true_lgbm"].to_numpy(dtype=float)

    global_corr = float(np.corrcoef(err_tcn, err_lgbm)[0, 1]) if len(err_tcn) > 1 else 0.0

    per_well_corrs = []
    for wid, group in merged.groupby("well_id"):
        if len(group) < 3:
            continue
        e_t = group["y_pred_tcn"].to_numpy(dtype=float) - group["y_true_tcn"].to_numpy(dtype=float)
        e_l = group["y_pred_lgbm"].to_numpy(dtype=float) - group["y_true_lgbm"].to_numpy(dtype=float)
        c = np.corrcoef(e_t, e_l)[0, 1]
        if not np.isnan(c):
            per_well_corrs.append(c)

    tcn_delta_rmse = rmse(
        merged["y_true_tcn"].to_numpy(dtype=float),
        merged["y_pred_tcn"].to_numpy(dtype=float),
    )
    lgbm_delta_rmse = rmse(
        merged["y_true_lgbm"].to_numpy(dtype=float),
        merged["y_pred_lgbm"].to_numpy(dtype=float),
    )

    blend_pred = (merged["y_pred_tcn"].to_numpy(dtype=float) + merged["y_pred_lgbm"].to_numpy(dtype=float)) / 2.0
    blend_rmse_val = rmse(merged["y_true_tcn"].to_numpy(dtype=float), blend_pred)

    return {
        "n_shared_rows": len(merged),
        "n_shared_wells": merged["well_id"].nunique(),
        "global_error_corr": round(global_corr, 4),
        "per_well_corr_median": round(float(np.median(per_well_corrs)) if per_well_corrs else 0.0, 4),
        "per_well_corr_mean": round(float(np.mean(per_well_corrs)) if per_well_corrs else 0.0, 4),
        "per_well_corr_std": round(float(np.std(per_well_corrs)) if per_well_corrs else 0.0, 4),
        "tcn_rmse_delta": round(tcn_delta_rmse, 4),
        "lgbm_rmse_delta": round(lgbm_delta_rmse, 4),
        "mean_blend_rmse_delta": round(blend_rmse_val, 4),
        "blend_gain_vs_best": round(blend_rmse_val - min(tcn_delta_rmse, lgbm_delta_rmse), 4),
    }


def compute_per_well_rmse(oof_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top-N worst and best wells by delta RMSE."""
    rows = []
    for wid, group in oof_df.groupby("well_id"):
        r = rmse(group["y_true"].to_numpy(dtype=float), group["y_pred"].to_numpy(dtype=float))
        rows.append({
            "well_id": wid,
            "n_rows": len(group),
            "rmse_delta": round(r, 4),
        })
    df = pd.DataFrame(rows).sort_values("rmse_delta", ascending=False)
    return pd.concat([df.head(top_n), df.tail(top_n)], ignore_index=True)


def generate_report(
    oof_df: pd.DataFrame,
    lgbm_oof_path: str | Path | None = None,
) -> str:
    """Generate full Phase 0 diagnostic report as a formatted string."""

    lines: list[str] = []
    lines.append("=" * 65)
    lines.append("TCN Phase 0 Diagnostic Report")
    lines.append("=" * 65)
    lines.append(f"Rows: {len(oof_df):,}")
    lines.append(f"Wells: {oof_df['well_id'].nunique()}")
    lines.append("")

    has_fold = "fold" in oof_df.columns and not oof_df["fold"].isna().all()

    # 1. RMSE by fold
    if has_fold:
        lines.append("--- 1. RMSE by Fold ---")
        fold_df = compute_rmse_by_fold(oof_df)
        if len(fold_df) > 0:
            lines.append(fold_df.to_string(index=False))
        else:
            lines.append("  (no fold data)")
        lines.append("")

    # 2. RMSE by position
    if "frac_after_ps" in oof_df.columns:
        lines.append("--- 2. RMSE by Position (frac_after_ps) ---")
        pos_df = compute_rmse_by_position(oof_df)
        lines.append(pos_df.to_string(index=False))
        lines.append("")

    # 3. RMSE by well length
    if "well_post_ps_length" in oof_df.columns:
        lines.append("--- 3. RMSE by Well Length ---")
        len_df = compute_rmse_by_well_length(oof_df)
        if len(len_df) > 0:
            lines.append(len_df.to_string(index=False))
        lines.append("")

    # 4. Prediction dispersion
    lines.append("--- 4. Prediction Dispersion ---")
    disp = compute_prediction_dispersion(oof_df)
    lines.append(f"  Target std:       {disp['target_std']:,.2f}")
    lines.append(f"  Prediction std:   {disp['pred_std']:,.2f}")
    lines.append(f"  Std ratio:        {disp['std_ratio']:.4f}  {'<-- FLATTENING' if disp['std_ratio'] < 0.7 else '<-- OK'}")
    lines.append(f"  Target mean:      {disp['target_mean']:,.2f}")
    lines.append(f"  Prediction mean:  {disp['pred_mean']:,.2f}")
    lines.append(f"  Per-well ratio median: {disp['per_well_ratio_median']:.4f}")
    lines.append(f"  Per-well ratio std:    {disp['per_well_ratio_std']:.4f}")
    lines.append(f"  Flattening wells: {disp['n_flattening_wells']} (ratio < 0.5)")
    if disp["flattening_wells"]:
        lines.append("  Worst flattening:")
        for wid, ratio in disp["flattening_wells"][:5]:
            lines.append(f"    {wid}: ratio={ratio:.3f}")
    lines.append("")

    # 5. Error correlation
    if lgbm_oof_path:
        lines.append("--- 5. Error Correlation (TCN vs LGBM) ---")
        try:
            lgbm_oof = load_oof(lgbm_oof_path)
            corr = compute_error_correlation(oof_df, lgbm_oof)
            if "error" in corr:
                lines.append(f"  {corr['error']}")
            else:
                lines.append(f"  Shared rows:      {corr['n_shared_rows']:,}")
                lines.append(f"  Shared wells:     {corr['n_shared_wells']}")
                lines.append(f"  Global error corr:   {corr['global_error_corr']:.4f}")
                lines.append(f"  Per-well corr median: {corr['per_well_corr_median']:.4f}")
                lines.append(f"  Per-well corr mean:   {corr['per_well_corr_mean']:.4f}")
                lines.append(f"  TCN RMSE (delta):     {corr['tcn_rmse_delta']:.4f}")
                lines.append(f"  LGBM RMSE (delta):    {corr['lgbm_rmse_delta']:.4f}")
                lines.append(f"  Mean blend RMSE:      {corr['mean_blend_rmse_delta']:.4f}")
                gain = corr["blend_gain_vs_best"]
                verdict = "WORTH BLENDING" if gain < 0 else "no gain"
                lines.append(f"  Blend gain vs best:   {gain:+.4f}  <- {verdict}")
        except Exception as e:
            lines.append(f"  Error loading LGBM OOF: {e}")
        lines.append("")

    # 6. Per-well RMSE
    lines.append("--- 6. Worst 10 + Best 10 Wells ---")
    well_df = compute_per_well_rmse(oof_df, top_n=10)
    lines.append(well_df.to_string(index=False))
    lines.append("")

    # 7. Summary
    lines.append("--- 7. Summary ---")
    total_rmse_delta = rmse(
        oof_df["y_true"].to_numpy(dtype=float),
        oof_df["y_pred"].to_numpy(dtype=float),
    )
    lines.append(f"  Overall RMSE (delta): {total_rmse_delta:.4f}")
    if has_fold:
        fold_df = compute_rmse_by_fold(oof_df)
        if len(fold_df) > 1:
            worst_fold = fold_df.loc[fold_df["rmse_delta"].idxmax()]
            best_fold = fold_df.loc[fold_df["rmse_delta"].idxmin()]
            lines.append(f"  Best fold:  {int(best_fold['fold'])}  RMSE={best_fold['rmse_delta']:.4f}")
            lines.append(f"  Worst fold: {int(worst_fold['fold'])}  RMSE={worst_fold['rmse_delta']:.4f}")
            lines.append(f"  Fold spread: {worst_fold['rmse_delta'] - best_fold['rmse_delta']:.4f}")
    lines.append("=" * 65)

    return "\n".join(lines)
