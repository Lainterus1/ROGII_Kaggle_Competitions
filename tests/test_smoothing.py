"""Tests for PostprocConfig, grid search, parsers, and smoothing functions."""

import numpy as np
import pandas as pd
import pytest

from rogii.smoothing import (
    PostprocConfig,
    PostprocParamGrid,
    PostprocResult,
    apply_postprocessing,
    compute_tvt_clip_bounds,
    evaluate_postproc_config,
    generate_postproc_configs,
    grid_search_postprocessing,
    parse_int_list,
    parse_percentile_pairs,
    select_best_postproc,
)


# ---------------------------------------------------------------------------
# PostprocConfig — validate / to_dict / from_dict
# ---------------------------------------------------------------------------

def test_postproc_config_default() -> None:
    cfg = PostprocConfig()
    cfg.validate()
    assert cfg.savgol_window == 31
    assert cfg.savgol_polyorder == 2
    assert cfg.clip_lower is None
    assert cfg.clip_upper is None
    assert cfg.apply_order == "clip_smooth"


def test_postproc_config_validate_window_must_be_odd() -> None:
    cfg = PostprocConfig(savgol_window=10)
    with pytest.raises(ValueError, match="must be odd"):
        cfg.validate()


def test_postproc_config_validate_window_greater_than_polyorder() -> None:
    cfg = PostprocConfig(savgol_window=5, savgol_polyorder=5)
    with pytest.raises(ValueError, match="must be >"):
        cfg.validate()

    cfg2 = PostprocConfig(savgol_window=5, savgol_polyorder=7)
    with pytest.raises(ValueError, match="must be >"):
        cfg2.validate()


def test_postproc_config_validate_clip_bounds_order() -> None:
    cfg = PostprocConfig(clip_lower=100, clip_upper=50)
    with pytest.raises(ValueError, match=r"clip_lower.*clip_upper"):
        cfg.validate()


def test_postproc_config_validate_apply_order() -> None:
    cfg = PostprocConfig(apply_order="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="apply_order must be"):
        cfg.validate()


def test_postproc_config_validate_window_positive() -> None:
    cfg = PostprocConfig(savgol_window=0)
    with pytest.raises(ValueError, match="must be positive"):
        cfg.validate()
    cfg2 = PostprocConfig(savgol_window=-1)
    with pytest.raises(ValueError, match="must be positive"):
        cfg2.validate()


def test_postproc_config_to_dict_roundtrip() -> None:
    cfg = PostprocConfig(
        savgol_window=51, savgol_polyorder=3,
        clip_lower=5000.0, clip_upper=20000.0,
        apply_order="smooth_clip",
    )
    d = cfg.to_dict()
    assert d["savgol_window"] == 51
    assert d["savgol_polyorder"] == 3
    assert d["clip_lower"] == 5000.0
    assert d["clip_upper"] == 20000.0
    assert d["apply_order"] == "smooth_clip"

    cfg2 = PostprocConfig.from_dict(d)
    assert cfg2 == cfg


def test_postproc_config_from_dict_defaults() -> None:
    cfg = PostprocConfig.from_dict({})
    assert cfg.savgol_window is None  # gets None from empty dict
    assert cfg.savgol_polyorder == 2
    assert cfg.clip_lower is None
    assert cfg.clip_upper is None
    assert cfg.apply_order == "clip_smooth"


def test_postproc_config_no_postproc() -> None:
    cfg = PostprocConfig(savgol_window=None, savgol_polyorder=1)
    cfg.validate()


def test_postproc_config_frozen() -> None:
    cfg = PostprocConfig()
    with pytest.raises(Exception):
        cfg.savgol_window = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PostprocParamGrid
# ---------------------------------------------------------------------------

def test_param_grid_defaults() -> None:
    grid = PostprocParamGrid()
    assert len(grid.savgol_windows) == 7
    assert grid.savgol_polyorders == (2, 3)
    assert grid.clip_percentiles == ((0.1, 99.9), (0.5, 99.5), (1.0, 99.0))


def test_param_grid_frozen() -> None:
    grid = PostprocParamGrid()
    with pytest.raises(Exception):
        grid.savgol_windows = (99,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PostprocResult
# ---------------------------------------------------------------------------

def test_postproc_result_frozen() -> None:
    cfg = PostprocConfig()
    r = PostprocResult(config=cfg, rmse=12.0, delta_vs_raw=-0.1)
    assert r.rmse == 12.0
    assert r.delta_vs_raw == -0.1
    assert r.delta_vs_current is None
    with pytest.raises(Exception):
        r.rmse = 13.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def test_parse_int_list_valid() -> None:
    assert parse_int_list("5,11,17") == [5, 11, 17]
    assert parse_int_list(" 5 , 11 , 17 ") == [5, 11, 17]
    assert parse_int_list("42") == [42]


def test_parse_int_list_invalid() -> None:
    with pytest.raises(ValueError, match="Empty string"):
        parse_int_list("")
    with pytest.raises(ValueError, match="Empty element"):
        parse_int_list("5,,11")
    with pytest.raises(ValueError, match="Invalid integer"):
        parse_int_list("abc")
    with pytest.raises(ValueError, match="Invalid integer"):
        parse_int_list("5,x,11")


def test_parse_percentile_pairs_valid() -> None:
    assert parse_percentile_pairs("0.1:99.9") == [(0.1, 99.9)]
    assert parse_percentile_pairs("0.1:99.9,0.5:99.5") == [(0.1, 99.9), (0.5, 99.5)]
    assert parse_percentile_pairs(" 0.1 : 99.9 ") == [(0.1, 99.9)]


def test_parse_percentile_pairs_invalid() -> None:
    with pytest.raises(ValueError, match="Empty string"):
        parse_percentile_pairs("")
    with pytest.raises(ValueError, match="Empty element"):
        parse_percentile_pairs("0.1:99.9,,0.5:99.5")
    with pytest.raises(ValueError, match="':' separator"):
        parse_percentile_pairs("0.1-99.9")
    with pytest.raises(ValueError, match="must be in"):
        parse_percentile_pairs("-1:99")
    with pytest.raises(ValueError, match="must be in"):
        parse_percentile_pairs("0:101")
    with pytest.raises(ValueError, match="must be <"):
        parse_percentile_pairs("99:1")
    with pytest.raises(ValueError, match="Invalid percentile pair"):
        parse_percentile_pairs("abc:def")


# ---------------------------------------------------------------------------
# generate_postproc_configs
# ---------------------------------------------------------------------------

def test_generate_configs_always_includes_raw() -> None:
    grid = PostprocParamGrid(
        savgol_windows=(5,),
        savgol_polyorders=(2,),
        clip_percentiles=(),
    )
    configs = generate_postproc_configs(grid, {})
    assert any(c.savgol_window is None for c in configs)


def test_generate_configs_always_includes_current_savgol() -> None:
    grid = PostprocParamGrid()
    configs = generate_postproc_configs(grid, {})
    current = PostprocConfig(savgol_window=31, savgol_polyorder=2)
    assert current in configs


def test_generate_configs_includes_savgol_only() -> None:
    grid = PostprocParamGrid(
        savgol_windows=(5, 11),
        savgol_polyorders=(2,),
        clip_percentiles=(),
    )
    configs = generate_postproc_configs(grid, {})
    cfg5 = PostprocConfig(savgol_window=5, savgol_polyorder=2)
    cfg11 = PostprocConfig(savgol_window=11, savgol_polyorder=2)
    assert cfg5 in configs
    assert cfg11 in configs


def test_generate_configs_includes_clip_only() -> None:
    grid = PostprocParamGrid(
        savgol_windows=(),
        savgol_polyorders=(),
        clip_percentiles=((0.1, 99.9),),
    )
    bounds_map = {"p0.1-p99.9": (5000.0, 20000.0)}
    configs = generate_postproc_configs(grid, bounds_map)
    clip_cfg = PostprocConfig(
        savgol_window=None, savgol_polyorder=0,
        clip_lower=5000.0, clip_upper=20000.0,
    )
    assert clip_cfg in configs


def test_generate_configs_includes_savgol_clip_combos() -> None:
    grid = PostprocParamGrid(
        savgol_windows=(5,),
        savgol_polyorders=(2,),
        clip_percentiles=((0.1, 99.9),),
    )
    bounds_map = {"p0.1-p99.9": (1000.0, 30000.0)}
    configs = generate_postproc_configs(grid, bounds_map)
    combo = PostprocConfig(
        savgol_window=5, savgol_polyorder=2,
        clip_lower=1000.0, clip_upper=30000.0,
    )
    assert combo in configs


def test_generate_configs_no_duplicates() -> None:
    grid = PostprocParamGrid()
    configs = generate_postproc_configs(grid, {})
    seen = set()
    for c in configs:
        key = (c.savgol_window, c.savgol_polyorder, c.clip_lower, c.clip_upper)
        assert key not in seen
        seen.add(key)


def test_generate_configs_skips_invalid_combos() -> None:
    grid = PostprocParamGrid(
        savgol_windows=(5,),
        savgol_polyorders=(5,),  # w <= p for w=5
        clip_percentiles=(),
    )
    configs = generate_postproc_configs(grid, {})
    cfg_5_5 = PostprocConfig(savgol_window=5, savgol_polyorder=5)
    assert cfg_5_5 not in configs


# ---------------------------------------------------------------------------
# evaluate_postproc_config
# ---------------------------------------------------------------------------

def _make_oof_df(n_wells: int = 3, n_rows_per_well: int = 20) -> pd.DataFrame:
    np.random.seed(0)
    rows = []
    for wi in range(n_wells):
        wid = f"well_{wi}"
        baseline = float(10000 + wi * 1000)
        for ri in range(n_rows_per_well):
            true_delta = np.random.randn() * 100
            pred_delta = true_delta + np.random.randn() * 10
            rows.append((wid, ri, 0, true_delta, pred_delta, baseline))
    return pd.DataFrame(
        rows, columns=["well_id", "row_idx", "fold", "y_true", "y_pred", "baseline"],
    )


def _make_smooth_oof_df(n_wells: int = 2, n_rows: int = 100) -> pd.DataFrame:
    """Generate OOF data with smooth signal + noise where Savgol helps."""
    np.random.seed(1)
    rows = []
    for wi in range(n_wells):
        wid = f"well_{wi}"
        base = 10000.0
        true_signal = base + 500 * np.sin(np.linspace(0, 4 * np.pi, n_rows))
        noise = np.random.randn(n_rows) * 30
        pred_signal = true_signal + noise
        for ri in range(n_rows):
            rows.append((wid, ri, 0, float(true_signal[ri]), float(pred_signal[ri]), 0.0))
    return pd.DataFrame(
        rows, columns=["well_id", "row_idx", "fold", "y_true", "y_pred", "baseline"],
    )


def test_evaluate_postproc_config_returns_lower_rmse_with_smoothing() -> None:
    oof = _make_smooth_oof_df(n_wells=2, n_rows=100)
    raw_cfg = PostprocConfig(savgol_window=None, savgol_polyorder=0)
    smooth_cfg = PostprocConfig(savgol_window=11, savgol_polyorder=2)

    raw_result = evaluate_postproc_config(oof, raw_cfg)
    smooth_result = evaluate_postproc_config(oof, smooth_cfg)

    assert raw_result.rmse > 0
    # Smoothing should reduce RMSE on smooth signal with noise
    assert smooth_result.rmse < raw_result.rmse


def test_evaluate_postproc_config_with_clipping() -> None:
    oof = _make_oof_df(n_wells=2, n_rows_per_well=30)
    y_full = oof["y_pred"] + oof["baseline"]
    clip_lower = float(np.percentile(y_full, 10))
    clip_upper = float(np.percentile(y_full, 90))

    clip_cfg = PostprocConfig(
        savgol_window=None, savgol_polyorder=0,
        clip_lower=clip_lower, clip_upper=clip_upper,
    )
    result = evaluate_postproc_config(oof, clip_cfg)
    assert result.rmse > 0


def test_evaluate_postproc_config_delta_vs_raw() -> None:
    oof = _make_oof_df(n_wells=2, n_rows_per_well=30)
    raw_cfg = PostprocConfig(savgol_window=None, savgol_polyorder=0)
    result = evaluate_postproc_config(oof, raw_cfg)
    assert result.delta_vs_raw == 0.0


def test_evaluate_postproc_config_smooth_clip_order() -> None:
    oof = _make_oof_df(n_wells=2, n_rows_per_well=30)
    cfg = PostprocConfig(
        savgol_window=11, savgol_polyorder=2,
        clip_lower=8000.0, clip_upper=15000.0,
        apply_order="smooth_clip",
    )
    result = evaluate_postproc_config(oof, cfg)
    assert result.rmse > 0


# ---------------------------------------------------------------------------
# grid_search_postprocessing
# ---------------------------------------------------------------------------

def test_grid_search_returns_sorted_results() -> None:
    oof = _make_smooth_oof_df(n_wells=2, n_rows=100)
    grid = PostprocParamGrid(
        savgol_windows=(5, 11),
        savgol_polyorders=(2,),
        clip_percentiles=(),
    )
    results = grid_search_postprocessing(oof, grid, {})
    for i in range(len(results) - 1):
        assert results[i].rmse <= results[i + 1].rmse


def test_grid_search_includes_raw_baseline() -> None:
    oof = _make_smooth_oof_df(n_wells=2, n_rows=50)
    grid = PostprocParamGrid()
    results = grid_search_postprocessing(oof, grid, {})
    raw_config = PostprocConfig(savgol_window=None, savgol_polyorder=0)
    assert any(r.config == raw_config for r in results)


def test_grid_search_with_clip_bounds() -> None:
    oof = _make_smooth_oof_df(n_wells=2, n_rows=50)
    grid = PostprocParamGrid(
        savgol_windows=(5,),
        savgol_polyorders=(2,),
        clip_percentiles=((0.1, 99.9),),
    )
    bounds_map = {"p0.1-p99.9": (5000.0, 25000.0)}
    results = grid_search_postprocessing(oof, grid, bounds_map)
    assert len(results) > 2
    assert any(r.config.clip_lower == 5000.0 for r in results)


# ---------------------------------------------------------------------------
# select_best_postproc
# ---------------------------------------------------------------------------

def test_select_best_chooses_better_config() -> None:
    baseline = PostprocConfig(savgol_window=31, savgol_polyorder=2)
    better = PostprocConfig(savgol_window=51, savgol_polyorder=3)
    results = [
        PostprocResult(config=better, rmse=10.0, delta_vs_raw=-1.0),
        PostprocResult(config=baseline, rmse=10.5, delta_vs_raw=-0.5),
    ]
    selected = select_best_postproc(results, baseline, min_delta=0.01)
    assert selected == better


def test_select_best_falls_back_when_improvement_small() -> None:
    baseline = PostprocConfig(savgol_window=31, savgol_polyorder=2)
    barely_better = PostprocConfig(savgol_window=51, savgol_polyorder=3)
    results = [
        PostprocResult(config=barely_better, rmse=10.499, delta_vs_raw=-0.001),
        PostprocResult(config=baseline, rmse=10.500, delta_vs_raw=0.0),
    ]
    selected = select_best_postproc(results, baseline, min_delta=0.01)
    assert selected == baseline


def test_select_best_returns_best_when_baseline_not_found() -> None:
    baseline = PostprocConfig(savgol_window=31, savgol_polyorder=2)
    cfg = PostprocConfig(savgol_window=51, savgol_polyorder=3)
    results = [PostprocResult(config=cfg, rmse=10.0)]
    selected = select_best_postproc(results, baseline, min_delta=0.01)
    assert selected == cfg


def test_select_best_returns_baseline_on_empty() -> None:
    baseline = PostprocConfig(savgol_window=31, savgol_polyorder=2)
    selected = select_best_postproc([], baseline)
    assert selected == baseline


# ---------------------------------------------------------------------------
# apply_postprocessing (DataFrame-based)
# ---------------------------------------------------------------------------

def test_apply_postprocessing_savgol_only() -> None:
    submission = pd.DataFrame({"id": ["well_a_0", "well_a_1", "well_a_2"], "tvt": [10000.0, 10100.0, 10200.0]})
    result = apply_postprocessing(submission, savgol_window=3, savgol_polyorder=1)
    assert list(result.columns) == ["id", "tvt"]
    assert len(result) == 3


def test_apply_postprocessing_clip_only() -> None:
    submission = pd.DataFrame({"id": ["well_a_0", "well_a_1"], "tvt": [100.0, 20000.0]})
    result = apply_postprocessing(submission, savgol_window=None,
                                  clip_lower=1000.0, clip_upper=15000.0)
    assert result["tvt"].iloc[0] == 1000.0
    assert result["tvt"].iloc[1] == 15000.0


def test_apply_postprocessing_noop() -> None:
    submission = pd.DataFrame({"id": ["well_a_0"], "tvt": [5000.0]})
    result = apply_postprocessing(submission)
    assert result["tvt"].iloc[0] == 5000.0
