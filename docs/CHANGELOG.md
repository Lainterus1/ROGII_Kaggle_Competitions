# Changelog

## 2026-06-05 — Stage A2-A4: Feature experiments

### Added
- `src/rogii/gr_dwt.py` — causal GR DWT features (PyWavelets, db4, window=256). Two features: `gr_dwt_approx`, `gr_dwt_detail_energy`. **Promoted (A2a, CV +0.06).**
- `src/rogii/spatial_features.py` — strict OOF spatial KNN (k=5,10,50) with ball_tree. Nine features. **Not promoted (flat CV).**
- `src/rogii/typewell_alignment.py` — DTW typewell-horizontal GR alignment (Sakoe-Chiba window=50). Two features. **Rejected (CV +0.50).**
- `src/rogii/target.py` — signed-log and derivative target transforms. **Rejected.**
- `src/rogii/geology_features.py` — formation geology from typewell. v1 (well-level, 7 features). v2 (per-row GR z-scores, 9 features). **Rejected / not promoted.**
- `configs/a2_lgbm.yaml` — A2 config (DWT + spatial combined).
- `tests/test_spatial_oof.py` — 9 OOF leakage tests.
- `tests/test_dtw_features.py` — 9 DTW tests.
- `tests/test_geology_features.py` — 8 geology tests.
- `requirements.txt` — added `scipy`, `pywavelets`.
- `docs/DECISIONS.md` — ADR-009 through ADR-012.
- `docs/CHANGELOG.md` — this file.

### Changed
- `src/rogii/features.py` — added `GR_DWT_FEATURES`, `DTW_FEATURES`, `GEOLOGY_FEATURES` constants; `build_features()` accepts `include_gr_dwt`, `include_dtw`, `include_geology`.
- `src/rogii/model_io.py` — `FEATURE_FLAG_KEYS` extended: `include_gr_dwt`, `include_spatial`, `include_dtw`, `include_geology`.
- `src/rogii/train.py` — `_collect_train_post_ps` returns `well_ids_used`; fold-aware spatial KNN in CV loop.
- `src/rogii/predict.py` — `_run_predict_with_spatial` path; all new flags propagated.
- `scripts/run_train.py`, `scripts/run_predict.py` — CLI flags for all new feature blocks.
- `configs/baseline_lgbm.yaml` — added `include_gr_dwt`, `include_spatial`, `include_dtw`, `include_geology` (all `false`).
- `tests/test_feature_engineering.py` — 8 DWT tests.
- `docs/ROADMAP.md`, `docs/TASKS.md`, `docs/HOW_IT_WORKS.md`, `docs/KNOWN_ISSUES.md`, `docs/EXPERIMENT_LOG.md` — full session documentation.
- `README.md` — updated active baseline and CLI examples.

### Active baseline
**A2a**: 20 features (6 base + 9 geometry + 3 GR + 2 DWT), residual target, CV 14.13 ± 0.77, LightGBM.

### Tabular ceiling
CV ~14.13 confirmed after 8 experiments (A1 trajectory, A2a DWT, A2b spatial, A3a DTW, A3b signed-log, A3b derivative, geology v1/v2). Only DWT gave marginal improvement (+0.06).
