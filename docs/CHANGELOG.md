# Changelog

## 2026-06-06 — A2a Kaggle candidate packaging + LB result

### Added
- `notebooks/kernels/a2a-dwt/` — candidate kernel folder with `kernel-metadata.json` and `00_kaggle_inference.ipynb` for offline A2a DWT inference.
- Kaggle datasets: `rogii-models-a2a-dwt` (trained A2a model), `rogii-wheels-a2a-dwt` (pywavelets 1.8.0 wheel).
- `wheels/`, `kaggle_datasets/` — added to `.gitignore`.

### Changed
- `models/a2a_dwt.pkl` — retrained with current code; CV `14.13 ± 0.77` confirmed.
- `docs/TASKS.md` — A2a submit task marked Done.
- `docs/KNOWN_ISSUES.md` — A2a pywavelets dependency marked Resolved.
- `docs/DECISIONS.md` — ADR-013 consequences updated: A2a no longer blocked.
- `docs/EXPERIMENT_LOG.md` — new entry for A2a Kaggle packaging.

### Verification
- `python -m pytest tests` — 113 passed.
- Kaggle kernel `daniilgonchar/00-rogii-inference-a2a-dwt` v1 produced validated `/kaggle/working/submission.csv` (14151 rows, 463970 bytes, 0 NaN, 0 Inf).
- Kaggle logs confirmed: pywavelets found, repo/model/data paths resolved, 20 DWT features used, residual prediction mode.

## 2026-06-06 - Kaggle offline inference workflow repair

### Added
- `src/rogii/kaggle_runtime.py` - marker-based Kaggle repo/model/data discovery and output checks.
- `scripts/kaggle_offline_inference.py` - CLI wrapper for offline Kaggle inference.
- `notebooks/kernel-metadata.json` - versioned metadata for `daniilgonchar/00-rogii-inference-r1` with internet OFF and fixed dataset/competition inputs.
- `tests/test_kaggle_runtime.py` - tests for flat/nested repo datasets, hidden-style data roots, preferred model dataset selection and non-empty output validation.

### Changed
- `notebooks/00_kaggle_inference.ipynb` - removed hardcoded nested dataset path and shell-copy flow; now runs fail-fast marker-based offline inference.
- `notebooks/01_kaggle_train.ipynb` - aligned stable R1 training artifact name to `baseline_lgbm.pkl` for `rogii-models-v2`.
- `notebooks/02_kaggle_update_repo.ipynb` - aligned instructions with `rogii-repo-v2`.
- `scripts/kaggle_runner.py` - delegates to the implemented offline inference runner instead of a placeholder error.
- `configs/a2_lgbm.yaml` - corrected A2a DWT-only config by disabling spatial features.
- `.agents/skills/kaggle-runner/SKILL.md` - replaced stale manual `rogii-repo` workflow with metadata-driven `rogii-repo-v2`/`rogii-models-v2` kernel-version submit workflow.
- `.agents/skills/kaggle-runner/SKILL.md`, `docs/DECISIONS.md`, `docs/ROADMAP.md`, `README.md` - added a generic candidate build workflow for A2a and future variants, including separate model/dependency/kernel artifacts.
- `.agents/skills/kaggle-candidate-build/SKILL.md` - added strict candidate packaging standard for future Kaggle builds.

### Verification
- `python -m pytest tests` - 113 passed.
- Kaggle kernel `daniilgonchar/00-rogii-inference-r1` version 3 produced validated `/kaggle/working/submission.csv`.
- Submission `53410572` confirmed the fixed R1 workflow public LB: `12.247`.

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
