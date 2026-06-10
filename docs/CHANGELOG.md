# Changelog

## 2026-06-10 - B4 Optuna-tuned baseline

### Added
- `scripts/run_tune.py` — CLI entry point for Optuna hyperparameter tuning.
- `src/rogii/tuning.py` — Optuna TPESampler tuning with 2-fold screening + 5-fold top-3 verification, disk caching via `joblib.hash`.
- `configs/b4_tuned.yaml` — tuning config: 30 trials, 120 min timeout, 2-fold screening.
- `configs/b4_best_params.yaml` — best params: lr=0.0664, num_leaves=48, min_child_samples=60, subsample=0.716, colsample_bytree=0.733.
- `tests/test_tuning.py` — tests for tuning pipeline, cache and config.

### Changed
- `src/rogii/train.py` — added `TuningRunner` integration; `run_train.py` supports `--best-params` to load tuned params.
- `scripts/run_train.py` — added `--best-params` and `--use-tuned` CLI flags.
- `configs/a4_multiseed.yaml` — updated to reference tuned param defaults.
- `AGENTS.md` — updated active baseline to B4 (CV 13.948, LB TBD).
- `docs/BASELINE_PLAN.md` — promoted B4 as canonical active baseline, moved R3 to superseded status.
- `docs/ROADMAP.md` — updated current baseline to B4.
- `docs/EXPERIMENT_LOG.md` — recorded B4 experiment: CV 13.948 ± 0.764 (−0.104 vs R3).
- `docs/DECISIONS.md` — added ADR-020 Optuna HPO decision.
- `docs/CHANGELOG.md` — this entry.

### Verification
- `python -m pytest tests` — 226 passed.
- `python scripts/run_tune.py --config configs/b4_tuned.yaml --data-dir data` — completed 30 trials, top-3 verified on 5-fold.
- Best params CV: 13.948 ± 0.764 vs R3 14.052 ± 0.868 (−0.104, std improved).
- OOF + Savgol w=31 p=2: 13.965.

## 2026-06-10 - Linear MCP task workflow

### Changed
- Made Linear MCP (`ROG-*` issues) the centralized source of truth for current tasks, status, blockers and next actions.
- Marked `docs/TASKS.md` as a read-only historical pre-Linear archive with no backfill of old completed tasks.
- Updated agent docs and workflow skills so future agents do not mirror Linear state into markdown.
- Set the authenticated Linear user from `linear_linear_getViewer` as the default assignee for Linear issues created or worked by the agent.
- Documented the Linear GraphQL PowerShell fallback for MCP optional-UUID validation failures and UTF-8-safe Russian issue content.

### Verification
- Docs-only change; no code tests required.

## 2026-06-08 - TCN tuning validation alignment

### Changed
- `scripts/tune_tcn.py` - replaced the accidental 50/50 tuning split with fold-selectable 5-fold `GroupKFold`, added dense validation RMSE monitoring and final `run_train.py` command output.
- `src/rogii/train.py` - TCN CV now fits target scaling on each train fold and stores an all-train scaler for the final model.
- `scripts/run_train.py`, `scripts/run_predict.py`, `src/rogii/model_io.py`, `src/rogii/predict.py` - propagated TCN architecture and baseline metadata needed for Kaggle transfer.
- `src/rogii/data_loading.py` - disabled pandas Arrow string inference for CSV loading and inserts `well_id` as object dtype to avoid local pyarrow-backed string access violations.
- `docs/EXPERIMENT_LOG.md` - recorded `a5_tcn_tuned_small` full 5-fold tuner validation.

### Verification
- `python -m py_compile scripts/tune_tcn.py scripts/run_train.py scripts/run_predict.py`
- `python -m pytest tests/test_model_io.py tests/test_validation_split.py`
- `python -m pytest tests/test_tcn_pipeline.py tests/test_tcn_model.py`
- `python -m pytest tests` - 210 passed.
- `python scripts/tune_tcn.py --folds all` - best tuned small TCN CV `15.036 ± 0.848`.

## 2026-06-08 - Documentation state refresh

### Changed
- Synchronized README and source-of-truth docs with the current active baseline: R3 3-seed LightGBM + Savgol, CV `14.052`, LB `12.177`.
- Updated data, validation, architecture, roadmap, task, experiment, risk and decision docs to reflect implemented A5 TCN/OOF/diagnostics work and pending Phase 2 training gate.
- Marked stale A2a/R2 references as historical or superseded where applicable.

### Verification
- `git diff --check` - no whitespace errors.
- `python -m pytest tests` - 226 passed, 105 PyWavelets boundary-effect warnings.

## 2026-06-07 - OpenCode project guard hooks

### Added
- `.opencode/opencode.json` - loads project-specific OpenCode hooks.
- `.opencode/plugin/rogii-guards.ts` - blocks staged runtime artifacts/secrets on Git commit, tracked artifacts on Git push and Kaggle submit commands without `ROGII_ALLOW_KAGGLE_SUBMIT=1`; emits reminders for submission validation, leakage-sensitive edits, training logs and relevant tests.

### Changed
- `docs/CONTEXT_MAP.md` - added OpenCode guard locations for future agents.

### Verification
- JSON syntax check for `.opencode/opencode.json`.
- TypeScript parse check for `.opencode/plugin/rogii-guards.ts`.
- Runtime smoke check confirmed unapproved `kaggle competitions submit` is blocked.

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
