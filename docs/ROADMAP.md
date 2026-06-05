# Roadmap

## Purpose

Define the post-baseline development plan after the first Kaggle baseline has been frozen.

## Owns

Post-baseline improvement stages, priorities, promotion gates, high-risk deferred ideas and the short-term execution plan.

## Update when

- A roadmap stage is started, completed, promoted, rejected or reprioritized.
- A new public-notebook-derived idea is accepted into the project plan.
- A roadmap gate, constraint or verification rule changes.

## Do not store here

- Full experiment history; use `docs/EXPERIMENT_LOG.md`.
- Detailed baseline metrics and commands; use `docs/BASELINE_PLAN.md`.
- Raw public notebook code; use `docs/PUBLIC_NOTEBOOK_REFERENCES.md` for review notes.
- Data schema facts; use `docs/DATA_MAP.md`.

## Current content

Current best clean baseline:

| Item | Value |
|---|---|
| Stage | **A2a** (DWT) |
| Model | LightGBM |
| Features | **20 features**: 6 base + 9 geometry + 3 GR + 2 DWT |
| Target | `residual = TVT - last_tvt_input` |
| Validation | 5-fold `GroupKFold` by well |
| Local CV | RMSE `14.13 ± 0.77` |
| Public LB | pending |
| Explanation | `docs/HOW_IT_WORKS.md` |

Tabular feature ceiling confirmed at CV ~14.13. All subsequent feature blocks (spatial KNN, DTW, target engineering, geology v1/v2) produced flat or degraded CV. Active development focus shifts to architectural improvements (CNN, ensemble).

## Guiding constraints

- Keep core logic in `src/rogii/` and executable entry points in `scripts/`.
- Keep Kaggle notebooks thin runners.
- Keep `GroupKFold` by well as the primary validation strategy unless a documented decision changes it.
- Treat `TVT_input` as allowed only up to Prediction Start; never use post-PS labels or target-derived columns as features.
- Do not use public saved model artifacts, TabICL artifacts or exact train/test coordinate overlap blending in the clean mainline roadmap.
- Add new dependencies only as approved staged dependencies: `PyWavelets`, `scipy`, `catboost` and optionally `torch`.
- Promote a stage only after tests pass, leakage review passes, submission validation passes, and CV improves or the stage produces useful diagnostics.
- Kaggle submission is always manual. The project may generate and validate `submission.csv`, but the user performs the actual submit.
- After a code push intended for Kaggle: `02_kaggle_update_repo` (Pull from GitHub → Run → Create Dataset `rogii-repo`) → `01_kaggle_train` if model changed (Pull from GitHub → Run → Create Dataset `rogii-models`) → `00_kaggle_inference` (Run, internet OFF → Submit). Notebooks `01` and `02` are GitHub-linked (setup once via File → Link to GitHub). See `.agents/skills/kaggle-runner/SKILL.md`.

## Standard Gate After Each Stage

1. Run `python -m pytest tests`.
2. Run full 5-fold GroupKFold CV for the candidate stage.
3. Compare against R1 optimized and the latest promoted stage.
4. Run leakage review for every new feature family or target transform.
5. Generate a candidate submission only in an ignored runtime path such as `outputs/submission.csv` or `/kaggle/working/submission.csv`.
6. Run `python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv` locally or the Kaggle-equivalent command in the notebook.
7. Update `docs/EXPERIMENT_LOG.md` after meaningful runs.
8. Update `docs/ROADMAP.md`, `docs/TASKS.md`, `docs/VALIDATION_STRATEGY.md` or `docs/KNOWN_ISSUES.md` when stage status, validation contracts or risks change.
9. Do not commit `data/`, `outputs/`, `models/`, `submissions/`, `mlruns/`, OOF artifacts or generated submissions.
10. After a code push: open `02` → Pull from GitHub → Run → Create Dataset `rogii-repo` → if model changed, open `01` → Pull from GitHub → Run → Create Dataset `rogii-models` → open `00` → Run (offline) → Submit.

## Stage A0: Pipeline Contracts and Roadmap Reset

Goal: make the new staged roadmap executable before adding high-risk features.

Status: Done. Train can load `configs/baseline_lgbm.yaml`; saved model payloads include feature flags and exact feature columns; predict validates the generated feature matrix against the saved payload before writing a submission.

Scope:

- Replace old pending roadmap items with stages A1-A4.
- Keep R1 optimized as the active baseline and Stage 4 as the historical frozen baseline.
- Ensure model payloads carry enough metadata to reproduce prediction: feature flags, target mode, feature columns and run name.
- Sync README and config examples with the currently supported CLI commands.
- Keep Kaggle workflow: training notebook (`01_kaggle_train.ipynb`) saves model to `rogii-models` Kaggle Dataset; inference notebook (`00_kaggle_inference.ipynb`) loads model and generates submission (ADR-007).

Primary files:

- `docs/ROADMAP.md`
- `docs/TASKS.md`
- `docs/DECISIONS.md`
- `docs/KNOWN_ISSUES.md`
- `README.md`
- `configs/baseline_lgbm.yaml`
- `src/rogii/config.py`
- `src/rogii/model_io.py`
- `src/rogii/train.py`
- `src/rogii/predict.py`
- `scripts/run_train.py`
- `scripts/run_predict.py`
- `tests/test_model_io.py`
- `tests/test_predict_contract.py`

Verification:

- `python -m pytest tests`
- `python scripts/run_train.py --help`
- `python scripts/run_predict.py --help`
- `PYTHONPATH=src python -c "from rogii.config import load_yaml_config; c=load_yaml_config('configs/baseline_lgbm.yaml'); assert c['features']['residual_target'] is True"` or PowerShell equivalent.
- `git diff -- docs/ROADMAP.md docs/TASKS.md docs/DECISIONS.md docs/KNOWN_ISSUES.md README.md`

Promotion gate:

- New roadmap is the only active development plan and old pending tasks are explicitly cancelled or replaced.

## Stage A1: Spatial Kinematics and Trajectory Geometry

Status: **Rejected.** CV 14.24 (flat vs R1 14.19), Kaggle LB 12.487 (worse vs R1 12.247). All 5 trajectory features are near-perfect linear transforms of existing geometry features (r >= 0.99). No new signal — only importance redistribution across duplicates. Code remains in `features.py` for future experiments.

Goal: improve the tree model's representation of 3D trajectory curves without changing the model family.

Code remaining in repo (`TRAJECTORY_FEATURES`, `build_trajectory_features()`, `include_trajectory` flag). Rejection reason: all 5 features are linear duplicates of `GEOMETRY_FEATURES` (r >= 0.99 with `dz_since_ps`, `dzdmd`, `dxdmd`, `dydmd`). `dogleg_severity_10m` was ablated mid-experiment (importance 0.00).

## Stage A2: GR DWT and Strict OOF Spatial KNN

Status: **Partially promoted.** A2a (DWT) promoted (+0.06 CV). A2b (spatial KNN) not promoted (flat CV, −0.02 vs R1). No leakage detected in either block.

Goal: extract deeper GR structure and add safe inter-well spatial context without target leakage.

### Stage A2a: Causal GR DWT — **Promoted**

CV: 14.13 (R1: 14.19, +0.06). Runtime: ~1.4 min full train.

Feature scope:

- Added dependency `PyWavelets`.
- `gr_dwt_approx`: low-frequency GR approximation via trailing-window DWT (db4, window=256).
- `gr_dwt_detail_energy`: mean squared detail coefficient energy from same window.
- Causal only: feature at row `i` depends only on `GR[0:i+1]` (verified by test).

Primary files:

- `requirements.txt` — added `pywavelets`
- `src/rogii/gr_dwt.py` — new module
- `src/rogii/features.py` — `GR_DWT_FEATURES` constant, `build_gr_dwt_features()` integration
- `src/rogii/model_io.py` — `include_gr_dwt` flag
- `tests/test_feature_engineering.py` — 8 DWT tests including causal verification
- `configs/a2_lgbm.yaml` — DWT config

### Stage A2b: Strict OOF Spatial KNN — **Not promoted (flat CV)**

CV: 14.21 (R1: 14.19, −0.02). No leakage (CV not implausibly low). Spatial features add no new signal beyond what X/Y/Z coordinates already encode for LightGBM.

Feature scope implemented:

- KNN for `k = 5`, `10`, `50` in 3D space using `sklearn.neighbors.NearestNeighbors` (ball_tree).
- `spatial_nn{k}_mean_tvt`, `spatial_nn{k}_median_tvt`, `spatial_nn{k}_std_tvt`.
- Reference: pre-PS rows from other wells, `TVT_input` only.

OOF contract (all verified by tests):

- For validation fold K, reference tree built only from wells outside fold K.
- Reference rows are only pre-PS rows.
- Reference target is pre-PS known `TVT_input`, not post-PS `TVT`.
- Validation wells never appear in their own spatial reference tree.
- Test-time tree uses all train pre-PS rows.

Primary files:

- `src/rogii/spatial_features.py` — new module
- `src/rogii/train.py` — fold-aware spatial feature building
- `src/rogii/predict.py` — `_run_predict_with_spatial`
- `tests/test_spatial_oof.py` — 9 tests including OOF leakage checks

## Stage A3: DTW Typewell Alignment and Target Engineering

Status: **Rejected.** All three sub-stages degraded CV vs A2a (14.13).

- A3a DTW: CV 14.63 (+0.50). Median GR cross-correlation 0.43 — insufficient signal. Code kept in `src/rogii/typewell_alignment.py`.
- A3b.1 Signed-log: CV 14.64 (+0.51). Residuals not heavy-tailed (bounded ±40, skew 0.74).
- A3b.2 Derivative model: CV 14.32 (+0.19). Integration error accumulation negates any slope-prediction benefit.

## Stage A4: Structural Blending and Pipeline-Dependent Models

Goal: improve generalization through diverse models and standardized OOF predictions.

Scope:

- Standardize OOF prediction artifacts for stacking under ignored runtime directories.
- Add multi-seed LightGBM averaging, starting with seeds `42`, `7`, `123`.
- Add CatBoost with Ordered Boosting after dependency installation is verified.
- Fit CatBoost on the same folds and feature set as the promoted tabular baseline.
- Add Ridge stacking or weighted averaging over OOF predictions.
- Consider a 1D CNN only after separate approval of the optional `torch` dependency and runtime budget.

Primary files:

- New `src/rogii/oof.py`
- New `src/rogii/ensemble.py`
- `src/rogii/models.py`
- `src/rogii/train.py`
- `src/rogii/predict.py`
- `scripts/run_train.py`
- `scripts/run_predict.py`
- `requirements.txt`
- `tests/test_ensemble.py`

Verification:

- `python -m pytest tests`
- Compare single-model OOF RMSE, multi-seed average OOF RMSE and stacking OOF RMSE.
- Validate that OOF artifacts contain no raw target columns beyond allowed prediction/target arrays needed for local scoring.
- Generate and validate submission from the final blended model.

Promotion gate:

- Promote only if ensemble OOF improves over the best single model and Kaggle runtime remains acceptable.

## Stop Criteria

| Signal | Action |
|---|---|
| Spatial KNN CV drops to implausibly low RMSE such as `2-3` | Stop, disable spatial features, run leakage audit |
| CV improves but public LB sharply worsens | Do not promote; document CV/LB gap |
| Feature block degrades CV by more than `0.3` RMSE without useful diagnostics | Disable or keep as rejected experiment |
| Target transform improves transformed-space score but worsens TVT RMSE | Revert to plain residual target |
| Predicted TVT variance collapses | Revert target/model change and inspect per-well predictions |
| Runtime exceeds Kaggle limits | Simplify or defer the block |
| New dependency is unstable in Kaggle | Revert dependency and keep branch local until resolved |

## Rejected or Deferred Ideas

| Idea | Status | Reason |
|---|---|---|
| Simple Typewell V1 anchor residuals | Rejected | CV degraded and features were mostly `GR - const` redundancy |
| Trajectory features (A1) | Rejected | r >= 0.99 with geometry features, CV flat, LB worse |
| Spatial KNN (A2b) | Not promoted | Flat CV (−0.02), no leakage, code kept |
| DTW typewell alignment (A3a) | Rejected | CV +0.50, GR cross-correlation only 0.43 |
| Signed-log target (A3b.1) | Rejected | CV +0.51, residuals not heavy-tailed |
| Derivative target (A3b.2) | Rejected | CV +0.19, integration error accumulation |
| Geology v1 well-level (A4) | Rejected | CV +0.44, signal unstable across folds |
| Geology v2 per-row (A4) | Not promoted | CV −0.04 flat, code kept |
| Public saved artifacts / TabICL artifact stack | Rejected for clean mainline | Opaque external artifacts and heavy dependency path |
| Exact train/test coordinate overlap blend | Rejected for clean mainline | High leakage risk; diagnostics only |
| Particle filters | Deferred | Complex and stochastic |
| **1D CNN sequence model** | **Deferred for A4+** | Architecture diversity, not feature; revisit after tabular ceiling |
| **Multi-seed LGBM + CatBoost + stacking** | **Deferred for A4+** | Cosmetic ensemble on same features |
