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
| Stage | **PrP3** (R1 + Savgol w=31 p=2), LB 12.239 |
| R1 | 18 tabular features, LB 12.247 |
| A2a | CV 14.13, LB 12.558 — DWT does not generalise, superseded |
| Tabular ceiling | LB ~12.2, all A1-A4 experiments flat or degraded |

Active development: **Tabular ceiling confirmed** — all feature families exhausted. Next: architecture diversity (CNN, ensemble).

## Guiding constraints

- Keep core logic in `src/rogii/` and executable entry points in `scripts/`.
- Keep Kaggle notebooks thin runners.
- Keep `GroupKFold` by well as the primary validation strategy unless a documented decision changes it.
- Treat `TVT_input` as allowed only up to Prediction Start; never use post-PS labels or target-derived columns as features.
- Do not use public saved model artifacts, TabICL artifacts or exact train/test coordinate overlap blending in the clean mainline roadmap.
- Add new dependencies only as approved staged dependencies: `PyWavelets`, `scipy`, `catboost` and optionally `torch`.
- Promote a stage only after tests pass, leakage review passes, submission validation passes, and CV improves or the stage produces useful diagnostics.
- Kaggle submission requires explicit user approval. After approval, the agent may submit a validated kernel version through Kaggle CLI/API.
- Current stable offline submit path: `00-rogii-inference-r1` uses `rogii-repo-v2` + `rogii-models-v2`, internet OFF, and `notebooks/kernel-metadata.json` for `kaggle kernels push -p notebooks`.
- For candidate builds, do not overwrite R1 fallback artifacts. Use candidate-specific model/dependency datasets and kernel metadata. Update `rogii-repo-v2` first if source code changed.

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
10. For R1 recovery submissions: push `00` with `kaggle kernels push -p notebooks`, validate kernel output, then submit the kernel version with `kaggle competitions submit -k daniilgonchar/00-rogii-inference-r1 -v <version> -f submission.csv` after explicit approval.
11. For candidate submissions: update repo dataset if code changed, train/upload a candidate model dataset, attach offline dependency dataset if needed, push a candidate kernel version, validate output, then submit the candidate kernel version after explicit approval.

## Stage A0: Pipeline Contracts and Roadmap Reset

Goal: make the new staged roadmap executable before adding high-risk features.

Status: Done. Train can load `configs/baseline_lgbm.yaml`; saved model payloads include feature flags and exact feature columns; predict validates the generated feature matrix against the saved payload before writing a submission.

Scope:

- Replace old pending roadmap items with stages A1-A4.
- Keep R1 optimized as the active baseline and Stage 4 as the historical frozen baseline.
- Ensure model payloads carry enough metadata to reproduce prediction: feature flags, target mode, feature columns and run name.
- Sync README and config examples with the currently supported CLI commands.
- Keep Kaggle workflow: training notebook (`01_kaggle_train.ipynb`) saves R1 model to `rogii-models-v2`; inference notebook (`00_kaggle_inference.ipynb`) loads `rogii-repo-v2` + `rogii-models-v2` and generates a validated submission (ADR-007, ADR-013).

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

## Stage B1: Beam Search Stratigraphic Alignment

Status: **Rejected.** CV 14.43 (5-fold) vs R1 14.19 — worse by +0.24. `beam_std` is #2 feature (6.2% importance) but cannibalizes X/Y/Z spatial importance: X 17%→5.6%, Y 14.9%→5.5%, Z 12.5%→6.0%. Beam-only (no geometry/GR) scores CV 16.02 — worse than naive baseline (15.91). Same pattern as typewell/DTW/geology — typewell-referenced features redistribute without net gain. Code kept in `src/rogii/beam_search.py`, feature flag `include_beam` available for future compound experiments.

Goal: implement Numba JIT beam search along typewell GR to produce TVT alignment signals that close the gap between our tabular baseline (LB 12.2, CV 14.1) and top public solutions (LB 5.9–7.5).

### Why

All top public solutions (Roman Tamrazov sub-9, Ravaghi hill-climbing, Pilkwang EDA) use physics-based stratigraphic alignment (beam search, particle filters, NCC) rather than pure tabular features. Our tabular approach is confirmed saturated at LB 12.2 / CV 14.1. Beam search is the strongest single alignment signal observed in public code.

### Scope

- `src/rogii/beam_search.py` — Numba JIT beam search kernel (`_beam_jit`) + feature builder (`build_beam_features`).
- 7 beam configs with diverse move_cost / emit_scale / smooth_radius.
- ~19 features: per-config TVT estimates, consensus (avg cons+sm5), std, consensus differences at 11 offsets (−40..+40).
- `include_beam` flag in model payload v2 (`FEATURE_FLAG_KEYS`).
- Optional GPU for LightGBM training (auto-detect, fallback CPU). Beam search runs on CPU via Numba JIT.

### Dependencies

- `numba` (new, approved) — CPU JIT compilation for beam search.
- `scipy` (already approved).

### Primary files

- `src/rogii/beam_search.py` — new module
- `src/rogii/features.py` — `BEAM_FEATURES`, `include_beam` integration
- `src/rogii/model_io.py` — `include_beam` flag in `FEATURE_FLAG_KEYS`
- `scripts/run_train.py`, `scripts/run_predict.py` — `--include-beam` CLI flag
- `configs/b1_lgbm.yaml` — beam config
- `tests/test_beam_search.py` — 9 tests including causal verification
- `requirements.txt` — added `numba`

### Verification

- `python -m pytest tests/test_beam_search.py` — 9 tests including JIT compilation, causal construction, shape checks
- `python scripts/run_train.py --config configs/b1_lgbm.yaml --data-dir data`
- `python scripts/run_predict.py --data-dir data --model models/b1_lgbm.pkl --output outputs/submission.csv`
- `python scripts/validate_submission.py --submission outputs/submission.csv`
- Kaggle: `kaggle kernels push -p notebooks/kernels/b1-beam/` (with numba offline)

### Promotion gate

- CV improves by > 1.0 RMSE vs R1 (14.19) → target < 13.0.
- Beam search is causal (verified by `test_beam_causal_construction`).
- Numba JIT compiles without errors on Kaggle.
- Runtime < 10 min full train locally.

## Stage PrP3: Post-Processing Pipeline (Savgol Smoothing + TVT Clipping)

Goal: Improve CV/LB through per-well post-processing of predicted TVT sequences without changing model architecture or features.

Status: **Promoted. Savgol w=31 p=2 is the new active baseline.** LB 12.239 (−0.008 vs R1 12.247).

Results (OOF 5-fold CV, 3.78M rows, 773 wells):

| Config | OOF RMSE | vs Raw |
|--------|----------|--------|
| **Savgol w=31 p=2** | **14.2123** | **−0.0064** |
| Savgol w=25 p=2 | 14.2128 | −0.0059 |
| Savgol w=17 p=3 | 14.2135 | −0.0052 |
| Savgol w=11 p=2 | 14.2141 | −0.0046 |
| Savgol w=5 p=2 | 14.2154 | −0.0033 |
| Raw (no postproc) | 14.2187 | reference |
| Clip p0.1-p99.9 + Savgol w=31 | 14.2208 | +0.0021 |

TVT clip bounds (p0.1-p99.9): [9851.80, 12860.23]. Only 0.2% of data outside.
Per-well visualization: 3/3 wells improved, max raw jump 1.6-4.1 ft (noise, not geology).

Decision: **Savgol w=31 p=2 PROMOTED. TVT clipping REJECTED.** Savgol is now the recommended default post-processing step. Default params updated: window=17→31, polyorder=3→2.

Scope:

- `src/rogii/smoothing.py` — added `clip_predictions()`, `compute_tvt_clip_bounds()`, `apply_postprocessing()` (clip → smooth chain).
- `src/rogii/train.py` — CV loop collects per-well OOF predictions; `evaluate_postprocessing()` tests all Savgol window/polyorder/clip bound combos and ranks by OOF RMSE.
- `scripts/run_train.py` — added `--eval-postproc` flag.
- `scripts/run_predict.py` — added `--savgol-window`, `--savgol-polyorder`, `--tvt-clip` flags. Auto-detects clip bounds from model payload.
- `scripts/inspect_tvt_range.py` — new diagnostic script for TVT percentile analysis.
- `scripts/visualize_postproc.py` — new per-well visualization script with continuity checks.
- `src/rogii/model_io.py` — `clip_lower` / `clip_upper` stored in model payload v2.
- `tests/test_baseline.py` — 7 new clipping + apply_postprocessing tests (17 total).

Tunable parameters:

| Param | Default | Grid search |
|---|---|---|
| Savgol window | 17 | [5, 11, 17, 25, 31] |
| Savgol polyorder | 3 | [2, 3] |
| Clip bounds | p0.1–p99.9 from train TVT | none, p0.1-p99.9, p0.5-p99.5, p1-p99 |
| Order | clip → smooth | fixed |

Primary files:

- `src/rogii/smoothing.py`
- `src/rogii/train.py`
- `src/rogii/model_io.py`
- `scripts/run_train.py`
- `scripts/run_predict.py`
- `scripts/inspect_tvt_range.py`
- `scripts/visualize_postproc.py`
- `tests/test_baseline.py`

Verification:

- `python -m pytest tests/test_baseline.py` — 17 tests pass
- `python scripts/inspect_tvt_range.py --data-dir data` — prints TVT percentile table with suggested clip bounds
- `python scripts/run_train.py --config configs/baseline_lgbm.yaml --data-dir data --eval-postproc` — prints OOF post-processing evaluation table (top 15 configs sorted by RMSE)
- `python scripts/visualize_postproc.py --data-dir data --model models/r1_lgbm.pkl` — generates per-well comparison plots
- `python scripts/run_predict.py --data-dir data --model models/r1_lgbm.pkl --savgol-smooth --tvt-clip --output outputs/submission.csv` — generates clipped+smoothed submission

Promotion gate:

- OOF CV RMSE with best postproc config ≤ raw OOF CV RMSE (no degradation).
- Per-well visualization shows smoothing does not erase real formation boundaries.
- Continuity check: max jump ≤ 30 ft for ≥ 99% of adjacent points after smoothing.
- If CV improves: update active baseline config and LB-submit.
- If CV flat: keep code as optional pipeline, do not change active baseline.

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
| **Beam Search (B1)** | **Rejected** | CV 14.43 (worse than R1 14.19). beam_std — #2 feature but cannibalizes X/Y/Z. Beam-only CV 16.02. Typewell-GR alignment adds no net signal. |
| **Slope baseline (B2b)** | **Rejected** | Best CV 14.16 (slope_recent) flat vs R1. slope_md CV 284, wls CV 130. TVT-vs-MD trend does not extrapolate after PS — wells change direction. |
| **Formation Plane KNN (B3)** | **Rejected** | CV 14.99 (+0.80 vs R1). fp_knn_mean_dist #5 feature but cannibalizes X/Y/Z (35% importance from spatial coordinates). Well-level formation imputation via KNN adds no net signal. |
| **Z-Drift Physics (PrP2)** | **Not promoted** | CV 14.20 (flat vs R1 14.19, +0.01). 3 TVT-Z coupling features. Only offset_at_anchor is new signal; implied_tvt = Z+const, implied_tvt_resid = dz_since_ps (r=1.0). Fold inconsistency: fold 2 -0.80, fold 3 +0.62. Code kept behind include_z_drift flag. |
| **1D CNN sequence model** | **Deferred for A4+** | Architecture diversity, not feature; revisit after tabular ceiling |
| **Multi-seed LGBM + CatBoost + stacking** | **Deferred for A4+** | Cosmetic ensemble on same features |
