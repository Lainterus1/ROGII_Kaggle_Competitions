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
| Stage | R1 optimized (+A1 implemented, pending CV) |
| Model | LightGBM |
| Features | 24 features (with A1): 6 base + 9 geometry + 6 trajectory + 3 GR |
| Target | `residual = TVT - last_tvt_input` |
| Validation | 5-fold `GroupKFold` by well |
| Local/Kaggle CV | RMSE `~14.19` |
| Public LB | RMSE `12.247` |
| Explanation | `docs/HOW_IT_WORKS.md` |

The Stage 4 baseline remains frozen as the historical reference baseline in `docs/BASELINE_PLAN.md`. R1 optimized is the active comparison point for new work. The old pending model-upgrade roadmap is superseded by the staged plan below.

## Guiding constraints

- Keep core logic in `src/rogii/` and executable entry points in `scripts/`.
- Keep Kaggle notebooks thin runners.
- Keep `GroupKFold` by well as the primary validation strategy unless a documented decision changes it.
- Treat `TVT_input` as allowed only up to Prediction Start; never use post-PS labels or target-derived columns as features.
- Do not use public saved model artifacts, TabICL artifacts or exact train/test coordinate overlap blending in the clean mainline roadmap.
- Add new dependencies only as approved staged dependencies: `PyWavelets`, `scipy`, `catboost` and optionally `torch`.
- Promote a stage only after tests pass, leakage review passes, submission validation passes, and CV improves or the stage produces useful diagnostics.
- Kaggle submission is always manual. The project may generate and validate `submission.csv`, but the user performs the actual submit.
- After a code push intended for Kaggle, update the `rogii-repo` Kaggle Dataset, then run `01_kaggle_train.ipynb` (if model changed) and `00_kaggle_inference.ipynb` (for submission).

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
10. After a code push: update `rogii-repo` Kaggle Dataset → run `01_kaggle_train.ipynb` to rebuild model → update `rogii-models` Kaggle Dataset → run `00_kaggle_inference.ipynb` for submission.

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

Status: Implemented. Pending CV run and promotion gate.

Goal: improve the tree model's representation of 3D trajectory curves without changing the model family.

Feature scope (6 features, `z_ps_residual` excluded — already covered by `dz_since_ps` in `GEOMETRY_FEATURES`):

- `z_local_delta`: current `Z` minus the mean pre-PS `Z` for the same well.
- `dip_angle_proxy_10`: `(Z_i - Z_{i-10}) / (MD_i - MD_{i-10})`.
- `dogleg_severity_10m`: local 3D direction change over approximately 10m MD.
- `tortuosity_window_50`: arc length over approximately 50m MD divided by straight-line 3D distance.
- `sin_azimuth` and `cos_azimuth`: directional drilling azimuth encoded from `atan2(dY, dX)`.

Implementation notes:

- `TRAJECTORY_FEATURES` constant (6 features) and `build_trajectory_features()` in `features.py`.
- `include_trajectory` is a superset of `include_geometry`: setting `--include-trajectory` automatically includes geometry features. `include_geometry` remains as a legacy flag for backward compatibility (R1 models load without changes).
- Feature flag stored in model payload; predict auto-detects from payload — no CLI flags needed for inference.
- Resulting R1+A1 feature count: 6 base + 9 geometry + 6 trajectory + 3 GR = **24 features**.
- Zero MD deltas and early windows handled with `np.divide(out=zeros)` and finite fallbacks.

Primary files:

- `src/rogii/features.py` — `TRAJECTORY_FEATURES`, `build_trajectory_features()`
- `src/rogii/model_io.py` — `include_trajectory` in `FEATURE_FLAG_KEYS`
- `src/rogii/train.py` — `include_trajectory` parameter
- `src/rogii/predict.py` — `include_trajectory` parameter
- `scripts/run_train.py` — `--include-trajectory` CLI flag
- `scripts/run_predict.py` — `--include-trajectory` CLI flag
- `tests/test_feature_engineering.py` — 12 trajectory tests
- `tests/test_no_target_leakage.py` — leakage test for trajectory features

Verification:

- `python -m pytest tests` — 72 passed, 0 warnings.
- Candidate train command: `python scripts/run_train.py --data-dir data --n-splits 5 --seed 42 --include-trajectory --include-gr --residual-target --output-model models/a1_lgbm.pkl`
- Candidate predict command: `python scripts/run_predict.py --data-dir data --model models/a1_lgbm.pkl --output outputs/a1_submission.csv`
- Submission validation: `python scripts/validate_submission.py --data-dir data --submission outputs/a1_submission.csv`

Promotion gate:

- Promote only if leakage tests pass and CV improves over R1 optimized (~14.19 RMSE) or feature importance gives strong evidence for keeping the block.

## Stage A2: GR DWT and Strict OOF Spatial KNN

Goal: extract deeper GR structure and add safe inter-well spatial context without target leakage.

### Stage A2a: Causal GR DWT

Feature scope:

- Add approved dependency `PyWavelets`.
- `gr_dwt_approx`: low-frequency GR approximation.
- `gr_dwt_detail_energy`: trailing-window detail energy.

Implementation notes:

- Use causal or expanding/trailing windows only. Feature value at row `i` must not depend on `GR` after `i` unless a later ADR explicitly allows full-test-log context for this feature family.
- First run a runtime spike on a small subset before full CV.
- If DWT is too slow on full data, keep it as a diagnostic branch and do not promote.

Primary files:

- `requirements.txt`
- `src/rogii/features.py` or new `src/rogii/gr_features.py`
- `tests/test_feature_engineering.py`
- `tests/test_no_target_leakage.py`

### Stage A2b: Strict OOF Spatial KNN

Feature scope:

- Build KNN features for `k = 5`, `10`, `50` in 3D space.
- Add `spatial_nn{k}_mean_tvt`, `spatial_nn{k}_median_tvt`, `spatial_nn{k}_std_tvt`.

OOF contract:

- For validation fold K, build the reference tree only from wells outside fold K.
- Reference rows are only pre-PS rows.
- Reference target is pre-PS known `TVT_input`, not post-PS `TVT`.
- Validation wells must never appear in their own spatial reference tree.
- Test-time tree uses train pre-PS reference rows only by default; test pre-PS rows are excluded for a stricter train/test contract.

Primary files:

- New `src/rogii/spatial_features.py`
- `src/rogii/train.py`
- `src/rogii/predict.py`
- `src/rogii/validation.py`
- `tests/test_spatial_oof.py`
- `tests/test_no_target_leakage.py`

Verification:

- Tests prove validation groups are excluded from the tree.
- Tests prove post-PS target rows are excluded from the reference set.
- Tests prove target column `TVT` is not used to build reference values.
- Full CV must be inspected for suspicious leakage.

Rollback rule:

- If CV drops to an implausibly low range such as RMSE `2-3`, stop immediately, do not submit, disable the spatial block and audit OOF construction.

Promotion gate:

- Promote only if OOF leakage tests pass, CV improvement is plausible, and Kaggle LB does not show a severe CV/LB mismatch after manual submission.

## Stage A3: DTW Typewell Alignment and Target Engineering

Goal: revisit typewell information with elastic alignment and address residual-target flattening risk.

Prerequisite:

- Before implementation starts, create a clean rollback checkpoint after tests pass. Commit only after explicit user approval.

### Stage A3a: Typewell DTW Features

Feature scope:

- Align horizontal `GR` to typewell `GR` with Dynamic Time Warping or a Viterbi-style constrained path.
- Add `dtw_optimal_tvt` as an anchor-depth feature.
- Add `dtw_cost_cumulative` as a path-cost feature.

Rules:

- Do not use raw DTW output as the final answer.
- Do not use train post-PS `TVT` to guide alignment.
- Keep Typewell V1 rejected unless a new alignment approach beats R1/A-stage baselines.

Primary files:

- New `src/rogii/typewell_alignment.py`
- `src/rogii/features.py`
- `src/rogii/train.py`
- `src/rogii/predict.py`
- `tests/test_dtw_features.py`
- `tests/test_no_target_leakage.py`

### Stage A3b: Target Engineering

Options to evaluate separately:

- Signed-log residual: `sign(x) * log1p(abs(x))` with inverse transform before TVT-scale RMSE.
- Parallel derivative model: predict `d(TVT)/d(MD)` and combine with residual prediction.

Primary files:

- New `src/rogii/target.py`
- `src/rogii/train.py`
- `src/rogii/predict.py`
- `tests/test_target_transforms.py`

Verification:

- Always report RMSE in reconstructed TVT scale, not only transformed target space.
- Compare OOF prediction variance against OOF target variance.
- Inspect per-well prediction dispersion to detect flattening.

Rollback rule:

- If predicted TVT curves lose dispersion or TVT-scale RMSE degrades, revert to the plain residual target.

Promotion gate:

- Promote DTW and target changes only if they improve TVT-scale CV and do not create flattening, leakage or unacceptable runtime.

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
| Public saved artifacts / TabICL artifact stack | Rejected for clean mainline | Opaque external artifacts and heavy dependency path |
| Exact train/test coordinate overlap blend | Rejected for clean mainline | High leakage risk; may be studied only as a separate diagnostic with explicit approval |
| Particle filters | Deferred | Complex and stochastic; revisit only after A1-A4 evidence |
| 1D CNN | Optional | Requires separate runtime/dependency approval after tabular ensemble |
