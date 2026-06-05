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

The Stage 4 baseline is frozen as the reference baseline:

- Model: LightGBM with `safe_numeric_v1 + last_tvt_input`.
- Validation: 5-fold `GroupKFold` by well.
- Recorded result: Kaggle CV RMSE `20.58 +/- 3.99`, public LB RMSE `24.114`.
- Source of truth for frozen baseline details: `docs/BASELINE_PLAN.md` and `docs/EXPERIMENT_LOG.md`.

Future work should branch from this baseline without rewriting its recorded result.

## Guiding constraints

- Keep core logic in `src/rogii/` and executable entry points in `scripts/`.
- Keep Kaggle notebooks thin runners.
- Keep group-aware validation by well unless a documented decision changes it.
- Use public notebooks only for understood ideas, not copied code or opaque artifacts.
- Do not use public saved model artifacts, TabICL artifacts or exact train/test coordinate overlap blending in the clean mainline roadmap.
- Treat `TVT_input` as allowed only up to Prediction Start; never use post-PS labels or target-derived columns as features.
- Promote a stage only after tests pass, submission validation passes and CV improves or the stage produces useful diagnostics.

## Stage R1: Residual GR and Geometry Features ✅ Done

Goal: make the model predict `TVT - last_tvt_input` and add deterministic GR/trajectory features inspired by reviewed public notebooks.

Result: LightGBM, 18 features (6 base + 9 geometry + 3 GR), residual delta target, CV RMSE `14.19 ± 0.89` (GroupKFold 5-fold, 3.78M post-PS rows, 773 wells). ~31% improvement over frozen Stage 4 baseline (20.58 → 14.19). After feature ablation: 20 zero/low-importance features removed with zero CV regression. Full feature rationale in `docs/HOW_IT_WORKS.md`.

Scope:

- Add residual target handling: train on `target_delta = TVT - last_tvt_input`.
- Add prediction reconstruction: `pred_tvt = last_tvt_input + pred_delta`.
- Add geometry features: `md_since_ps`, `frac_after_ps`, `dx_since_ps`, `dy_since_ps`, `dz_since_ps`, `dxy_since_ps`, `dxdmd`, `dydmd`, `dzdmd`.
- Add GR features: rolling means/stds for windows `5, 21, 51, 101`, lags/leads `1, 5, 15, 30`, `gr_d1`, `gr_d2`, `gr_energy`, `gr_envelope`.

Primary files:

- `src/rogii/features.py`
- `src/rogii/train.py`
- `src/rogii/predict.py`
- `scripts/run_train.py`
- `scripts/run_predict.py`
- `configs/baseline_lgbm.yaml`
- `tests/test_feature_engineering.py`
- `tests/test_no_target_leakage.py`
- `tests/test_smoke_pipeline.py`

Verification:

- `python -m pytest tests`
- `python scripts/run_train.py --data-dir data --n-splits 5 --seed 42 --include-tvt-input`
- `python scripts/run_predict.py --data-dir data --model models/baseline_lgbm.pkl --output outputs/submission.csv --include-tvt-input`
- `python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv`

Promotion gate:

- Promote as the default next baseline only if validation stays leakage-safe and CV improves or provides clearly useful diagnostics.

## Stage R2: Typewell Features V1 ❌ Degraded

Goal: add simple, explainable typewell-reference features before heavier alignment methods.

Result: LightGBM, 53 features (R1 38 + 15 typewell), residual delta target, CV RMSE `14.75 ± 0.77`. **+0.66 degradation vs R1 (14.09)**. Typewell features rejected for current baseline due to CV regression. Likely cause: `tw_gr_residual_*` features ≈ `GR - const`, adding redundant correlated columns without new signal. May revisit with feature selection or different typewell alignment strategy in future roadmap stage.

Decision: Do not promote. Keep R1 as the best current baseline. Submit R1 to Kaggle.

Scope:

- Add `read_typewell_well(data_dir, split, well_id)` to data loading.
- Pass each well's typewell frame into feature construction.
- Interpolate typewell `GR` by `TVT`.
- Add anchor-offset residual features comparing horizontal `GR` to typewell `GR` at `last_tvt_input + offset`.
- Start with offsets `[-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]`.
- Add simple typewell summary features such as `tw_range`, `tw_gr_mean`, `tw_gr_std`, `tw_gr_at_last_tvt`.

Primary files:

- `src/rogii/data_loading.py`
- `src/rogii/features.py`
- `src/rogii/train.py`
- `src/rogii/predict.py`
- `tests/test_feature_engineering.py`
- `tests/test_no_target_leakage.py`

Verification:

- `python -m pytest tests`
- Repeat Stage R1 train, predict and submission validation commands.

Promotion gate:

- Promote only after leakage review confirms typewell features use information available at prediction time and CV improves or feature importance supports keeping the block.

## Stage R3: Model Upgrade and Simple Ensemble

Goal: improve model strength without opaque public artifacts.

Scope:

- Add multi-seed LightGBM runs, starting with seeds `42`, `7`, `123`.
- Add a simple saved ensemble object that can be loaded by `run_predict.py`.
- Evaluate simple OOF average before more flexible stacking.
- Add CatBoost only after explicit user approval because it adds a new dependency.
- If CatBoost is approved, train CatBoost under the same folds and feature set.
- Consider non-negative ridge stacking only after OOF predictions are saved and validated.

Primary files:

- `src/rogii/models.py`
- `src/rogii/train.py`
- `src/rogii/predict.py`
- `scripts/run_train.py`
- `scripts/run_predict.py`
- `configs/baseline_lgbm.yaml`
- `requirements.txt` only if CatBoost is approved.

Verification:

- `python -m pytest tests`
- Train/predict/validate commands from Stage R1.
- Compare single-model OOF RMSE, average-ensemble OOF RMSE and any stack OOF RMSE.

Promotion gate:

- Promote ensemble only if OOF improves over the best single model without unacceptable runtime or dependency cost.

## Deferred Ideas

| Idea | Status | Reason |
|---|---|---|
| Beam search GR-to-typewell alignment | Deferred | Promising but heavier than R1/R2; implement after simple features stabilize |
| Multi-scale self-correlation | Deferred | Deterministic and useful, but belongs after typewell v1 |
| Formation-plane KNN and dense ANCC | Deferred | Must be fold-aware to avoid validation leakage |
| Particle filters | Deferred | Complex and stochastic; needs reproducibility tests |
| TabICL/public artifact stack | Rejected for clean mainline | Opaque external artifacts and heavy dependency path |
| Exact train/test coordinate overlap blend | Rejected for clean mainline | High leakage risk; may be studied only as a separate diagnostic with explicit approval |
