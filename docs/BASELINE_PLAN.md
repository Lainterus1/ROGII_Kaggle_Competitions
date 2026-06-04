# Baseline Plan

## Purpose

Define the baseline stages, scope and acceptance criteria.

## Owns

Baseline sequence, model scope, feature safety rules, acceptance criteria and future improvement path.

## Update when

- A baseline stage is completed.
- Feature strategy changes.
- Model family changes.
- Acceptance criteria change.

## Do not store here

- Detailed experiment metrics.
- Full data inventory.
- Public notebook review details.

## Current content

Initial baseline sequence:

1. Valid submission baseline: last-known-`TVT_input` prediction to test submission contract. Status: done in Step 07.
2. Naive local baseline: last-known-`TVT_input` rule evaluated on train post-PS rows. Status: done in Step 07.
3. First ML baseline: LightGBM using safe numeric features available in train and test. Status: done.
4. Stronger baseline later: depth features, rolling/lag features, spatial/typewell features, CatBoost/XGBoost comparison and ensemble.

Acceptance criteria for first useful baseline:

- Generates valid `submission.csv`.
- Produces local validation score.
- Logs model runs in MLflow once model training exists.
- Saves config snapshot and selected artifacts for model runs.
- Documents assumptions and leakage checks.

Step 07 naive baseline result:

- Script: `python scripts/run_naive_baseline.py --data-dir data --output outputs/submission.csv`
- Local naive RMSE: `15.909853`
- Validation rows: `3,783,989`
- Validation wells: `773`
- Generated submission rows: `14,151`
- Submission validator: passed

Stage 3 ML baseline result (without TVT_input):

- Script: `python scripts/run_train.py --data-dir data --n-splits 5 --seed 42`
- Model: LightGBM, 9 numeric features (MD, X, Y, Z, GR, GR_is_missing, MD_delta, MD_relative, row_position)
- Validation: GroupKFold 5-fold by well_id, post-PS rows only
- CV RMSE (mean ± std): `120.06 ± 11.31`
- Train rows (post-PS): `3,783,989`, wells: `773`
- Generated submission rows: `14,151`, passed validator
- Pure geometric baseline deliberately excludes TVT_input; RMSE gap vs naive (15.91 → 120.06) shows TVT_input dominance

## Open questions

- Is LightGBM available in the target Kaggle environment or must it be installed?
- Should TVT_input be re-introduced as a well-level feature for Stage 4?
