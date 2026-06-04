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

1. Valid submission baseline: constant or mean/median prediction to test submission contract.
2. Naive local baseline: global mean/median or simple safe rule after schema inspection.
3. First ML baseline: LightGBM using safe numeric features available in train and test.
4. Stronger baseline later: depth features, rolling/lag features, spatial/typewell features, CatBoost/XGBoost comparison and ensemble.

Acceptance criteria for first useful baseline:

- Generates valid `submission.csv`.
- Produces local validation score.
- Logs run in MLflow.
- Saves config snapshot and selected artifacts.
- Documents assumptions and leakage checks.

## Open questions

- Which features are safe after data inspection?
- Is LightGBM available in the target Kaggle environment or must it be installed?
