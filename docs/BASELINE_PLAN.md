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

Baseline status:

- Stage 4 is frozen as the reference baseline.
- **R3** = 3-seed ensemble [42,7,123] (18 tabular features, residual target) + Savgol w=31 p=2 — **canonical active baseline** (CV 14.052, LB 12.177).
- R2 = R1 model + Savgol w=31 p=2 (OOF 14.21, LB 12.239).
- R1 = raw 18-feature LightGBM model without post-processing (OOF 14.22, LB 12.247).
- Future improvement planning has moved to `docs/ROADMAP.md`.
- Do not rewrite the recorded Stage 4 result when experimenting with new feature sets or models; log new runs separately in `docs/EXPERIMENT_LOG.md`.

Initial baseline sequence:

1. Valid submission baseline: last-known-`TVT_input` prediction to test submission contract. Status: done in Step 07.
2. Naive local baseline: last-known-`TVT_input` rule evaluated on train post-PS rows. Status: done in Step 07.
3. First ML baseline: LightGBM using safe numeric features (no TVT_input). Status: done. RMSE 120.06.
4. Stronger baseline: LightGBM + last_tvt_input well-level constant. Status: done. CV 20.58, LB 24.11.
5. Next development stages: see `docs/ROADMAP.md`.

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

Stage 4 ML baseline result (with last_tvt_input well-level constant):

- Script: `python scripts/run_train.py --data-dir data --n-splits 5 --seed 42 --include-tvt-input`
- Model: LightGBM, 10 numeric features (+ last_tvt_input: last known pre-PS TVT_input value per well)
- CV RMSE (mean ± std): `20.84 ± 3.24`
- Train rows (post-PS): `3,783,989`, wells: `773`
- Generated submission rows: `14,151`, passed validator
- Adding TVT_input as a well-level constant improved RMSE 5.8x (120.06 → 20.84), only 5 points above naive baseline

Kaggle run (2026-06-05):

- Notebook: `notebooks/00_kaggle_thin_runner.ipynb` (offline, uses `rogii-repo` Kaggle Dataset)
- Workflow: copy dataset → cd → run train → run predict
- Data: `/kaggle/input/competitions/rogii-wellbore-geology-prediction`
- Output: `/kaggle/working/submission.csv`
- Kaggle CV RMSE: `20.58 ± 3.99` (5 folds, 3.78M rows, 773 wells)
- Official LB RMSE: `24.114`
- LB/CV gap: `+3.5` (reasonable for Kaggle)

Kaggle run R1 (2026-06-05):

- Notebook: `notebooks/00_kaggle_thin_runner.ipynb` (offline, uses `rogii-repo` Kaggle Dataset)
- Model: LightGBM, 18 features (6 base + 9 geometry + 3 GR), residual delta target
- Kaggle CV RMSE: `~14.19` (5 folds, 3.78M rows, 773 wells) — estimated from local run
- Official LB RMSE: **`12.247`**
- LB/CV gap: `−2.0` **(LB BETTER than CV — test wells easier than train)**
- 49% improvement over Stage 4 LB (24.114 → 12.247)
