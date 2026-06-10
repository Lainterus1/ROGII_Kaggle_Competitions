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
- **B4** = Optuna-tuned 3-seed LightGBM ensemble [42,7,123] (18 R1 tabular features, residual delta target) + Savgol w=31 p=2 — **canonical active baseline** (CV 13.948, LB TBD). Tuned via Optuna TPESampler 30 trials, 2-fold screening → top-3 verified on 5-fold. Params: lr=0.0664, num_leaves=48, min_child_samples=60, subsample=0.716, colsample_bytree=0.733. CV std improved 0.868→0.764 vs R3. OOF + Savgol: 13.965.
- **R3** = 3-seed ensemble [42,7,123] (18 tabular features, residual target) + Savgol w=31 p=2 (CV 14.052, LB 12.177) — superseded by B4.
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

Kaggle run R2 / PrP3 (2026-06-06):

- Model: R1 LightGBM, 18 features, residual delta target
- Post-processing: per-well Savgol smoothing `window=31`, `polyorder=2`
- OOF RMSE: `14.2123` vs raw `14.2187` (`−0.0064`)
- Official LB RMSE: **`12.239`**
- Status: promoted as R2, later superseded by R3

Kaggle run R3 / A4 multi-seed (2026-06-07):

- Config: `configs/a4_multiseed.yaml`
- Model: 3-seed LightGBM ensemble `[42, 7, 123]`, 18 R1 features, residual delta target
- Post-processing: Savgol `window=31`, `polyorder=2`
- CV RMSE: `14.052 ± 0.868`
- Official LB RMSE: **`12.177`**
- Status: superseded by B4

Kaggle run B4 / Optuna-tuned (2026-06-10):

- Config: `configs/b4_tuned.yaml`, best params: `configs/b4_best_params.yaml`
- Model: Optuna-tuned 3-seed LightGBM ensemble `[42, 7, 123]`, 18 R1 features, residual delta target
- Tuning: Optuna TPESampler, 30 trials, 2-fold screening, top-3 verified on 5-fold
- Best params: `lr=0.0664`, `num_leaves=48`, `min_child_samples=60`, `subsample=0.716`, `colsample_bytree=0.733`, `min_child_weight=0.0096`, `reg_alpha=0.0015`, `reg_lambda=0.0004`
- Post-processing: Savgol `window=31`, `polyorder=2`
- CV RMSE: `13.948 ± 0.764` (`−0.104` vs R3 14.052, std improved 0.868→0.764)
- Official LB RMSE: TBD
- Status: canonical active baseline
- MLflow: `rogii-wellbore-tuning` / `b4_optuna_tuned`
- Scripts: `scripts/run_tune.py` (new), `src/rogii/tuning.py` (new), `tests/test_tuning.py` (new)
