# Validation Strategy

## Purpose

Document the local validation design and leakage controls.

## Owns

Fold strategy, group split rules, leakage checks, target distribution checks, local/public leaderboard comparison notes and validation limitations.

## Update when

- Data schema or group ID candidates are confirmed.
- Fold strategy changes.
- Leakage risk changes.
- Local CV and public LB relationship is analyzed.

## Do not store here

- Full data inventory.
- Raw fold predictions.
- Long experiment logs.

## Current content

Default validation recommendation:

- Use group-aware validation by `well_id`, where `well_id` is the file prefix before `__horizontal_well.csv` or `__typewell.csv`.
- Use 5-fold `GroupKFold` by `well_id` as the primary CV strategy unless a documented decision changes the default.
- Do not use row-level random KFold as the primary CV because rows from the same well leak trajectory/geology context.

Step 07 naive validation:

- Evaluate the last-known-`TVT_input` rule on train post-PS rows.
- Prediction for each train well is the final non-null `TVT_input` before PS.
- Target rows are rows after PS where `TVT_input` is missing and `TVT` is known.
- This is a sanity baseline, not the final model CV strategy.
- Result from local run: RMSE `15.909853` over `3,783,989` post-PS rows across `773` train wells.

Stage 3 ML validation:

- Strategy: GroupKFold (n_splits=5) by well_id on post-PS train rows.
- Features: MD, X, Y, Z, GR, GR_is_missing, MD_delta, MD_relative, row_position.
- TVT_input deliberately excluded to measure pure geometric baseline strength.
- Target: TVT column.
- Result: CV RMSE `120.06 ± 11.31` (5 folds) over `3,783,989` post-PS rows across `773` wells.

Current active baseline validation:

- Active comparison stage: **R3** (3-seed LightGBM `[42, 7, 123]` on the R1 18-feature set + Savgol w=31 p=2 post-processing).
- Strategy: 5-fold `GroupKFold` by `well_id` on post-PS train rows.
- Target: residual delta `TVT - last_tvt_input`, reconstructed to TVT for submission.
- Features: 18 features (6 base + 9 geometry + 3 GR), documented in `docs/HOW_IT_WORKS.md`.
- R1 model OOF CV: `~14.22`, R2 pipeline OOF CV: `~14.21`, R3 GroupKFold CV: `14.052 ± 0.868`.
- Public LB: R1 `12.247`, R2 `12.239`, R3 `12.177`.

Implemented Stage A2 spatial KNN validation contract:

- Spatial KNN features must be built fold-aware, not once on the full dataset before CV.
- For validation fold K, the spatial reference set must contain only wells outside fold K.
- Reference rows must be pre-PS rows only.
- Reference values must come from known pre-PS `TVT_input`, not post-PS `TVT`.
- Validation wells must never appear in their own KNN tree.
- Test-time spatial KNN uses train pre-PS reference rows only by default; test pre-PS rows are excluded for a stricter train/test contract.
- If spatial KNN produces implausibly low CV such as RMSE `2-3`, treat it as leakage until proven otherwise and do not submit.

TCN/A5 validation status:

- TCN uses the same primary 5-fold `GroupKFold` by `well_id`.
- Phase 0 control (`a5_tcn_control`) completed 4 folds and aborted fold 5; CV was `15.031 ± 1.35` with severe global prediction flattening (`std_ratio 0.42`).
- Phase 2 dual normalization is implemented and must be validated before further tuning: pass condition is `std_ratio > 0.7` and screening folds better than the Phase 0 control.
- TCN OOF outputs use `well_id`, `row_idx`, `fold`, `y_true`, `y_pred`, `baseline` and are stored outside Git under `outputs/oof/` when `--save-oof` is used.

Required checks:

- No group appears in both train and validation folds.
- Fold target distributions are reasonable.
- Fold group counts and row counts are reasonable.
- Metric matches Kaggle or the mismatch is documented.

## Open questions

- Default remains 5-fold `GroupKFold` by `well_id`.
- `StratifiedGroupKFold` (4,5) is available as an experimental option: `--cv-strategy stratified`. It produced CV mean `13.806 ± 1.522` for single-seed R1 and `13.763 ± 1.523` for multi-seed, but is not promoted to default until CV/LB correlation is validated.
