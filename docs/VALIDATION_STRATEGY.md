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

Default recommendation before data inspection:

- Prefer group-aware validation if a well or wellbore identifier exists.
- Use `GroupKFold` by well/group ID unless data inspection proves another strategy is safer.
- Do not use row-level random KFold as the primary CV if rows from the same well can leak information.

Preliminary group candidate after light data inspection:

- Use the file prefix before `__horizontal_well.csv` or `__typewell.csv` as the initial well/group ID candidate.
- Confirm this during full data inventory before finalizing folds.

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

- Active comparison stage: **R2** (R1 model + Savgol w=31 p=2 post-processing).
- Strategy: 5-fold `GroupKFold` by `well_id` on post-PS train rows.
- Target: residual delta `TVT - last_tvt_input`, reconstructed to TVT for submission.
- Features: 18 features (6 base + 9 geometry + 3 GR), documented in `docs/HOW_IT_WORKS.md`.
- R1 model OOF CV: `~14.22`, R2 pipeline OOF CV: `~14.21`.
- Public LB: R1 `12.247`, R2 `12.239`.

Planned Stage A2 spatial KNN validation contract:

- Spatial KNN features must be built fold-aware, not once on the full dataset before CV.
- For validation fold K, the spatial reference set must contain only wells outside fold K.
- Reference rows must be pre-PS rows only.
- Reference values must come from known pre-PS `TVT_input`, not post-PS `TVT`.
- Validation wells must never appear in their own KNN tree.
- Test-time spatial KNN uses train pre-PS reference rows only by default; test pre-PS rows are excluded for a stricter train/test contract.
- If spatial KNN produces implausibly low CV such as RMSE `2-3`, treat it as leakage until proven otherwise and do not submit.

Required checks:

- No group appears in both train and validation folds.
- Fold target distributions are reasonable.
- Fold group counts and row counts are reasonable.
- Metric matches Kaggle or the mismatch is documented.

## Open questions

- None for the current roadmap. Default remains 5-fold `GroupKFold` by `well_id`.
- `StratifiedGroupKFold` (4,5) available as experimental option: `--cv-strategy stratified`. Produces CV mean 13.806 ± 1.522 (vs GroupKFold 14.191 ± 0.887). Higher std may indicate better CV/LB correlation. Not yet promoted to default.
