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

Required checks:

- No group appears in both train and validation folds.
- Fold target distributions are reasonable.
- Fold group counts and row counts are reasonable.
- Metric matches Kaggle or the mismatch is documented.

## Open questions

- Which column is the correct validation group?
- Should file prefix be materialized as an explicit `well_id` column during loading?
- How many folds are appropriate for available wells/groups?
