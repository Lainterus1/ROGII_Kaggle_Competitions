# Experiment Log

## Purpose

Provide a human-readable experiment history complementary to MLflow.

## Owns

Run summaries, local CV scores, public leaderboard scores entered manually, feature set names, model types, configs, notes and follow-up ideas.

## Update when

- A baseline or experiment is run.
- A Kaggle public leaderboard score is recorded.
- An experiment changes the next-step plan.

## Do not store here

- Raw MLflow artifact directories.
- Large prediction files.
- Secrets or Kaggle credentials.

## Current content

Template for future entries:

| Date | Run name | Model | Features | CV metric | Public LB | Config | Notes |
|---|---|---|---|---|---|---|---|
| 2026-06-04 | naive_last_known_tvt_input | Rule baseline | Last non-null pre-PS `TVT_input` per well | RMSE `15.909853` on train post-PS rows | Not submitted | `configs/baseline_naive.yaml` | Generated and validated `outputs/submission.csv`; no MLflow run because this is a non-model smoke baseline |

## Open questions

- None.
