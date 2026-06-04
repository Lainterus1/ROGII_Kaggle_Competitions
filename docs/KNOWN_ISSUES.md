# Known Issues

## Purpose

Track known risks, blockers, unresolved questions and validation doubts.

## Owns

Current issues that could affect correctness, reproducibility, data leakage, Kaggle execution or submission validity.

## Update when

- A blocker or risk is discovered.
- A validation concern appears.
- A known issue is resolved.
- Data inspection reveals schema or leakage risks.

## Do not store here

- General backlog tasks.
- Full experiment logs.
- Public notebook references.

## Current content

| Status | Issue | Impact | Next action |
|---|---|---|---|
| Open | Data inventory is compact, not exhaustive | Missing-value profiles and typewell alignment details are still incomplete | Extend inventory before feature engineering |
| Open | Validation group column unknown | Leakage-safe CV cannot be finalized | Find well/group ID candidates during data inventory |
| Open | Kaggle Evaluation page requires better access than anonymous HTML fetch | Metric wording is confirmed from task deck, but page text still has not been cross-checked | Re-check official Evaluation page when accessible |
| Open | `TVT_input` exists in train and test horizontal well files | It is allowed until PS per task deck; post-PS values must not be used | Keep leakage tests and docs updated when using it |

## Open questions

- What missing-value patterns matter for first ML features?
- How should typewell data be aligned safely?
