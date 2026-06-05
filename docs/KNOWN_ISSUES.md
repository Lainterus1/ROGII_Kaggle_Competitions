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
| Open | Kaggle Evaluation page requires better access than anonymous HTML fetch | Metric wording is confirmed from task deck, but page text still has not been cross-checked | Re-check official Evaluation page when accessible |
| Open | `TVT_input` exists in train and test horizontal well files | It is allowed until PS per task deck; post-PS values must not be used | Keep leakage tests and docs updated when using it |
| Open | Stage A2 spatial KNN has high leakage risk | A tree built on validation wells or post-PS `TVT` can create unrealistically low CV | Implement strict OOF tests before trusting CV; rollback if RMSE falls to implausible `2-3` range |
| Open | Stage A2 DWT and Stage A3 DTW may be expensive on full data | Full CV or Kaggle runs can exceed runtime limits | Run subset runtime spikes before full CV and keep blocks behind feature flags |
| Open | Stage A3 target engineering can flatten TVT curves | Transformed-target improvements may not improve TVT-scale RMSE | Compare reconstructed TVT RMSE and OOF prediction variance before promotion |
| Open | Kaggle notebook must be manually updated after code pushes | Kaggle may run stale commands or stale repository dataset contents | After each push intended for Kaggle, provide exact notebook edit instructions to the user |

## Open questions

- None for the roadmap reset. Stage-specific risks are tracked above until resolved.
