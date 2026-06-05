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
|---|---|---|---|---|
| Open | Data inventory is compact, not exhaustive | Missing-value profiles and typewell alignment details are still incomplete | Extend inventory before feature engineering |
| Open | Kaggle Evaluation page requires better access than anonymous HTML fetch | Metric wording is confirmed from task deck, but page text still has not been cross-checked | Re-check official Evaluation page when accessible |
| Open | `TVT_input` exists in train and test horizontal well files | It is allowed until PS per task deck; post-PS values must not be used | Keep leakage tests and docs updated when using it |
| Resolved | Stage A2 DWT runtime concern | Runtime ~1.4 min full train, well within limits | Promoted to A2a |
| Resolved | Stage A2 spatial KNN leakage risk | No leakage detected (CV 14.21, not implausibly low) | Code kept; not promoted due to flat CV |
| Resolved | Stage A3 DTW runtime concern | DTW rejected (CV 14.63), runtime was not the bottleneck | N/A |
| Resolved | Stage A3 target flattening risk | Both signed-log and derivative rejected (CV worse) | N/A |
| Open | Tabular feature ceiling at CV ~14.13 | All 8 feature experiments (A1-A4) flat or degraded; X,Y,Z,GR signal saturated | Shift focus to architecture (CNN) and ensemble |
| Open | Kaggle notebook must be manually updated after code pushes | Kaggle may run stale commands or stale repository dataset contents | After each push intended for Kaggle, provide exact notebook edit instructions to the user |

## Open questions

- None for the roadmap reset. Stage-specific risks are tracked above until resolved.
