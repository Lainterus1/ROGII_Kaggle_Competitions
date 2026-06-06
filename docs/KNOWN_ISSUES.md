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
| Resolved | Stage A2 DWT runtime concern | Runtime ~1.4 min full train, well within limits | Promoted to A2a |
| Resolved | Stage A2 spatial KNN leakage risk | No leakage detected (CV 14.21, not implausibly low) | Code kept; not promoted due to flat CV |
| Resolved | Stage A3 DTW runtime concern | DTW rejected (CV 14.63), runtime was not the bottleneck | N/A |
| Resolved | Stage A3 target flattening risk | Both signed-log and derivative rejected (CV worse) | N/A |
| Open | Tabular feature ceiling at CV ~14.13 | All 11+ feature experiments (A1-B1) flat or degraded on CV; A2a LB 12.558 worse than R1 12.247 — CV→LB gap confirmed | Shift focus to architecture (CNN), post-processing (PrP3) and ensemble |
| Resolved | Savgol smoothing never tested on real predictions (ADR-015) | Code existed since B2b but `--savgol-smooth` never ran on real data | PrP3: Savgol w=31 p=2 promotes OOF CV (14.2123 vs raw 14.2187). TVT clipping rejected. 3/3 wells improved per-well. Defaults updated. |
| Open | A2a DWT CV→LB inversion | CV +0.06 (14.13) but LB +0.311 (12.558) vs R1. DWT features do not generalize to test set despite causal windows and GroupKFold | R1 (12.247) remains active baseline; A2a artifacts preserved as candidate example |
| Resolved | R1 offline Kaggle inference notebook used fragile dataset paths | `00-rogii-inference-r1` version 2 failed on `rogii-repo-v2` flat layout and produced no LB score | Version 3 uses marker-based path discovery and validated kernel output |
| Resolved | A2a DWT model requires `pywavelets` during offline inference | `rogii-wheels-a2a-dwt` dataset created with pywavelets 1.8.0 wheel; Kaggle base env also has pywavelets 1.9.0 pre-installed | Kernel `00-rogii-inference-a2a-dwt` v1 validated — successful offline inference with DWT features |

## Open questions

- None for the roadmap reset. Stage-specific risks are tracked above until resolved.
