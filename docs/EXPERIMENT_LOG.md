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
| 2026-06-04 | baseline_lgbm_safe_numeric | LightGBM | MD, X, Y, Z, GR, GR_is_missing, MD_delta, MD_relative, row_position | RMSE `120.06 ± 11.31` (GroupKFold 5) | Not submitted | `configs/baseline_lgbm.yaml` | Pure geometric baseline; deliberately excludes TVT_input to measure geometry-only signal |
| 2026-06-04 | baseline_lgbm_tvt | LightGBM | safe_numeric_v1 + last_tvt_input (well-level constant) | RMSE `20.84 ± 3.24` (GroupKFold 5, local) | Not submitted | `configs/baseline_lgbm.yaml` | 5.8x improvement over geometry-only; local run on 3.78M post-PS rows, 773 wells |
| 2026-06-05 | baseline_lgbm_tvt_kaggle | LightGBM | safe_numeric_v1 + last_tvt_input | RMSE `20.58 ± 3.99` (GroupKFold 5, Kaggle) | **24.114** | `configs/baseline_lgbm.yaml` | First Kaggle submission. Offline notebook via `rogii-repo` dataset. LB/CV gap +3.5. |
| 2026-06-05 | r1_lgbm_residual_geo_gr | LightGBM | safe_numeric_v1 + 9 geometry + 20 GR (38 total), residual delta target | RMSE `14.09 ± 0.88` (GroupKFold 5, local, delta space) | Not submitted | `configs/baseline_lgbm.yaml` | Roadmap R1 first run. 32% improvement over Stage 4. Target: `TVT - last_tvt_input`. |
| 2026-06-05 | r1_lgbm_optimized | LightGBM | 6 base + 9 geometry + 3 GR (18 total), residual delta target | RMSE `14.19 ± 0.89` (GroupKFold 5, local) | **12.247** | `configs/baseline_lgbm.yaml` | R1 Kaggle submission. CV→LB: 14.19→12.247 (−2.0 gap, LB BETTER than CV). 49% improvement over Stage 4 LB (24.11→12.25). After feature ablation: 20 zero/low-importance features removed with zero CV regression. 18 features: MD, X, Y, Z, GR, MD_relative + 9 geometry + gr_roll_mean_101, gr_roll_std_101, gr_energy. |
| 2026-06-05 | r2_lgbm_typewell_v1 | LightGBM | R1 features + 15 typewell (53 total), residual delta target | RMSE `14.75 ± 0.77` (GroupKFold 5, local, delta space) | Not submitted | `configs/baseline_lgbm.yaml` | Roadmap R2. Typewell features degraded CV (+0.66 vs R1). Likely cause: `tw_gr_residual_*` ≈ `GR - const`, highly correlated with base `GR`, adding noise without new signal. Typewell summary features (`tw_range`, `tw_gr_mean`, `tw_gr_std`) are well-level constants already partially captured by residual approach. **Decision: do not promote R2; submit R1 to Kaggle.** |
| 2026-06-05 | a1_lgbm_trajectory | LightGBM | R1 features + 5 trajectory + 1 dogleg (24→23 total), residual delta target | RMSE `14.24 ± 0.82` (GroupKFold 5, local) | **12.487** | `configs/baseline_lgbm.yaml` + `--include-trajectory` | Stage A1. CV flat (+0.05 vs R1 14.19), LB worse (+0.24 vs R1 12.247). All 5 trajectory features are r >= 0.99 with existing geometry features (z_local_delta≈dz_since_ps, dip_angle≈dzdmd, sin_azimuth≈dydmd, cos_azimuth≈dxdmd). No new signal — only importance redistribution across duplicates. `dogleg_severity_10m` ablated mid-experiment (importance 0.00). **Decision: REJECTED. Revert to R1 as active baseline.** |
| 2026-06-05 | a2a_lgbm_dwt | LightGBM | R1 (18) + 2 DWT = 20 features, residual delta target | RMSE `14.13 ± 0.77` (GroupKFold 5, local) | Not submitted | `configs/a2_lgbm.yaml` (DWT only) | Stage A2a. Causal trailing-window GR DWT (db4, window=256). `gr_dwt_approx` + `gr_dwt_detail_energy`. Runtime ~1.4 min full train, no leakage risk (causal verified). Marginal improvement +0.06 vs R1 14.19. **Decision: PROMOTED.** |
| 2026-06-05 | a2b_lgbm_spatial | LightGBM | R1 (18) + 9 spatial KNN = 27 features, residual delta target | RMSE `14.21 ± 0.81` (GroupKFold 5, local) | Not submitted | `configs/a2_lgbm.yaml` (spatial only) | Stage A2b. Strict OOF spatial KNN (k=5,10,50) with ball_tree. Reference: pre-PS TVT_input from OOF wells only. No leakage (CV not implausibly low). Flat CV (-0.02 vs R1). Spatial features add no new signal beyond X/Y/Z. **Decision: NOT PROMOTED (code kept).** |
| 2026-06-05 | a2_lgbm_combined | LightGBM | R1 (18) + 2 DWT + 9 spatial = 29 features, residual delta target | RMSE `14.19 ± 0.74` (GroupKFold 5, local) | Not submitted | `configs/a2_lgbm.yaml` | Stage A2 combined. CV flat vs R1. Spatial neutralizes DWT's marginal gain. **Not promoted as combined; A2a (DWT-only) is the promoted config.** |
| 2026-06-05 | a3a_lgbm_dtw | LightGBM | A2a (20) + 2 DTW = 22 features, residual delta target | RMSE `14.63 ± 1.14` (GroupKFold 5, local) | Not submitted | DWT+DTW CLI flags | Stage A3a. DTW typewell alignment (Sakoe-Chiba window=50) on pre-PS GR. CV +0.50 vs A2a (14.13). Median GR cross-correlation 0.43 — DTW forces alignment without real signal. Fold 5 RMSE 16.5 vs fold 3 RMSE 13.1 (high variance). **Decision: REJECTED.** |
| 2026-06-05 | a3b1_signed_log | LightGBM | A2a (20), signed-log residual target | RMSE `14.64` (GroupKFold 3, local) | Not submitted | N/A | Stage A3b.1. Signed-log: sign(x)*log1p(abs(x)). Residuals bounded ±40, skew 0.74 — not heavy-tailed. +0.51 vs A2a. **Decision: REJECTED.** |
| 2026-06-05 | a3b2_derivative | LightGBM | A2a (20), derivative dTVT/dMD target | RMSE `14.32` (GroupKFold 3, local) | Not submitted | N/A | Stage A3b.2. Predict derivative + integrate to TVT. +0.19 vs A2a (14.13). Integration error accumulation. **Decision: REJECTED.** |
| 2026-06-05 | a4_geology | LightGBM | A2a (20) + 7 geology = 27 features, residual delta target | RMSE `14.57 ± 0.80` (GroupKFold 5, local) | Not submitted | DWT+Geology CLI flags | Geology v1: well-level constants (form at PS, next form, boundaries). Signal unstable: folds 1,3 improved (−0.5), folds 2,4,5 degraded (+1.0). **Decision: REJECTED.** |
| 2026-06-05 | a4_geology_v2 | LightGBM | A2a (20) + 9 per-row GR z-scores = 29 features, residual delta target | RMSE `14.17 ± 0.86` (GroupKFold 5, local) | Not submitted | DWT+Geology CLI flags | Geology v2: per-row GR-vs-formation z-scores (8 formations + rolling). More stable than v1 but flat CV (−0.04 vs A2a 14.13). Tabular feature ceiling confirmed at ~14.13. **Decision: NOT PROMOTED. Code kept.** |

## Open questions

- None.
