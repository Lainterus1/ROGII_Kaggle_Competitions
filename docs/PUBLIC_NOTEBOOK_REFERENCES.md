# Public Notebook References

## Purpose

Track public Kaggle notebooks reviewed and any ideas reused from them.

## Owns

Notebook links, review notes, adopted ideas, rejected ideas and attribution for public-notebook-derived concepts.

## Update when

- A public notebook is reviewed.
- A public notebook idea is used, modified or rejected.
- A comparison to public baselines is made.

## Do not store here

- Copied notebook code.
- Private notes or credentials.
- Experiment metrics not tied to notebook review.

## Current content

Reviewed public notebooks:

| Notebook | Reviewed idea | Decision |
|---|---|---|
| https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based/notebook | Residual target, rolling/lag GR features, geometry features, typewell/DTW/beam/PF feature families, LGBM/CatBoost blending | Use as reference. Adopt residual, GR rolling, geometry and simple typewell ideas first. Defer DTW, beam, spatial KNN and particle filters until simpler roadmap stages stabilize. Do not copy notebook code blindly. |
| https://www.kaggle.com/code/romantamrazov/rogii-super-solution-lb-top-3/notebook | Beam search (±2 delta, 7 configs), particle filter (N=500), multi-scale NCC, formation plane KNN, LGBM+CatBoost hill-climbing ensemble | **Adopt beam search for B1.** Reference for algorithm implementation (cells 3-7). Defer PF, NCC, CatBoost, and hill climbing for later stages. |
| https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-hill-climbing/notebook | Same beam+Pf+NCC pipeline with artifact caching, multiple LGBM/CB configs | **Adopt as secondary reference.** Validates beam configs and feature integration patterns. |
| https://www.kaggle.com/code/pilkwang/rogii-eda-target-free-alignment-for-tvt/notebook | Target-free stratigraphic alignment, formation-aware TVT, DTW-like matching | **Reviewed.** Confirms alignment approach. No direct code adoption. |
| https://www.kaggle.com/code/ruhul20/temporal-cnn-tcn-baseline-rogii-wellbore | Temporal CNN / TCN baseline for ROGII | **Use as orientation only.** Supports A5a TCN direction; do not copy code blindly. Current implementation lives in `src/rogii/sequence_features.py`, `src/rogii/sequence_data.py`, `src/rogii/tcn_model.py`, `src/rogii/train.py`. |
| Public notebook idea referenced in internal notes: plagiagia v2.8 | Model + beam + particle-filter blend, post-processing and clipping ideas | **Referenced, exact URL pending re-check.** Ideas are documented in ADR-019/ADR-023; PoP2 blend with weak Z/DTW predictors was rejected, stronger Beam/PF predictor mode remains planned as A5b. |
| Public notebook idea referenced in internal notes: Scott Weeden v13 | Z-physics prior, DTW GR matching, Savgol params | **Referenced, exact URL pending re-check.** Z-Drift features and PoP2 blend were implemented and rejected; Savgol parameter ideas were evaluated in PrP3. |
| Public notebook idea referenced in internal notes: Matteo Niccoli | StratifiedGroupKFold-style fold balancing and TVT/Z analysis | **Referenced, exact URL pending re-check.** StratifiedGroupKFold is implemented as experimental `--cv-strategy stratified`, not default. |
| Public notebook idea referenced in internal notes: stpeteishii TCN | TCN with causal convolutions for sequence modeling of well trajectories | **Referenced, exact URL pending re-check.** Supports A5a TCN direction; referenced in ROADMAP.md leader analysis (TCN ~LB 7.0). |
| Public notebook idea referenced in internal notes: adarsh5harma Stacker v2 | Multi-strategy stacking/ensemble with diverse model types | **Referenced, exact URL pending re-check.** Supports A5c ensemble direction; referenced in ROADMAP.md leader analysis. |

Rules:

- Public notebooks may be used for orientation, feature ideas, validation hints and comparison.
- Do not blindly copy code.
- Document any adopted idea and why it is safe.
- Do not overfit to public leaderboard by copying unexplained notebooks.

## Open questions

- Exact train/test coordinate overlap tricks remain rejected for the clean mainline unless competition rules and leakage risk are explicitly reviewed.
- Exact URLs for plagiagia v2.8, Scott Weeden v13, Matteo Niccoli, stpeteishii TCN and adarsh5harma Stacker v2 references should be rechecked before adding hyperlinks.
