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

Rules:

- Public notebooks may be used for orientation, feature ideas, validation hints and comparison.
- Do not blindly copy code.
- Document any adopted idea and why it is safe.
- Do not overfit to public leaderboard by copying unexplained notebooks.

## Open questions

- Are exact train/test coordinate overlaps permitted by competition rules, or should they remain diagnostic-only?
