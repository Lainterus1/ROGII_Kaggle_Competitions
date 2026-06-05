# Tasks

## Purpose

Track the current project backlog and next actions.

## Owns

Actionable tasks, status, near-term priorities and explicit blockers.

## Update when

- A step is completed.
- New required work is discovered.
- A task becomes blocked, cancelled or done.

## Do not store here

- Long technical explanations.
- Experiment result details.
- Architecture rationale that belongs in `DECISIONS.md`.

## Current content

| Status | Priority | Task |
|---|---|---|
| Done | High | Step 01: create `docs/PROJECT_CONTEXT.md` |
| Done | High | Step 02: create source-of-truth documentation skeletons |
| Done | High | Step 03: select initial architecture and record ADR |
| Done | High | Step 04: create project skeleton |
| Done | High | Initialize local git and push bootstrap to `ROGII_Kaggle_Competitions` |
| Done | High | Step 05: create final `AGENTS.md` |
| Done | Medium | Step 06: create reusable project-specific agent skills |
| Done | High | Step 07: implement data inventory, RMSE metric, submission validator and naive baseline |
| Done | Medium | Step 08: create task contract template |
| Done | Medium | Step 09: create documentation maintenance policy |
| Done | Medium | Step 10: create review and optimization protocol |
| Done | Medium | Step 11: create handoff and context compaction |
| Done | High | Freeze Stage 4 LightGBM + `last_tvt_input` as the reference baseline |
| Done | High | Roadmap R1: implement residual target plus deterministic GR and geometry features |
| Done | High | Roadmap R2: implement simple typewell-reference features after leakage review (degraded CV, not promoted) |
| Done | High | R1 feature ablation: remove 20 zero/low-importance features, finalize 18-feature set with identical CV |
| Done | High | Submit R1 optimized model to Kaggle (LB: 12.247, 49% improvement over Stage 4) |
| Done | High | Replace old pending roadmap with the staged A0-A4 development plan |
| Cancelled | Medium | Old Roadmap R3 standalone model-upgrade task; superseded by Stage A4 structural blending |
| Cancelled | Medium | Old CatBoost decision task; dependency is now approved and belongs to Stage A4 |
| Done | High | Stage A0: sync CLI/config/README/model-payload contracts before new feature implementation |
| Rejected | High | Stage A1: trajectory features (CV flat, LB worse: 12.487 vs R1 12.247, all features are geometry duplicates) |
| Done | High | Stage A2a: add causal GR DWT features — CV 14.13 (+0.06 vs R1), promoted |
| Done | High | Stage A2b: implement strict OOF spatial KNN — CV 14.21 (−0.02 flat), not promoted, code kept |
| Done | Medium | Stage A2 combined (DWT + spatial): CV 14.19 (flat), spatial neutralizes DWT gain |
| Rejected | High | Stage A3a: DTW typewell alignment — CV 14.63 (+0.50 vs A2a), insufficient GR correlation |
| Rejected | High | Stage A3b.1: signed-log residual target — CV 14.64 (+0.51), residuals not heavy-tailed |
| Rejected | High | Stage A3b.2: derivative dTVT/dMD target — CV 14.32 (+0.19), integration error accumulation |
| Rejected | High | Geology v1 (well-level): CV 14.57 (+0.44 vs A2a), signal unstable across folds |
| Not promoted | High | Geology v2 (per-row GR z-scores): CV 14.17 (−0.04 flat vs A2a), code kept |
| Deferred | Medium | 1D CNN sequence model — architecture diversity, revisit after tabular ceiling |
| Deferred | Medium | Multi-seed LGBM + CatBoost + Ridge stacking — cosmetic ensemble on same features |
| Deferred | Medium | Stage A4: standardize OOF artifacts, add multi-seed LGBM, CatBoost and stacking |
| Done | High | Split Kaggle runner into separate training and inference notebooks (ADR-007) |
| In Progress | High | Submit A2a to Kaggle: prediction validated locally, docs updated, awaiting Kaggle run |

## Open questions

- None for the current roadmap reset. Stage-specific blockers are tracked in `docs/ROADMAP.md` and `docs/KNOWN_ISSUES.md`.
