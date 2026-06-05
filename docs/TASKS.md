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
| Pending | Medium | Roadmap R3: implement multi-seed LightGBM and simple ensemble |
| Pending | Medium | Decide whether to add CatBoost as a new dependency for Roadmap R3 |

## Open questions

- What missing-value patterns matter for first ML features?
- How should typewell data be aligned safely?
- Should CatBoost be added to `requirements.txt` for the next model upgrade stage?
