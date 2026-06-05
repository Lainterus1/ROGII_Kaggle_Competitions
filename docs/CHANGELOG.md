# Changelog

## Purpose

Record chronological project changes.

## Owns

Human-readable change history for documentation, structure, code, configs and workflow changes.

## Update when

- Files are created or materially changed.
- Project structure changes.
- Baselines or validation logic are added.
- Important docs are updated.

## Do not store here

- Detailed experiment metrics.
- Full ADR rationale.
- Raw command outputs.

## Current content

## 2026-06-04

- Created `docs/PROJECT_CONTEXT.md` from `ROGII_PROJECT_INTAKE_DOSSIER.md` and user clarifications.
- Created source-of-truth documentation skeletons for bootstrap Step 02.
- Accepted balanced Python package plus scripts architecture in `docs/ARCHITECTURE.md`.
- Added `ADR-001: Initial architecture` to `docs/DECISIONS.md`.
- Created initial project skeleton with configs, package placeholders, scripts, tests, notebooks and protective root files.
- Recorded public GitHub repository, Kaggle runtime setup and light local data observations.
- Replaced temporary `AGENTS.md` with final project-specific operating contract.
- Added `.gitattributes` to keep repository text files normalized to LF.
- Initialized local git repository and pushed bootstrap commits to public GitHub repo.
- Created project-specific agent skills for data inventory, submission validation, leakage review, Kaggle runner workflow and experiment logging.
- Implemented Step 07 vertical slice: data inventory CLI, RMSE metric, submission validator and last-known-`TVT_input` naive baseline.
- Ignored local `.serena/` workspace metadata.
- Created `docs/TASK_TEMPLATE.md` as the project-specific contract for future implementation tasks.
- Created documentation maintenance policy in `AGENTS.md` and `.agents/skills/documentation-maintenance/SKILL.md`.
- Created review and optimization protocol in `.agents/skills/code-review/SKILL.md` and `docs/REVIEW_CHECKLIST.md`.
- Created handoff and context compaction skill in `.agents/skills/handoff/SKILL.md`.
- Removed `steps/` folder and `ROGII_PROJECT_INTAKE_DOSSIER.md`; moved remaining domain terms into `docs/PROJECT_CONTEXT.md`.
- Implemented Stage 3 ML baseline: LightGBM with GroupKFold CV (safe numeric features only, no TVT_input). CV RMSE 120.06, valid submission generated.
- Implemented Stage 4 ML baseline: added last_tvt_input as well-level constant feature. CV RMSE improved to 20.84 (5.8x improvement).
- Switched Kaggle notebook to offline mode: uses `rogii-repo` Kaggle Dataset instead of git clone. Works with Internet OFF.
- First Kaggle submission: Stage 4 LightGBM + last_tvt_input. Official LB RMSE: 24.114 (CV: 20.58). Recorded in `docs/EXPERIMENT_LOG.md`.
- Created `docs/HOW_IT_WORKS.md` — beginner-friendly model explanation in Russian.

## 2026-06-05

- Created `docs/ROADMAP.md` for post-baseline development stages and promotion gates.
- Froze Stage 4 LightGBM + `last_tvt_input` as the reference baseline.
- Recorded reviewed public notebooks and rejected high-risk artifact/coordinate-overlap routes for the clean mainline roadmap.
- Added `ADR-003` for the baseline freeze and roadmap decision.
- Implemented Roadmap R1: residual target (`TVT - last_tvt_input`), 9 geometry features, 20 GR features (38 total). CV 14.09, ~32% improvement over Stage 4.
- Added `ADR-004` for residual target and forward-looking GR feature decisions.
- Implemented Roadmap R2: 15 typewell-reference features (11 anchor-offset + 4 summary). CV degraded to 14.75 → not promoted. Added `ADR-005`.
- Ran feature importance ablation on R1: dropped 20 zero/low-importance features (GR_is_missing, MD_delta, row_position, 5 rolling windows, 8 lag/leads, gr_d1, gr_d2, gr_envelope). Final set: 18 features with identical CV (14.19).
- Added `docs/HOW_IT_WORKS.md` — feature-by-feature explanation with importance analysis and ablation rationale.

## Open questions

- None.
