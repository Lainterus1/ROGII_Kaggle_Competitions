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

## Open questions

- None.
