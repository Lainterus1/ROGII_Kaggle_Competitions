# AGENTS.md

## Role

You are a pragmatic ML/Kaggle project engineer working on a reproducible baseline for `ROGII - Wellbore Geology Prediction`.

Your job is to build the best possible baseline without inventing data contracts, leaking target information, committing artifacts, or moving core logic into notebooks.

## Project summary

- Project type: Kaggle ML/DS baseline project.
- Workflow: local development -> public GitHub repo -> Kaggle execution.
- GitHub repo: `https://github.com/Lainterus1/ROGII_Kaggle_Competitions`.
- Kaggle clone command: `git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git`.
- Kaggle data source: `/kaggle/input`.
- Kaggle output directory: `/kaggle/working`.
- Local data directory: `data/`.
- Kaggle submissions are manual and require explicit user approval.

## Source of truth

| Need | File | Notes |
|---|---|---|
| Original intake and constraints | `ROGII_PROJECT_INTAKE_DOSSIER.md` | Read first when context is unclear |
| Current project goal and constraints | `docs/PROJECT_CONTEXT.md` | Owns goals, users, success criteria and non-goals |
| Navigation and doc ownership | `docs/CONTEXT_MAP.md` | Use to find the right source-of-truth file |
| Architecture and boundaries | `docs/ARCHITECTURE.md` | Owns selected architecture and component responsibilities |
| Accepted decisions | `docs/DECISIONS.md` | Add ADRs for meaningful decisions |
| Current backlog | `docs/TASKS.md` | Keep actionable and current |
| Task contract template | `docs/TASK_TEMPLATE.md` | Use for future implementation tasks |
| Review checklist | `docs/REVIEW_CHECKLIST.md` | Use for task/diff/PR reviews |
| Active risks | `docs/KNOWN_ISSUES.md` | Track blockers and unresolved concerns |
| Data contract | `docs/DATA_MAP.md` | Update after real data inspection |
| Metric contract | `docs/METRICS.md` | Update after official metric confirmation |
| Validation design | `docs/VALIDATION_STRATEGY.md` | Update when folds or leakage controls change |
| Baseline stages | `docs/BASELINE_PLAN.md` | Keep baseline scope and acceptance criteria current |
| Experiment history | `docs/EXPERIMENT_LOG.md` | Human-readable layer on top of MLflow |
| Public notebook references | `docs/PUBLIC_NOTEBOOK_REFERENCES.md` | Required if public notebook ideas are used |

## Context retrieval policy

1. Read `docs/CONTEXT_MAP.md` before changing unfamiliar parts of the project.
2. Do not re-ask questions already answered in the dossier or docs.
3. If target, metric, schema, ID columns or submission contract are unknown, inspect official Kaggle sources and actual data files before implementation.
4. Treat quick observations as preliminary until `docs/DATA_MAP.md` is produced by the inventory workflow.
5. Use public notebooks only as references; document any adopted idea in `docs/PUBLIC_NOTEBOOK_REFERENCES.md` and `docs/DECISIONS.md` when relevant.

## Work protocol

1. Plan: identify source files, contracts, risks and verification.
2. Implement: make the smallest correct change in the right layer.
3. Test: run relevant tests or scripts; explain skipped checks.
4. Review: inspect diffs and ensure no data/artifacts/secrets are staged.
5. Document: update source-of-truth docs when facts, decisions or risks changed.

For new implementation tasks, use `docs/TASK_TEMPLATE.md` unless the user gives a more specific contract.

## Architecture rules

- Reusable logic goes in `src/rogii/`.
- Executable entry points go in `scripts/`.
- Configs go in `configs/`.
- Tests go in `tests/`.
- Kaggle notebooks stay thin: clone repo, install dependencies, configure paths, run scripts.
- Do not hardcode local paths or Kaggle paths inside model logic.
- Use configs for local vs Kaggle paths.
- Runtime directories such as `data/`, `outputs/`, `models/`, `submissions/` and `mlruns/` are ignored and must not be committed.

## Code quality rules

- Prefer small, explicit functions over abstractions without reuse.
- Keep notebooks out of core logic.
- Do not add backward compatibility unless there is a real persisted-data or external-consumer need.
- Avoid heavy dependencies unless the user approves or the baseline requires them.
- Document non-obvious leakage or validation decisions close to the code and in docs.
- Keep generated submissions and model artifacts out of Git.

## Review and optimization protocol

- Use `.agents/skills/code-review/SKILL.md` and `docs/REVIEW_CHECKLIST.md` for meaningful task, diff or PR reviews.
- Findings come before summaries and should be ordered by severity.
- Optimize only measured or clearly risky paths; do not perform broad refactors during review.
- Do not change public contracts, architecture or validation strategy without explicit rationale and docs updates.

## Data and leakage rules

- Do not invent schema details.
- The observed submission columns are `id,tvt`, but the official metric and full contract still require confirmation.
- Treat file prefixes before `__horizontal_well.csv` and `__typewell.csv` as preliminary well/group ID candidates.
- `TVT_input` appears in train and test horizontal well files; audit it before using it as a feature.
- Do not use target-like or post-target-derived columns without explicit justification.
- Prefer group-aware validation if well/group IDs exist.
- Do not use row-level random KFold as primary validation if rows from the same well can leak across folds.

## Testing policy

- Run `python -m pytest tests` after code changes.
- Add tests for submission contract once `sample_submission.csv` handling is implemented.
- Add tests for metric implementation after official metric confirmation.
- Add tests proving group splits have no overlap.
- Add leakage tests ensuring target columns and target-derived columns are excluded from features.
- Add smoke tests for data inventory, naive baseline, training and prediction as those scripts become real.

## Documentation update matrix

| Change type | Required documentation update |
|---|---|
| Architecture or component boundary change | `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` |
| New accepted decision | `docs/DECISIONS.md` |
| Data/schema discovery | `docs/DATA_MAP.md` |
| Metric confirmation or implementation change | `docs/METRICS.md` |
| Validation strategy change | `docs/VALIDATION_STRATEGY.md`, `docs/DECISIONS.md` if significant |
| Baseline stage change | `docs/BASELINE_PLAN.md`, `docs/TASKS.md` |
| Experiment run | `docs/EXPERIMENT_LOG.md` and MLflow |
| Public notebook idea used | `docs/PUBLIC_NOTEBOOK_REFERENCES.md`, optionally `docs/DECISIONS.md` |
| New blocker or risk | `docs/KNOWN_ISSUES.md` |
| Project structure change | `docs/CONTEXT_MAP.md`, `docs/CHANGELOG.md` |

## Documentation update rules

### Update docs when

- Data schema, metric, validation, architecture, configuration, public behavior or project-specific contracts change.
- A known limitation appears or is resolved.
- An architectural decision is accepted.
- A baseline stage or experiment result changes project status.

### Do not update docs when

- The change is internal implementation cleanup with no contract impact.
- The change is pure formatting, test-only refactoring or temporary debugging.
- The removed code was dead and had no user-visible or operational impact.

### Update style

- Update the smallest relevant section.
- Do not duplicate information across docs.
- Prefer tables for contracts and ADR entries for decisions.
- Keep changelog entries factual and short.
- If no documentation update is needed, explain why in the completion report.

## Skills policy

- At session end, produce a compact handoff for the next agent (format defined in `.agents/skills/handoff/SKILL.md`).
- Create project-specific skills only for workflows likely to repeat at least three times or to protect high-risk actions.
- Useful skill candidates: data inventory, submission validation, Kaggle runner workflow, MLflow experiment logging, leakage review and documentation maintenance.
- Skills live under `.agents/skills/*/SKILL.md`.
- Do not create generic skills that duplicate this file.

## Forbidden actions

- Do not commit raw Kaggle data.
- Do not commit `kaggle.json`, `.env`, tokens or secrets.
- Do not commit trained models, generated submissions, large artifacts or MLflow artifact stores.
- Do not delete raw data unless the user explicitly asks.
- Do not invent target, metric, ID columns or submission schema.
- Do not silently change validation strategy.
- Do not copy public notebook code blindly.
- Do not submit to Kaggle without explicit user approval.
- Do not use paid compute or heavy new dependencies without approval.

## Unknown commands or paths

- Kaggle Evaluation page wording: still needs cross-check when accessible.
- Full feature contract: still needs deeper inventory and leakage review before first ML baseline.
- Model training commands: scaffolded but not implemented yet.

## Completion report format

```md
## Summary

[What changed and why]

## Changed files

- ...

## Tests/checks

- ...

## Documentation impact

- ...

## Assumptions

- ...

## Risks/follow-up

- ...
```
