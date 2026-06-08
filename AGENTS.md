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
| Original intake and constraints | `docs/PROJECT_CONTEXT.md` | Read first when context is unclear |
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
| Post-baseline roadmap | `docs/ROADMAP.md` | Keep future development stages, priorities and promotion gates current |
| Experiment history | `docs/EXPERIMENT_LOG.md` | Human-readable layer on top of MLflow |
| Public notebook references | `docs/PUBLIC_NOTEBOOK_REFERENCES.md` | Required if public notebook ideas are used |
| Feature reference | `docs/HOW_IT_WORKS.md` | Feature-by-feature explanation, importance and ablation rationale |

## Context retrieval policy

1. Read `docs/CONTEXT_MAP.md` before changing unfamiliar parts of the project.
2. Do not re-ask questions already answered in the dossier or docs.
3. If target, metric, schema, ID columns or submission contract are unknown, inspect official Kaggle sources and actual data files before implementation.
4. Treat new data/schema observations as preliminary until `docs/DATA_MAP.md` is updated by the inventory workflow.
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
- Avoid heavy new dependencies unless the user approves or the active baseline stage requires them.
- Document non-obvious leakage or validation decisions close to the code and in docs.
- Keep generated submissions and model artifacts out of Git.

## Review and optimization protocol

- Use `.agents/skills/code-review/SKILL.md` and `docs/REVIEW_CHECKLIST.md` for meaningful task, diff or PR reviews.
- Findings come before summaries and should be ordered by severity.
- Optimize only measured or clearly risky paths; do not perform broad refactors during review.
- Do not change public contracts, architecture or validation strategy without explicit rationale and docs updates.

## Data and leakage rules

- Do not invent schema details.
- Submission columns are `id,tvt` per `sample_submission.csv`.
- Use file prefixes before `__horizontal_well.csv` and `__typewell.csv` as `well_id` and the default validation group.
- `TVT_input` appears in train and test horizontal well files; it is allowed only as the known pre-PS anchor/baseline described by the task deck.
- Do not use target-like or post-target-derived columns without explicit justification.
- Use group-aware validation by `well_id` by default.
- Do not use row-level random KFold as primary validation because rows from the same well can leak across folds.

## Testing policy

- Run `python -m pytest tests` after code changes.
- Keep submission-contract, metric, group-split, leakage, smoke, OOF and TCN tests current when those contracts change.
- Add or update tests for any new feature family, post-processing step, model payload field, validation rule or Kaggle runtime behavior.

## Documentation update matrix

| Change type | Required documentation update |
|---|---|
| Architecture or component boundary change | `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` |
| New accepted decision | `docs/DECISIONS.md` |
| Data/schema discovery | `docs/DATA_MAP.md` |
| Metric confirmation or implementation change | `docs/METRICS.md` |
| Validation strategy change | `docs/VALIDATION_STRATEGY.md`, `docs/DECISIONS.md` if significant |
| Baseline stage change | `docs/BASELINE_PLAN.md`, `docs/TASKS.md` |
| Roadmap stage or priority change | `docs/ROADMAP.md`, `docs/TASKS.md` |
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

- `kagglehub.dataset_upload()` is the correct Python tool for creating Kaggle datasets with directory structure. `kaggle datasets version -p` (CLI) skips subdirectories — never use it for repo datasets.
- Kaggle Evaluation page wording: still needs cross-check when accessible.
- Current active baseline is R3: 3-seed LightGBM `[42, 7, 123]` with R1 features + Savgol `w=31 p=2`, LB `12.177`.
- Current active development is A5 TCN. Phase 2 dual normalization is implemented but still needs the full/screening training gate before promotion.

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
