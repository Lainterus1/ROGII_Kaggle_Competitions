# Linear Issue Contract Template

Use this template as reference material for Linear issue descriptions in this project. Current task tracking lives in Linear MCP (`ROG-*` issues), not in `docs/TASKS.md`. Keep each issue narrow, verifiable and aligned with the current source-of-truth docs.

## Task

[What needs to be done]

## Why

[Why this matters for the ROGII baseline]

## Context

Use:

- `AGENTS.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_MAP.md`
- Current Linear issue via MCP
- `docs/TASKS.md` only for historical pre-Linear context
- `docs/DATA_MAP.md` for data/schema work
- `docs/METRICS.md` for metric work
- `docs/VALIDATION_STRATEGY.md` for split, leakage or CV work
- `docs/BASELINE_PLAN.md` for baseline stage work
- `docs/EXPERIMENT_LOG.md` for experiment runs
- Relevant `.agents/skills/*/SKILL.md`

## Scope

Implement:

- [Specific files, scripts, modules or docs to change]

Do not implement:

- [Adjacent features explicitly out of scope]

## Acceptance Criteria

1. [Observable outcome]
2. [Required test/check]
3. [Documentation or artifact rule]

## Constraints

- Do not invent fields, APIs, env vars, metrics, target columns, ID columns or business rules.
- Do not change public contracts unless necessary and documented.
- Do not add dependencies without justification.
- Do not perform broad refactoring.
- Keep changes minimal and testable.
- Keep reusable logic in `src/rogii/`.
- Keep executable entry points in `scripts/`.
- Keep Kaggle notebooks thin.
- Do not commit raw Kaggle data, secrets, generated submissions, trained models, `mlruns/` or large artifacts.
- Do not use `TVT_input` or target-adjacent columns without leakage review.
- Do not submit to Kaggle without explicit user approval.

## Required Workflow

1. Read relevant docs and skills.
2. Use `linear-issue-manager` to find or create the Linear issue and move it to `In Progress`.
3. Identify minimal files to change.
4. State a short implementation plan if the task is non-trivial.
5. Implement the smallest correct change.
6. Add or update tests.
7. Run relevant checks.
8. Check documentation impact.
9. Update the Linear issue with result, blockers or follow-up status.
10. Inspect `git status`, `git diff`, and staged files before commit.
11. Report assumptions and risks.

## Required Checks

Choose checks relevant to the task:

- `python -m pytest tests`
- `python scripts/make_data_inventory.py --data-dir data`
- `python scripts/run_naive_baseline.py --data-dir data --output outputs/submission.csv`
- `python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv`
- `git status --short --branch --ignored`
- staged-file audit ensuring ignored data/artifacts/secrets are not committed

## Documentation Impact

Choose one:

- Updated docs:
  - [List files]
- No docs update required because:
  - [Reason]
- Docs update deferred because:
  - [Reason and follow-up]

Use the project matrix:

| Change type | Required documentation update |
|---|---|
| Architecture or component boundary change | `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` |
| New accepted decision | `docs/DECISIONS.md` |
| Data/schema discovery | `docs/DATA_MAP.md` |
| Metric confirmation or implementation change | `docs/METRICS.md` |
| Validation strategy change | `docs/VALIDATION_STRATEGY.md`, `docs/DECISIONS.md` if significant |
| Task creation, status, blocker or next action | Linear MCP only |
| Baseline stage change | `docs/BASELINE_PLAN.md`, Linear issue result/status |
| Experiment run | `docs/EXPERIMENT_LOG.md` and MLflow |
| Public notebook idea used | `docs/PUBLIC_NOTEBOOK_REFERENCES.md`, optionally `docs/DECISIONS.md` |
| New blocker or risk | `docs/KNOWN_ISSUES.md` |
| Project structure change | `docs/CONTEXT_MAP.md`, `docs/CHANGELOG.md` |

## Completion Report

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
