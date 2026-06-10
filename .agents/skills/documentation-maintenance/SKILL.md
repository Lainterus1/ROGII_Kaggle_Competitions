---
name: documentation-maintenance
description: Use when a task may change project contracts, user-facing behavior, data facts, validation, metrics, experiments, risks, or source-of-truth documentation.
---

# Documentation Maintenance

## When to use

Use this skill during every implementation task's review phase, and before finishing any task that changes contracts, behavior, data facts, validation, metrics, baseline status, experiments or risks.

Do not use it for pure formatting, test-only refactors or temporary debugging changes with no project impact.

## Inputs

- Task description and completed changes.
- `git diff` or changed-file list.
- Relevant source-of-truth docs from `docs/CONTEXT_MAP.md`.
- Test/check results.
- New facts, decisions, risks or assumptions discovered during the task.

## Source-of-truth files

- `AGENTS.md`
- `docs/CONTEXT_MAP.md`
- `docs/TASK_TEMPLATE.md`
- `docs/CHANGELOG.md`
- `docs/DECISIONS.md`
- Linear MCP for current task status and backlog
- `docs/TASKS.md` as historical archive only
- Domain-specific docs under `docs/`

## Procedure

1. Identify whether the task changed a project contract, public behavior, data fact, metric, validation rule, baseline stage, experiment record, risk or project structure.
2. Use `docs/CONTEXT_MAP.md` to find the one document that owns each changed fact.
3. Update the smallest relevant section in that document.
4. Add an ADR in `docs/DECISIONS.md` only for meaningful decisions with alternatives or long-lived consequences.
5. Update `docs/CHANGELOG.md` only for user-visible or operationally relevant changes.
6. Update the Linear issue when backlog status, blockers or next actions change.
7. If no docs update is needed, record the reason in the completion report.
8. Re-check docs for duplicated facts and stale `TBD`/placeholder statements.

## Documentation update matrix

| Change type | Update required | Do not update when |
|---|---|---|
| Data schema, file layout, target, ID or leakage fact | `docs/DATA_MAP.md`, `docs/KNOWN_ISSUES.md` if risk remains | The task only changes code formatting |
| Metric implementation or metric confirmation | `docs/METRICS.md` | Tests are refactored without metric behavior changes |
| Validation split, group rules or leakage controls | `docs/VALIDATION_STRATEGY.md`, `docs/DECISIONS.md` if significant | A helper is renamed without behavior change |
| Task status, backlog ordering, blocker or next action | Linear MCP only | The fact is historical context already captured elsewhere |
| Baseline stage, runnable command or acceptance criteria | `docs/BASELINE_PLAN.md`, `README.md` if user-facing, Linear issue result/status | Internal code cleanup with same commands |
| Experiment run | `docs/EXPERIMENT_LOG.md` and MLflow | Dry-run or failed command with no useful result |
| Public notebook idea adopted | `docs/PUBLIC_NOTEBOOK_REFERENCES.md`, optionally `docs/DECISIONS.md` | Notebook only reviewed and no idea used |
| Architecture boundary or project structure | `docs/ARCHITECTURE.md`, `docs/CONTEXT_MAP.md`, `docs/DECISIONS.md` if meaningful | A file moves inside the same owned component without contract change |
| New accepted decision | `docs/DECISIONS.md` | The decision is local and reversible within one task |
| New blocker, unresolved risk or limitation | `docs/KNOWN_ISSUES.md` | The issue is fixed in the same task and has no residual risk |
| User-facing setup or run command | `README.md`, `docs/CONTEXT_MAP.md` if ownership changes | Command is experimental and not supported yet |

## Update style

- Update the smallest relevant section.
- Do not duplicate the same fact across many docs.
- Do not mirror Linear issue state into `docs/TASKS.md`; it is historical.
- Do not append generic summaries.
- Prefer tables for contracts and ownership.
- Prefer ADR entries for important decisions.
- Keep changelog entries factual and short.
- Replace stale placeholders instead of adding contradictory notes.

## Validation

- Run `python -m pytest tests` after code changes.
- For docs-only changes, tests are optional unless the docs reference commands or contracts that should be verified.
- Run `git status --short --branch --ignored` before commit.
- Audit staged files to ensure ignored data, outputs, secrets and artifacts are not staged.

## Completion checklist

- [ ] Each changed fact has exactly one source-of-truth home.
- [ ] Linear issue reflects completed or newly discovered current work.
- [ ] `docs/CHANGELOG.md` includes only operationally relevant changes.
- [ ] Stale `TBD` statements were removed or kept with a reason.
- [ ] Completion report explains updated, skipped or deferred docs.

## Forbidden actions

- Do not create duplicate policy documents unless the existing docs cannot hold the policy clearly.
- Do not update docs just to say that nothing changed.
- Do not copy large command outputs into docs.
- Do not hide unresolved risks by removing them from `docs/KNOWN_ISSUES.md` without resolution.
