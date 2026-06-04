---
name: code-review
description: Use when reviewing a module, diff, pull request, or completed agent task for correctness, architecture, test coverage, maintainability, and documentation impact.
---

# Code Review

## When to use

Use this skill before merging or accepting meaningful changes, when asked for a review, or after an agent completes a non-trivial implementation task.

Do not use it as permission to rewrite unrelated code. Review first; propose or make only minimal targeted fixes when the user asks for implementation.

## Inputs

- Review target: changed files, module, diff, PR, or completed task.
- Relevant docs: source-of-truth docs from `docs/CONTEXT_MAP.md`.
- Recent task contract: `docs/TASK_TEMPLATE.md` or user-provided task details.
- Test results: pytest, scripts, Kaggle runs, MLflow/log outputs if relevant.
- Current git diff and status.

## Source-of-truth files

- `AGENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_MAP.md`
- `docs/DECISIONS.md`
- `docs/TASK_TEMPLATE.md`
- `docs/REVIEW_CHECKLIST.md`
- `docs/DATA_MAP.md`
- `docs/METRICS.md`
- `docs/VALIDATION_STRATEGY.md`
- `docs/KNOWN_ISSUES.md`

## Procedure

1. Read the relevant docs and task contract.
2. Identify intended behavior and acceptance criteria.
3. Inspect only files relevant to the review target.
4. Check architecture boundaries: `src/rogii/` for reusable logic, `scripts/` for entry points, notebooks thin.
5. Check project contracts: data schema, metric, submission, validation, leakage and runtime paths.
6. Check tests: coverage for changed behavior, meaningful assertions, and expected commands run.
7. Check error handling and edge cases: missing files, NaNs, non-finite predictions, row order, group overlap.
8. Check performance-sensitive paths: avoid unnecessary full-data loads, repeated CSV reads, or excessive artifacts.
9. Check dependency changes: no heavy or paid dependencies without approval.
10. Check documentation impact with `documentation-maintenance` rules.
11. Produce findings before summaries or optimization suggestions.

## Findings format

| Priority | Issue | Evidence | Suggested fix | Risk |
|---|---|---|---|---|
| High | ... | File/line or command output | ... | ... |

If no findings are found, state that explicitly and list residual risks or unverified checks.

## Review criteria

- Correctness against task acceptance criteria.
- No schema, target, metric, ID or submission contract invention.
- No target leakage or validation leakage.
- No raw data, outputs, submissions, secrets, models or MLflow artifacts staged.
- Minimal change scope.
- Tests and commands match the changed behavior.
- Documentation is updated only where ownership requires it.

## Optimization boundaries

- Optimize only paths shown to be slow, risky or operationally blocking.
- Prefer algorithmic or IO reductions before adding dependencies.
- Do not optimize public leaderboard score at the expense of validation trust.
- Do not add caching or artifact generation unless ignored and documented.
- Defer broad performance work until a baseline command is proven too slow.

## Refactoring rules

- Prefer minimal safe changes.
- Do not change public contracts without approval.
- Do not introduce abstractions unless they remove clear duplication or risk.
- Do not refactor unrelated modules.
- Do not change architecture without ADR.
- Do not add backward compatibility unless there is persisted data or an external consumer.

## Validation

- Run `python -m pytest tests` after code changes.
- Run affected scripts for behavior changes, such as inventory, naive baseline or submission validator.
- Run `git status --short --branch --ignored` and inspect staged files before commit.
- For docs-only review protocol changes, tests are still preferred because repo commands reference current tests.

## Completion checklist

- [ ] Relevant docs inspected.
- [ ] Architecture boundaries checked.
- [ ] Data, metric, validation and submission contracts checked when relevant.
- [ ] Leakage risks checked.
- [ ] Tests and command outputs checked.
- [ ] Documentation impact checked.
- [ ] Findings prioritized.
- [ ] Generated artifacts and ignored files not staged.

## Forbidden actions

- Do not rewrite code during protocol creation.
- Do not perform broad refactors while reviewing.
- Do not approve changes that commit raw data, generated submissions, secrets or model artifacts.
- Do not bury high-severity findings below summaries.
