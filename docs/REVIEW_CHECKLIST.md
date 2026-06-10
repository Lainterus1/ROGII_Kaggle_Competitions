# Review Checklist

## Purpose

Provide a concise checklist for reviewing ROGII baseline changes.

## Owns

Review criteria, optimization boundaries and refactoring approval rules for completed tasks, diffs and pull requests.

## Update when

- Review criteria change.
- New high-risk workflow is added.
- Architecture, validation, submission or experiment contracts change in a way that affects review.

## Do not store here

- Detailed task backlog.
- Full code review findings.
- Experiment results.

## Current content

Use `.agents/skills/code-review/SKILL.md` for the full procedure.

Review every meaningful change for:

| Area | Check |
|---|---|
| Scope | Change is minimal and matches the task contract |
| Architecture | Reusable logic in `src/rogii/`, scripts thin, notebooks thin |
| Data contract | No invented schema, target, metric, ID or submission details |
| Leakage | Target-like columns and `TVT_input` usage are justified and tested |
| Validation | Group/well leakage controls are preserved when relevant |
| Submission | Columns, row count, ID order and finite predictions are validated |
| Tests | Relevant tests and scripts were run and reported |
| Dependencies | No heavy dependency added without approval |
| Artifacts | No raw data, outputs, submissions, models, secrets or `mlruns/` staged |
| Docs | Source-of-truth docs updated only where ownership requires it |
| Code quality | Efficiency, readability, interpretability and compactness checked per `code-review` skill |

Optimization boundaries:

- Optimize only measured or clearly risky paths.
- Do not perform broad refactoring as part of review.
- Do not change architecture without ADR.
- Do not optimize for public leaderboard without validation rationale.

Review report format:

| Priority | Issue | Evidence | Suggested fix | Risk |
|---|---|---|---|---|
| High | ... | ... | ... | ... |

If no findings are found, say so and list residual risks or skipped checks.

## Open questions

- None.
