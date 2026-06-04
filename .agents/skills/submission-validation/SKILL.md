---
name: submission-validation
description: Use before generating, validating, or preparing any Kaggle submission.csv for the ROGII competition.
---

# Submission Validation

## When to use

Use this skill when implementing submission code, validating `submission.csv`, or preparing a Kaggle output for manual submission.

## Inputs

- `sample_submission.csv` from local `data/` or Kaggle `/kaggle/input`.
- Candidate submission file.
- Submission schema facts from `docs/DATA_MAP.md`.
- Metric and target scale notes from `docs/METRICS.md` when available.

## Source-of-truth files

- `docs/DATA_MAP.md`
- `docs/METRICS.md`
- `src/rogii/submission.py`
- `scripts/validate_submission.py`
- `tests/test_submission_contract.py`
- `AGENTS.md`

## Procedure

1. Read the current sample submission schema from actual files, not memory.
2. Validate exact column names and order against `sample_submission.csv`.
3. Validate row count equals `sample_submission.csv`.
4. Validate ID values match `sample_submission.csv` exactly and in order.
5. Validate prediction column is numeric.
6. Reject NaN, inf and non-finite predictions.
7. Check prediction scale against documented target observations when available.
8. Write outputs only to ignored runtime directories such as `outputs/`, `submissions/` or `/kaggle/working`.
9. Update tests when validator behavior changes.

## Documentation updates

- Update `docs/DATA_MAP.md` when submission columns or ID format are confirmed.
- Update `docs/METRICS.md` if metric or target scale affects validation.
- Update `docs/KNOWN_ISSUES.md` for unresolved submission risks.

## Validation

- Run `python scripts/validate_submission.py --submission <path>` when implemented.
- Run `python -m pytest tests` after code changes.
- Check `git status --short --ignored` before commit.

## Completion checklist

- [ ] Submission columns match sample submission exactly.
- [ ] Submission row count matches sample submission.
- [ ] IDs match sample submission exactly and in order.
- [ ] Predictions are numeric and finite.
- [ ] No generated submission is staged.

## Forbidden actions

- Do not submit to Kaggle without explicit user approval.
- Do not commit generated `submission.csv` files.
- Do not change ID order to match model convenience.
