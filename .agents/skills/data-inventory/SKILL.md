---
name: data-inventory
description: Use when inspecting Kaggle data files, confirming schema, or updating docs/DATA_MAP.md for this ROGII baseline project.
---

# Data Inventory

## When to use

Use this skill when you need to inspect local `data/` files or Kaggle `/kaggle/input` files, confirm schema facts, or update the project data contract.

## Inputs

- Data root: local `data/` or Kaggle `/kaggle/input`.
- `data/sample_submission.csv` or Kaggle equivalent.
- Train/test well files, including `*_horizontal_well.csv` and `*_typewell.csv`.
- Task deck or official Kaggle pages when accessible.

## Source-of-truth files

- `docs/DATA_MAP.md`
- `docs/METRICS.md`
- `docs/KNOWN_ISSUES.md`
- `docs/VALIDATION_STRATEGY.md`
- `AGENTS.md`

## Procedure

1. Read `docs/CONTEXT_MAP.md`, `docs/DATA_MAP.md`, and `docs/KNOWN_ISSUES.md`.
2. Confirm the data root from config or runtime environment.
3. Inventory file counts, file sizes, extensions, train/test directories, sample submission and task deck files.
4. For CSV files, inspect headers, row counts, dtypes, missing values and small samples.
5. Extract well/group candidates from file prefixes before `__horizontal_well.csv` and `__typewell.csv`.
6. Confirm sample submission columns, row count and ID format.
7. Compare train/test horizontal and typewell columns.
8. Flag target-like or leakage-adjacent columns, especially `TVT`, `tvt`, `TVT_input`, and post-target-derived features.
9. Update `docs/DATA_MAP.md` with observed facts only.
10. Update `docs/KNOWN_ISSUES.md` for unresolved schema, metric, leakage or validation risks.

## Documentation updates

- Update `docs/DATA_MAP.md` for every new data/schema fact.
- Update `docs/METRICS.md` only after the official metric is confirmed.
- Update `docs/VALIDATION_STRATEGY.md` when group/well ID evidence changes.
- Update `docs/KNOWN_ISSUES.md` for blockers or unresolved data risks.

## Validation

- Run the data inventory script when implemented.
- Run `python -m pytest tests` after code changes.
- Check `git status --short --ignored` and verify `data/` remains ignored.

## Completion checklist

- [ ] No schema facts were invented.
- [ ] Sample submission columns and row count are documented.
- [ ] Train/test file patterns are documented.
- [ ] Target, ID and group candidates are documented as confirmed or preliminary.
- [ ] Leakage candidates are documented.
- [ ] No raw data files are staged.

## Forbidden actions

- Do not commit raw data, generated inventory artifacts, plots or large outputs.
- Do not assume the official metric from notebook scores or public discussions.
- Do not treat quick observations as final without documenting their scope.
