---
name: leakage-review
description: Use before adding features, changing validation, or using target-adjacent ROGII columns such as TVT_input.
---

# Leakage Review

## When to use

Use this skill before adding or changing features, folds, target handling, typewell alignment, postprocessing, or any use of `TVT_input`.

## Inputs

- Current feature list and config.
- Train/test schema from `docs/DATA_MAP.md`.
- Validation design from `docs/VALIDATION_STRATEGY.md`.
- Candidate target, ID and group columns.
- Relevant implementation files under `src/rogii/` and `scripts/`.

## Source-of-truth files

- `docs/DATA_MAP.md`
- `docs/VALIDATION_STRATEGY.md`
- `docs/KNOWN_ISSUES.md`
- `docs/DECISIONS.md`
- `src/rogii/features.py`
- `src/rogii/validation.py`
- `tests/test_no_target_leakage.py`

## Procedure

1. Identify target columns and aliases, including `TVT`, `tvt`, and any target-derived fields.
2. Compare train and test availability for every candidate feature.
3. Treat columns present only in train as excluded unless there is a documented non-leaky transform.
4. Audit `TVT_input` against official task materials before using it.
5. Confirm group-aware validation uses well/group IDs and has no overlap.
6. Check whether typewell features could encode target information unavailable at prediction time.
7. Document each accepted leakage-sensitive feature with rationale.
8. Add or update leakage tests for excluded target columns.

## Documentation updates

- Update `docs/VALIDATION_STRATEGY.md` when folds or group logic change.
- Update `docs/DATA_MAP.md` when target-like columns are discovered.
- Update `docs/KNOWN_ISSUES.md` for unresolved leakage risks.
- Add an ADR in `docs/DECISIONS.md` for meaningful leakage-sensitive choices.

## Validation

- Run leakage tests, especially `tests/test_no_target_leakage.py`.
- Run group split tests when validation changes.
- Run `python -m pytest tests` after code changes.

## Completion checklist

- [ ] Target columns are excluded from features.
- [ ] Train-only columns are excluded or explicitly justified.
- [ ] `TVT_input` usage is audited or blocked.
- [ ] Group overlap checks pass.
- [ ] Leakage assumptions are documented.

## Forbidden actions

- Do not use target-like columns because they improve CV.
- Do not use row-level random KFold as primary validation when wells can overlap.
- Do not silently change validation strategy.
