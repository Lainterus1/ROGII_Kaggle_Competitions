---
name: leakage-review
description: Use before adding features, changing validation, adding fold-aware/OOF features, changing targets, or using target-adjacent ROGII columns such as TVT_input.
---

# Leakage Review

## When to use

Use this skill before adding or changing features, folds, fold-aware/OOF feature generation, target handling, typewell alignment, postprocessing, or any use of `TVT_input`.

Use it especially for Roadmap stages A1-A3: trajectory features, causal DWT, strict OOF spatial KNN, DTW typewell alignment and target transforms.

## Inputs

- Current feature list and config.
- Train/test schema from `docs/DATA_MAP.md`.
- Validation design from `docs/VALIDATION_STRATEGY.md`.
- Roadmap stage contract from `docs/ROADMAP.md`.
- Candidate target, ID and group columns.
- Candidate reference rows and reference values for OOF features.
- Relevant implementation files under `src/rogii/` and `scripts/`.

## Source-of-truth files

- `docs/DATA_MAP.md`
- `docs/ROADMAP.md`
- `docs/VALIDATION_STRATEGY.md`
- `docs/KNOWN_ISSUES.md`
- `docs/DECISIONS.md`
- `src/rogii/features.py`
- `src/rogii/validation.py`
- `tests/test_no_target_leakage.py`
- `tests/test_validation_split.py`

## Procedure

1. Identify target columns and aliases, including `TVT`, `tvt`, and any target-derived fields.
2. Compare train and test availability for every candidate feature.
3. Treat columns present only in train as excluded unless there is a documented non-leaky transform.
4. Audit `TVT_input` against official task materials before using it.
5. Confirm group-aware validation uses well/group IDs and has no overlap.
6. For spatial KNN or other OOF features, verify the fold K reference set excludes validation wells entirely.
7. For spatial KNN, verify reference rows are pre-PS only and reference values come from known pre-PS `TVT_input`, not post-PS `TVT`.
8. For test-time spatial KNN, use the documented default: train pre-PS reference rows only, excluding test pre-PS rows unless a later ADR changes this.
9. For DWT or rolling signal features, verify whether the stage allows full-log context or requires causal/trailing windows; Stage A2a requires causal/trailing windows.
10. For typewell/DTW features, verify the alignment does not use post-PS `TVT` to guide the path and does not use raw DTW output as the final answer.
11. For target transforms, evaluate leakage and quality in reconstructed TVT scale, not only transformed target space.
12. Check predicted TVT variance/per-well dispersion when target engineering is used; flattening is a rollback signal.
13. Document each accepted leakage-sensitive feature with rationale.
14. Add or update leakage tests for excluded target columns and fold-aware reference construction.

## Documentation updates

- Update `docs/VALIDATION_STRATEGY.md` when folds or group logic change.
- Update `docs/DATA_MAP.md` when target-like columns are discovered.
- Update `docs/KNOWN_ISSUES.md` for unresolved leakage risks.
- Add an ADR in `docs/DECISIONS.md` for meaningful leakage-sensitive choices.

## Validation

- Run leakage tests, especially `tests/test_no_target_leakage.py`.
- Run group split tests when validation changes.
- Add or run OOF-specific tests for spatial KNN, e.g. `tests/test_spatial_oof.py` when implemented.
- Run `python -m pytest tests` after code changes.
- Treat implausibly low CV, especially spatial KNN RMSE around `2-3`, as leakage until proven otherwise; do not generate a Kaggle submission from that run.

## Completion checklist

- [ ] Target columns are excluded from features.
- [ ] Train-only columns are excluded or explicitly justified.
- [ ] `TVT_input` usage is audited or blocked.
- [ ] Group overlap checks pass.
- [ ] OOF references exclude validation wells and post-PS targets.
- [ ] Target transforms are scored in reconstructed TVT scale.
- [ ] Prediction variance/flattening was checked when target engineering changed.
- [ ] Leakage assumptions are documented.

## Forbidden actions

- Do not use target-like columns because they improve CV.
- Do not use row-level random KFold as primary validation when wells can overlap.
- Do not silently change validation strategy.
- Do not precompute fold-aware features on the full train data before CV.
- Do not submit a run with suspiciously low CV before completing leakage audit.
