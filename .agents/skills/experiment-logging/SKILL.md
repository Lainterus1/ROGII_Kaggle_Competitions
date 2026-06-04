---
name: experiment-logging
description: Use when running or changing model baselines, MLflow logging, or docs/EXPERIMENT_LOG.md entries.
---

# Experiment Logging

## When to use

Use this skill when implementing or running naive/model baselines, changing MLflow utilities, or recording experiment results.

## Inputs

- Run config from `configs/`.
- Model type, feature set name and seed.
- Local CV scores and fold scores.
- MLflow output location.
- Public leaderboard score after manual user submission, if available.

## Source-of-truth files

- `docs/EXPERIMENT_LOG.md`
- `docs/BASELINE_PLAN.md`
- `docs/METRICS.md`
- `docs/VALIDATION_STRATEGY.md`
- `src/rogii/mlflow_utils.py`
- `src/rogii/train.py`
- `configs/*.yaml`
- `AGENTS.md`

## Procedure

1. Start from a config file and record its path.
2. Log run name, model type, feature set name, seed and environment.
3. Log validation strategy, fold count and group column when known.
4. Log model parameters and safe feature list.
5. Log fold metrics, mean CV and standard deviation.
6. Log artifacts such as config snapshot, feature importance and validation report when implemented and size-safe.
7. Record a concise entry in `docs/EXPERIMENT_LOG.md`.
8. Add public leaderboard score only after manual Kaggle submission.

## Documentation updates

- Update `docs/EXPERIMENT_LOG.md` for every meaningful run.
- Update `docs/BASELINE_PLAN.md` when a baseline stage is completed.
- Update `docs/METRICS.md` when metric behavior changes.
- Update `docs/KNOWN_ISSUES.md` for run blockers or suspicious CV/LB gaps.

## Validation

- Confirm MLflow run files are written only to ignored `mlruns/` or `/kaggle/working/mlruns`.
- Run `python -m pytest tests` after code changes.
- Check `git status --short --ignored` before commit.

## Completion checklist

- [ ] Config path is logged.
- [ ] Model type and feature set name are logged.
- [ ] Fold scores and mean CV are logged.
- [ ] Artifacts are size-safe and ignored when generated.
- [ ] Markdown experiment log is updated.

## Forbidden actions

- Do not commit `mlruns/`, models, OOF files or generated submissions.
- Do not record public leaderboard scores before a real manual submission.
- Do not compare experiments without noting metric and validation strategy.
