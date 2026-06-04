---
name: kaggle-runner
description: Use when creating or changing Kaggle notebook/script execution for this public GitHub based workflow.
---

# Kaggle Runner

## When to use

Use this skill when editing Kaggle runner scripts, Kaggle notebook instructions, or local-to-GitHub-to-Kaggle execution flow.

## Inputs

- Public GitHub repo: `https://github.com/Lainterus1/ROGII_Kaggle_Competitions`.
- Clone command: `git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git`.
- Kaggle input root: `/kaggle/input`.
- Kaggle output root: `/kaggle/working`.
- Runner files under `scripts/` and `notebooks/`.

## Source-of-truth files

- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_MAP.md`
- `docs/DECISIONS.md`
- `configs/paths.kaggle.yaml`
- `scripts/kaggle_runner.py`
- `notebooks/00_kaggle_thin_runner.ipynb`
- `AGENTS.md`

## Procedure

1. Keep Kaggle notebooks thin: clone repo, install requirements, configure paths, run scripts.
2. Put reusable logic in `src/rogii/` or `scripts/`, not notebook cells.
3. Use `configs/paths.kaggle.yaml` for Kaggle paths.
4. Write outputs to `/kaggle/working`.
5. Do not require GitHub auth, tokens or Kaggle Secrets for cloning the public repo.
6. Ensure notebook commands match pushed repository files.
7. Document any Kaggle-specific runtime limitation.

## Documentation updates

- Update `docs/ARCHITECTURE.md` if runtime flow changes.
- Update `docs/DECISIONS.md` for meaningful runner strategy changes.
- Update `docs/TASKS.md` when Kaggle execution readiness changes.
- Update `README.md` if user-facing commands change.

## Validation

- Run the relevant local script with local config when possible.
- In Kaggle, verify clone, install and script invocation steps.
- Run `python -m pytest tests` after code changes.

## Completion checklist

- [ ] Notebook remains thin.
- [ ] Kaggle paths use `/kaggle/input` and `/kaggle/working`.
- [ ] Commands use the public GitHub clone URL.
- [ ] No secrets are required for clone.
- [ ] Runner invokes repository scripts.

## Forbidden actions

- Do not put core training, feature or validation logic in notebooks.
- Do not write outputs to `/kaggle/input`.
- Do not assume unpublished local code is available on Kaggle.
