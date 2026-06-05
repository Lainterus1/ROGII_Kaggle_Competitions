---
name: kaggle-runner
description: Use when creating or changing Kaggle notebook/script execution, Kaggle Dataset/GitHub sync, or notebook instructions for this ROGII workflow.
---

# Kaggle Runner

## When to use

Use this skill when editing Kaggle runner scripts, Kaggle notebook instructions, local-to-GitHub-to-Kaggle execution flow, or preparing instructions after a code push/update intended for Kaggle.

## Inputs

- Public GitHub repo: `https://github.com/Lainterus1/ROGII_Kaggle_Competitions`.
- Clone command: `git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git`.
- Current notebook may use an offline `rogii-repo` Kaggle Dataset instead of live clone; inspect the notebook before giving instructions.
- Kaggle input root: `/kaggle/input`.
- Kaggle output root: `/kaggle/working`.
- Runner files under `scripts/` and `notebooks/`.
- Candidate train/predict/validate commands.

## Source-of-truth files

- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_MAP.md`
- `docs/DECISIONS.md`
- `configs/paths.kaggle.yaml`
- `scripts/kaggle_runner.py`
- `notebooks/00_kaggle_inference.ipynb`
- `notebooks/01_kaggle_train.ipynb`
- `notebooks/02_kaggle_update_repo.ipynb`
- `AGENTS.md`

## Procedure (3-notebook git-clone workflow, ADR-007 + ADR-008)

Three notebooks, two with internet ON (clone fresh code), one OFF (submission):

| Notebook | Internet | Purpose |
|---|---|---|
| `02_kaggle_update_repo` | ON | `!git clone` → user creates `rogii-repo` Dataset |
| `01_kaggle_train` | ON | `!git clone` → train → user creates `rogii-models` Dataset |
| `00_kaggle_inference` | OFF | loads from `rogii-repo` + `rogii-models` → predict → submission |

1. Keep Kaggle notebooks thin: clone repo, install requirements, configure paths, run scripts.
2. Put reusable logic in `src/rogii/` or `scripts/`, not notebook cells.
3. Use `configs/paths.kaggle.yaml` for Kaggle paths.
4. Write outputs to `/kaggle/working`.
5. Do not require GitHub auth, tokens or Kaggle Secrets for cloning the public repo.
6. Ensure notebook commands match pushed repository files.
7. **Repo update notebook** (`02_kaggle_update_repo.ipynb`): internet ON, `!git clone`, user creates `rogii-repo` Dataset (private). Run after every git push.
8. **Training notebook** (`01_kaggle_train.ipynb`): internet ON, `!git clone`, runs `scripts/run_train.py`, saves model to `/kaggle/working/baseline_lgbm.pkl`. After run: user creates/updates `rogii-models` Kaggle Dataset from the model output.
9. **Inference notebook** (`00_kaggle_inference.ipynb`): internet OFF, uses `rogii-repo` + `rogii-models` Datasets, runs `scripts/run_predict.py` only (no training). Model payload auto-detects feature flags — no CLI flags needed.
10. After a push/update intended for Kaggle, tell the user:
    - Run `02_kaggle_update_repo.ipynb` → Create Dataset `rogii-repo`.
    - If model/features changed: run `01_kaggle_train.ipynb` → Create Dataset `rogii-models`.
    - Run `00_kaggle_inference.ipynb` → download `submission.csv` → manual submit.
11. Generate and validate `submission.csv`; do not submit it automatically.
12. Document any Kaggle-specific runtime limitation.

## Documentation updates

- Update `docs/ARCHITECTURE.md` if runtime flow changes.
- Update `docs/DECISIONS.md` for meaningful runner strategy changes.
- Update `docs/TASKS.md` when Kaggle execution readiness changes.
- Update `README.md` if user-facing commands change.
- Update `docs/EXPERIMENT_LOG.md` only after a meaningful run or after the user reports a real public LB score.

## Validation

- Run the relevant local script with local config when possible.
- In Kaggle, verify clone, install and script invocation steps.
- Always run submission validation before the user manually submits.
- Run `python -m pytest tests` after code changes.

## Completion checklist

- [ ] Notebooks remain thin (3 notebooks total).
- [ ] Kaggle paths use `/kaggle/input` and `/kaggle/working`.
- [ ] `02_kaggle_update_repo.ipynb` clones from GitHub, user creates `rogii-repo` Dataset.
- [ ] `01_kaggle_train.ipynb` clones from GitHub, trains, user creates `rogii-models` Dataset.
- [ ] `00_kaggle_inference.ipynb` is offline, loads from `rogii-repo` + `rogii-models`, no training.
- [ ] No secrets are required for cloning the public repo.
- [ ] Runners invoke repository scripts.
- [ ] After a Kaggle-intended push: instructions cover full 3-step workflow.
- [ ] `submission.csv` was validated before manual submission.

## Forbidden actions

- Do not put core training, feature or validation logic in notebooks.
- Do not write outputs to `/kaggle/input`.
- Do not assume unpublished local code is available on Kaggle.
- Do not implement automatic Kaggle submission.
- Do not submit to Kaggle; the user submits manually only.
