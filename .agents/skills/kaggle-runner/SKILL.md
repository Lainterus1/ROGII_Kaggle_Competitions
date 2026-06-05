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

## Procedure (3-notebook GitHub-linked workflow, ADR-008)

Three notebooks. `01` and `02` are linked to GitHub (File → Link to GitHub → `Lainterus1/ROGII_Kaggle_Competitions`). `00` is offline, uses `rogii-repo` Dataset for code.

### One-time setup (in Kaggle)

Create three notebooks and link two to GitHub:

| Notebook | File → Link to GitHub | Internet |
|---|---|---|
| `02_kaggle_update_repo` | `notebooks/02_kaggle_update_repo.ipynb` | ON |
| `01_kaggle_train` | `notebooks/01_kaggle_train.ipynb` | ON |
| `00_kaggle_inference` | Do NOT link | OFF |

### After every git push

```
1. Open 02_kaggle_update_repo → File → Pull from GitHub → Run
   → Save Version → Create Dataset "rogii-repo" (private)

2. If model/features changed:
   Open 01_kaggle_train → File → Pull from GitHub → Run
   → Save Version → Create Dataset "rogii-models" (private)

3. Open 00_kaggle_inference → Run (internet OFF, no pull)
   → Download submission.csv → Manual submit
```

1. Keep Kaggle notebooks thin: clone repo, run scripts.
2. Put reusable logic in `src/rogii/` or `scripts/`, not notebook cells.
3. Write outputs to `/kaggle/working`.
4. Do not require GitHub auth, tokens or Kaggle Secrets for cloning the public repo.
5. Ensure notebook commands match pushed repository files.
6. **Repo update notebook** (`02_kaggle_update_repo.ipynb`): linked to GitHub, internet ON, `!git clone`, user creates `rogii-repo` Dataset (private). Before each run: File → Pull from GitHub.
7. **Training notebook** (`01_kaggle_train.ipynb`): linked to GitHub, internet ON, `!git clone`, runs `scripts/run_train.py`, saves model to `/kaggle/working/baseline_lgbm.pkl`. After run: user creates `rogii-models` Dataset. Before each run: File → Pull from GitHub.
8. **Inference notebook** (`00_kaggle_inference.ipynb`): NOT linked to GitHub, internet OFF, uses `rogii-repo` + `rogii-models` Datasets, runs `scripts/run_predict.py` only. Model payload auto-detects feature flags.
9. After a push/update intended for Kaggle, tell the user the 3-step workflow above.
10. Generate and validate `submission.csv`; do not submit it automatically.
11. Document any Kaggle-specific runtime limitation.

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
