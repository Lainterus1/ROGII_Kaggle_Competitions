# ROGII Kaggle Competitions

Reproducible baseline project for the Kaggle competition `ROGII - Wellbore Geology Prediction`.

Competition page: <https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview>

GitHub repository: <https://github.com/Lainterus1/ROGII_Kaggle_Competitions>

## Goal

Build a strong, reproducible baseline that can run locally on small samples and on Kaggle for full-data execution and manual submission generation.

## Workflow

1. Develop locally.
2. Store code, configs and docs in GitHub repo `ROGII_Kaggle_Competitions`.
3. Execute full-data runs on Kaggle using thin runners.
4. Manually submit validated `submission.csv` after user approval.

Public clone command:

```bash
git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git
```

## Current status

Bootstrap in progress.

Completed:

- Project context document.
- Source-of-truth documentation skeletons.
- Initial balanced architecture decision.
- Initial project skeleton.

Not completed yet:

- Kaggle data inspection.
- Official metric confirmation.
- Submission contract confirmation.
- Naive/model baselines.

## Planned setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Planned commands

These commands are placeholders until the pipeline is implemented in later steps:

```bash
pytest tests/
python scripts/make_data_inventory.py --config configs/paths.local.yaml.example
python scripts/run_smoke.py --config configs/baseline_lgbm.yaml
python scripts/run_naive_baseline.py --config configs/baseline_naive.yaml
python scripts/run_train.py --config configs/baseline_lgbm.yaml --env local
python scripts/validate_submission.py --submission outputs/submission.csv
```

## Documentation

Read these first:

- `ROGII_PROJECT_INTAKE_DOSSIER.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/CONTEXT_MAP.md`
- `docs/ARCHITECTURE.md`
- `docs/DECISIONS.md`

## Data and artifacts

Do not commit Kaggle data, secrets, trained models, submissions or MLflow artifact stores. Runtime files belong in ignored directories such as `data/`, `outputs/`, `models/`, `submissions/` and `mlruns/`.

Local data is expected under `data/`. Kaggle data is expected under `/kaggle/input`; Kaggle outputs should be written to `/kaggle/working`.
