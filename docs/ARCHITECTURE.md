# Architecture

## Purpose

Describe the selected project architecture, component boundaries and runtime assumptions.

## Owns

Architecture decisions, component responsibilities, data/control flow, dependency boundaries and runtime assumptions.

## Update when

- Project structure changes.
- New major pipeline components are added.
- Local, GitHub or Kaggle execution flow changes.
- Dependency or runtime assumptions change.

## Do not store here

- Experiment results.
- Detailed data schema.
- Backlog tasks.
- Long implementation notes.

## Current content

## Overview

This project uses a balanced Kaggle ML baseline architecture: a small reusable Python package under `src/rogii/`, thin command-line scripts under `scripts/`, configuration files under `configs/`, tests under `tests/`, source-of-truth documentation under `docs/`, and thin notebooks under `notebooks/`.

The design optimizes for a fast first valid baseline while keeping enough structure for reproducibility, MLflow tracking, validation checks and Kaggle execution.

## Selected architecture

Selected option: Option B — Balanced.

The project will use:

- `src/rogii/` for reusable data, feature, validation, model, submission and MLflow logic.
- `scripts/` for executable entry points that call `src/rogii/` code.
- `configs/` for local/Kaggle paths and baseline parameters.
- `tests/` for contract, validation and smoke tests.
- `notebooks/` only for lightweight exploration and Kaggle thin runners.
- `docs/` for source-of-truth project context and decisions.

## Components

| Component | Responsibility | Main files/directories |
|---|---|---|
| Project context and docs | Source-of-truth context, decisions, tasks and risks | `docs/` |
| Config layer | Environment paths, model settings and run settings | `configs/*.yaml`, `src/rogii/config.py`, `src/rogii/paths.py` |
| Data inventory | File/schema inspection and data map generation | `scripts/make_data_inventory.py`, `src/rogii/data_inventory.py` |
| Data loading | Load Kaggle/local files without hardcoded paths | `src/rogii/data_loading.py` |
| Feature engineering | Safe feature construction after schema inspection | `src/rogii/features.py` |
| Validation | Group-aware splits and leakage checks | `src/rogii/validation.py` |
| Metrics | Local implementation of official Kaggle metric | `src/rogii/metrics.py` |
| Models | Naive and classical ML model wrappers plus saved model payload contracts | `src/rogii/models.py`, `src/rogii/model_io.py` |
| Training | Fit models, evaluate folds and log runs | `src/rogii/train.py`, `scripts/run_train.py` |
| Prediction | Load trained model/config and produce predictions | `src/rogii/predict.py`, `scripts/run_predict.py` |
| Submission | Validate and write `submission.csv` | `src/rogii/submission.py`, `scripts/validate_submission.py` |
| MLflow tracking | Centralized run metadata, metrics and artifacts | `src/rogii/mlflow_utils.py` |
| Kaggle runner | Thin execution wrappers for Kaggle: training, inference and offline path discovery | `src/rogii/kaggle_runtime.py`, `scripts/kaggle_offline_inference.py`, `scripts/kaggle_runner.py`, `notebooks/00_kaggle_inference.ipynb`, `notebooks/01_kaggle_train.ipynb`, `notebooks/kernel-metadata.json` |
| Tests | Submission, validation, metric and smoke contracts | `tests/` |

## Boundaries

- Core reusable logic belongs in `src/rogii/`.
- Scripts should parse arguments, load configs and call reusable code; scripts should not contain complex ML logic.
- Notebooks should not own training logic.
- Docs should record facts and decisions, not raw data or large outputs.
- Raw Kaggle data, trained models, submissions and MLflow artifact stores must stay out of Git.
- The repository should not assume exact data schema until data is inspected.

## Data/control flow

1. User or Kaggle environment provides competition data under a configured data directory.
2. `scripts/make_data_inventory.py` inspects files and updates or informs `docs/DATA_MAP.md`.
3. Config files select local or Kaggle paths and baseline settings.
4. Baseline scripts load data via `src/rogii/data_loading.py`.
5. Feature logic builds leakage-audited train/test matrices.
6. Validation logic creates local folds, preferably group-aware by well/group ID when available.
7. Model training computes local scores and logs params, metrics and artifacts to MLflow.
8. Training saves a versioned model payload with target mode, feature flags and exact feature columns.
9. Prediction validates the generated feature matrix against the saved payload before writing `submission.csv` with schema matching `sample_submission.csv`.
10. Kaggle runner executes the same repository code on Kaggle full data.
11. Offline inference resolves mounted repo/model/data paths by file markers, validates `submission.csv`, and can be submitted through a user-approved Kaggle kernel-version submit.
12. User or explicitly approved agent submission records public LB score in docs and MLflow notes.

## Dependencies

Initial preferred stack:

- Python.
- `pandas`, `numpy`, `scikit-learn`, `pyyaml`, `mlflow`, `pytest`.
- `lightgbm` for the first model baseline if available/installable.
- `catboost` and `xgboost` as later baseline comparisons.
- `matplotlib` for lightweight reports and feature importance plots.

Optional later dependencies must be justified before adding.

## Deployment/runtime assumptions

- Local machine is the main development environment.
- Public GitHub repo `https://github.com/Lainterus1/ROGII_Kaggle_Competitions` is the source of truth.
- Kaggle is a remote executor for full-data runs and submission generation.
- Kaggle submissions require explicit user approval; after approval, an agent may submit a validated kernel version through the Kaggle CLI/API.
- Local MLflow uses `mlruns/`; Kaggle may use `/kaggle/working/mlruns`.
- Kaggle training/update notebooks can clone the public repo without GitHub auth or Kaggle Secrets.
- Kaggle offline inference uses `rogii-repo-v2` and `rogii-models-v2` Datasets with internet OFF and versioned kernel metadata.

## Architecture risks

- Some modules will be thin until actual data schema is known.
- Kaggle official pages may require Kaggle API or authenticated access for full details.
- Validation cannot be finalized until group/well IDs are confirmed.
- Training and repo-update notebooks still depend on pushed repository code being available on `main`.
- A2a DWT inference is now packaged: `rogii-wheels-a2a-dwt` (pywavelets) + `rogii-models-a2a-dwt` + kernel `00-rogii-inference-a2a-dwt`. Kaggle base env also has pywavelets 1.9.0 pre-installed.

## Alternatives considered

| Option | Summary | Outcome |
|---|---|---|
| Option A — Simple | Scripts-first project with minimal structure | Rejected because logic would likely become scattered after MLflow, validation and Kaggle runner are added |
| Option B — Balanced | Small package plus scripts, configs, tests and docs | Accepted because it matches the dossier and balances speed with reproducibility |
| Option C — Scalable | Heavier pipeline/orchestration structure | Rejected for initial baseline because it would over-engineer before data inspection |

## Current constraints

- Python ML/DS project.
- Local development, GitHub source of truth, Kaggle execution.
- Core logic must live in `src/` and `scripts/`.
- Kaggle notebooks must be thin runners.
- MLflow is required for model baseline runs.

## Open questions

- What is the official Kaggle evaluation metric?
- What are the exact data files, target and submission schema?
