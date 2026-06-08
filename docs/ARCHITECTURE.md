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

The design optimizes for reproducible local/Kaggle experimentation while keeping notebooks thin and core logic testable. The current project has moved beyond the first baseline: R3 is the active LightGBM baseline and A5 is the active TCN/OOF architecture-diversity path.

## Selected architecture

Selected option: Option B — Balanced.

The project uses:

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
| Feature engineering | Safe feature construction and gated experimental feature families | `src/rogii/features.py`, `src/rogii/gr_dwt.py`, `src/rogii/spatial_features.py`, `src/rogii/typewell_alignment.py`, `src/rogii/geology_features.py`, `src/rogii/beam_search.py`, `src/rogii/formation_plane.py`, `src/rogii/z_physics.py` |
| Validation | Group-aware splits and leakage checks | `src/rogii/validation.py` |
| Metrics | Local implementation of official Kaggle metric | `src/rogii/metrics.py` |
| Models | Naive, LightGBM and TCN model contracts plus saved model payload metadata | `src/rogii/models.py`, `src/rogii/model_io.py`, `src/rogii/tcn_model.py` |
| Training | Fit models, evaluate folds and log runs | `src/rogii/train.py`, `scripts/run_train.py` |
| Prediction | Load trained model/config and produce predictions | `src/rogii/predict.py`, `scripts/run_predict.py` |
| Post-processing | Savgol smoothing, TVT clipping and rejected prediction-time blend hooks | `src/rogii/smoothing.py`, `src/rogii/postprocess.py`, `src/rogii/gr_matcher.py`, `scripts/inspect_tvt_range.py`, `scripts/visualize_postproc.py`, `scripts/eval_pop2_oof.py` |
| OOF and diagnostics | Persist out-of-fold predictions and analyze model failure modes | `src/rogii/oof.py`, `src/rogii/diagnostics.py`, `scripts/diagnose_tcn.py` |
| Sequence modeling | TCN sequence features, datasets, training and prediction | `src/rogii/sequence_features.py`, `src/rogii/sequence_data.py`, `src/rogii/tcn_model.py`, `scripts/tune_tcn.py`, `configs/a5_tcn.yaml` |
| Submission | Validate and write `submission.csv` | `src/rogii/submission.py`, `scripts/validate_submission.py` |
| MLflow tracking | Centralized run metadata, metrics and artifacts | `src/rogii/mlflow_utils.py` |
| Kaggle runner | Thin execution wrappers for Kaggle: training, inference, offline path discovery and candidate kernel folders | `src/rogii/kaggle_runtime.py`, `scripts/kaggle_offline_inference.py`, `scripts/kaggle_runner.py`, `notebooks/kernel-metadata.json`, `notebooks/kernels/*/` |
| Tests | Submission, validation, metric and smoke contracts | `tests/` |

## Boundaries

- Core reusable logic belongs in `src/rogii/`.
- Scripts should parse arguments, load configs and call reusable code; scripts should not contain complex ML logic.
- Notebooks should not own training logic.
- Docs should record facts and decisions, not raw data or large outputs.
- Raw Kaggle data, trained models, submissions and MLflow artifact stores must stay out of Git.
- The repository should not assume new schema details until data is inspected and `docs/DATA_MAP.md` is updated.

## Data/control flow

1. User or Kaggle environment provides competition data under a configured data directory.
2. `scripts/make_data_inventory.py` inspects files and updates or informs `docs/DATA_MAP.md`.
3. Config files select local or Kaggle paths and baseline settings.
4. Baseline scripts load data via `src/rogii/data_loading.py`.
5. Feature logic builds leakage-audited train/test matrices or sequence tensors.
6. Validation logic creates local folds by `well_id` using 5-fold `GroupKFold` by default.
7. Model training computes local scores and logs params, metrics and artifacts to MLflow.
8. Training saves a versioned model payload with target mode, feature flags and exact feature columns.
9. Prediction validates the generated feature matrix or TCN metadata against the saved payload before writing `submission.csv` with schema matching `sample_submission.csv`.
10. Kaggle runner executes the same repository code on Kaggle full data.
11. Offline inference resolves mounted repo/model/data paths by file markers, validates `submission.csv`, and can be submitted through a user-approved Kaggle kernel-version submit.
12. User or explicitly approved agent submission records public LB score in docs and MLflow notes.

## Dependencies

Current stack:

- Python `>=3.10`.
- `pandas`, `numpy`, `scikit-learn`, `pyyaml`, `mlflow`, `pytest`.
- `lightgbm` for the active R3 tabular baseline.
- `scipy` and `pywavelets` for feature/post-processing experiments.
- `numba` for beam-search experiments.
- `torch` for the active A5 TCN sequence-model path.
- `matplotlib` for lightweight reports and per-well visualization.

Deferred dependencies such as `catboost`, `xgboost` and `optuna` must be justified by the roadmap stage before adding.

## Deployment/runtime assumptions

- Local machine is the main development environment.
- Public GitHub repo `https://github.com/Lainterus1/ROGII_Kaggle_Competitions` is the source of truth.
- Kaggle is a remote executor for full-data runs and submission generation.
- Kaggle submissions require explicit user approval; after approval, an agent may submit a validated kernel version through the Kaggle CLI/API.
- Local MLflow uses `mlruns/`; Kaggle may use `/kaggle/working/mlruns`.
- Kaggle training/update notebooks can clone the public repo without GitHub auth or Kaggle Secrets.
- Kaggle offline inference uses a repo dataset and a model dataset with internet OFF and versioned kernel metadata. Each candidate gets its own repo dataset created via `kagglehub.dataset_upload()`.

## Architecture risks

- Kaggle official pages may require Kaggle API or authenticated access for full details.
- Training and repo-update notebooks still depend on pushed repository code being available on `main`.
- A2a DWT inference is now packaged: `rogii-wheels-a2a-dwt` (pywavelets) + `rogii-models-a2a-dwt` + kernel `00-rogii-inference-a2a-dwt`. Kaggle base env also has pywavelets 1.9.0 pre-installed.
- TCN training is heavier than LightGBM and needs GPU/runtime checks before any Kaggle candidate promotion.
- `.opencode/` guard-hook files are referenced by docs; if they remain untracked, confirm whether they should be committed before relying on them in shared workflows.

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

- Kaggle Evaluation page wording still needs cross-check; metric is confirmed from the task deck.
- A5 TCN Phase 2 still needs full/screening training to decide whether it should continue toward promotion.
