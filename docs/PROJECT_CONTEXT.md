# Project Context

## One-sentence summary

Build a reproducible, well-documented, locally developed and Kaggle-executed ML baseline for predicting TVT / geological position along horizontal wellbores in the ROGII Kaggle competition.

## Goal

Create the strongest practical baseline for the Kaggle competition `ROGII - Wellbore Geology Prediction`.

The baseline must be reproducible, easy to rerun locally and on Kaggle, capable of generating valid submissions, tracked with MLflow, and simple enough to debug and extend.

The intended workflow is:

1. Develop locally.
2. Store code, configs and documentation in public GitHub repo `ROGII_Kaggle_Competitions`.
3. Use Kaggle as the remote execution environment for full-data runs, validation and submission generation.

Kaggle competition page:

`https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview`

GitHub repository:

`https://github.com/Lainterus1/ROGII_Kaggle_Competitions`

Public clone command:

`git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git`

## Users

- Primary user: the project owner / Kaggle participant.
- Secondary users: future OpenCode agents continuing development.
- Execution users: local development environment, GitHub repository, Kaggle Notebooks or Kaggle API.

## Core scenarios

1. Initialize a clean repository and project structure.
2. Connect or download Kaggle competition data locally or inside Kaggle.
3. Create a data inventory and schema map after actual files are available.
4. Confirm target column, ID column, submission format and official metric from Kaggle and data files.
5. Generate a valid naive `submission.csv`.
6. Build a trustworthy local validation strategy, preferably group-aware by well if well IDs exist.
7. Train the first classical ML baseline using safe, leakage-audited features.
8. Log experiments with MLflow, including params, metrics, fold scores and artifacts.
9. Run the pipeline on Kaggle using a thin runner that calls repository code.
10. Record local CV, public leaderboard score, assumptions and next experiments.

## Success criteria

### Product criteria

- The repository is initialized and understandable.
- Data is mapped and documented after inspection.
- A valid `submission.csv` can be generated.
- At least one naive baseline exists.
- At least one model baseline exists.
- MLflow logs the experiment.
- Local validation score is produced.
- Public leaderboard score can be recorded after manual Kaggle submission.
- The user can explain what the baseline does.

### Technical criteria

- Pipeline can run locally on a small sample.
- Pipeline can run on Kaggle on full competition data.
- Kaggle runner uses GitHub code and keeps notebooks thin.
- Core training logic lives in `src/` and `scripts/`, not only in notebooks.
- Submission schema matches `sample_submission.csv` exactly.
- Submission IDs are in the correct order.
- Predictions are numeric and finite.
- Validation split is group-aware if well IDs exist.
- Feature columns are audited for target leakage.
- Experiment configs are saved and logged.

### Quality criteria

- Code is modular but not over-engineered.
- Documentation is precise and project-specific.
- Important decisions are logged.
- Assumptions and unknowns are explicit.
- Future experiments can branch from the baseline cleanly.

## Constraints

### Business constraints

- This is a Kaggle competition baseline project, not a production drilling or geosteering product.
- Public Kaggle notebooks may be used only as references for orientation, feature ideas and comparison.
- Ideas taken from public notebooks must be documented.
- Public leaderboard score must not be the only source of truth.

### Technical constraints

- Use Python.
- Current stack: pandas, numpy, scikit-learn, LightGBM, matplotlib, PyYAML, MLflow, pytest, scipy, PyWavelets, numba and torch.
- CatBoost/XGBoost remain deferred comparison/ensemble options; do not add or prioritize them ahead of the active A5 TCN path without a new decision.
- MLflow is required for model baseline runs when practical.
- Kaggle notebooks must be thin runners.
- Paths must be configurable for local and Kaggle environments.
- Do not add paid compute or heavy new dependencies without approval.
- Neural sequence modeling is now allowed only as the documented A5 path; do not promote it over R3 without validation and submission checks.

### Data constraints

- Local competition data is expected under ignored `data/` when available.
- Current schema, target, submission columns and group ID are documented in `docs/DATA_MAP.md`.
- Do not invent new data schema details before inspection.
- Use `well_id` from the file prefix as the group identifier for validation unless a documented decision changes this.
- Avoid row-level random split as primary validation because rows from the same well leak trajectory/geology context.
- Do not use target-like or target-derived columns as features. `TVT_input` is allowed only as the known pre-PS anchor/baseline described by the task deck.

### Security and privacy constraints

- Do not commit raw Kaggle data.
- Do not commit `kaggle.json`, `.env`, tokens or secrets.
- Do not commit trained models, large artifacts, raw submissions or local MLflow artifact stores unless explicitly approved.
- Keep `.env.example` lightweight and secret-free.

### Cost and infrastructure constraints

- Use local and Kaggle execution first.
- Local data is expected under `data/` after download.
- Kaggle data is expected under `/kaggle/input` with outputs under `/kaggle/working`.
- Use local `mlruns/` for local MLflow tracking.
- Use `/kaggle/working/mlruns` on Kaggle if needed.
- Do not assume a remote MLflow tracking server unless configured later.
- Kaggle submission is manual and requires explicit user approval.

## Non-goals

- Do not build a production drilling or geosteering service.
- Do not create a web API.
- Do not create a UI.
- Do not optimize infrastructure before the first baseline works.
- Do not start with neural networks.
- Do not rely on copied public notebooks without understanding.
- Do not optimize only for public leaderboard.
- Do not commit raw data, secrets, trained models or large artifacts.

## Domain terms

| Term | Meaning |
|---|---|
| Wellbore | The physical drilled path of a well. Each wellbore contains many depth-indexed measurement rows. |
| Horizontal well | A well that turns horizontally through a geological formation. The horizontal section is the primary prediction target. |
| TVT | True Vertical Thickness or a related target describing vertical position/thickness relative to geological layers. |
| Typewell / reference well | A reference well with known geological interpretation, used to infer geology around horizontal wells. |
| Gamma ray (GR) | A common well log measurement useful for identifying lithology and geological layer changes. |
| MD (measured depth) | Distance measured along the wellbore trajectory. |
| TVD (true vertical depth) | Vertical depth from surface/reference point to a point in the well. |

## Current confirmed state

- Repository structure is implemented: `src/rogii/`, `scripts/`, `configs/`, `tests/`, `docs/`, thin Kaggle runner notebooks and candidate kernel folders.
- GitHub repository is public: `https://github.com/Lainterus1/ROGII_Kaggle_Competitions`.
- Local development remains the primary development environment.
- GitHub is the source of truth for code, configs and documentation.
- Kaggle is used for full-data execution, offline inference and submission generation.
- Submission to Kaggle remains manual and requires explicit user approval.
- Current best baseline is R3: 3-seed LightGBM `[42, 7, 123]` on the R1 18-feature set + Savgol `w=31 p=2`, public LB `12.177`.
- Current active development is A5 TCN architecture diversity. TCN v0, OOF, diagnostics and Phase 2 dual normalization are implemented; the Phase 2 training gate is pending.

## Open questions

- Kaggle Evaluation page wording still needs cross-check when accessible; metric is currently confirmed from the official task deck included in the competition data.
- A5 TCN Phase 2 still needs a full/screening training gate to determine whether dual normalization fixes prediction flattening.
- Local TCN runtime assumptions are based on the available RTX 4050 path and should be rechecked after each material TCN architecture change.
