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
- Initial stack should prioritize pandas, numpy, scikit-learn, LightGBM, CatBoost, XGBoost, matplotlib, PyYAML, MLflow and pytest.
- MLflow is required from the first model baseline.
- Kaggle notebooks must be thin runners.
- Paths must be configurable for local and Kaggle environments.
- Do not add heavy dependencies or paid compute without approval.
- Do not implement complex neural networks before a working baseline exists.

### Data constraints

- No local competition data is assumed at bootstrap time.
- Actual schema, target, IDs, metric and submission contract must be confirmed from Kaggle and the downloaded files.
- Do not invent data schema details before inspection.
- Avoid row-level random split as primary validation if grouped well data exists.
- Do not use target-like or target-derived columns as features.

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

## Initial assumptions

- Current stage is initial bootstrap.
- No existing repository structure is assumed beyond the intake dossier and step prompts.
- GitHub repository is public: `https://github.com/Lainterus1/ROGII_Kaggle_Competitions`.
- Local development is the primary development environment.
- GitHub will be the source of truth for code, configs and documentation.
- Kaggle will be used for full-data execution and submission generation.
- Submission to Kaggle remains manual unless the user changes this rule.
- First strong baseline should be CPU-friendly and understandable.

## Open questions

- What is the official Kaggle evaluation metric?
- What are the actual competition data files and schemas?
- What are the target and ID columns?
- Which column, if any, is the correct well/group identifier for validation?
- What are the local machine constraints for full-data experiments?
