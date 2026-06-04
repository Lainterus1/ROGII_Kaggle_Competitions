# ROGII Wellbore Geology Prediction — Project Intake Dossier

## 0. Purpose of this file

This file is the project-specific input package for an agent that will bootstrap and develop a Kaggle competition project for:

`ROGII - Wellbore Geology Prediction`

The user’s goal is to build the best possible reproducible baseline.

This file must be used before executing project setup steps. The future agent should treat it as the primary project context and substitute these details into the sequential step prompts:

```text
steps/01_AGENT_PROMPT.md
steps/02_AGENT_PROMPT.md
steps/03_AGENT_PROMPT.md
steps/04_AGENT_PROMPT.md
steps/05_AGENT_PROMPT.md
steps/06_AGENT_PROMPT.md
steps/07_AGENT_PROMPT.md
steps/08_AGENT_PROMPT.md
steps/09_AGENT_PROMPT.md
steps/10_AGENT_PROMPT.md
steps/11_AGENT_PROMPT.md
```

The agent must not start real implementation until the user explicitly asks it to execute the steps.

---

# 1. Project identity

## Project name

`rogii-wellbore-baseline`

Alternative names:

- `rogii-baseline`
- `kaggle-rogii-wellbore`
- `rogii-wellbore-geology-prediction`

## One-sentence summary

Build a reproducible, well-documented, locally developed and Kaggle-executed ML baseline for predicting TVT / geological position along horizontal wellbores in the ROGII Kaggle competition.

## Project type

- ML/DS project
- Kaggle competition project
- tabular regression project
- spatial/sequential data project
- research prototype
- agentic development workflow project

## Current stage

Initial bootstrap. No existing repository and no downloaded data are assumed.

## Primary objective

Build the strongest possible baseline that can become the reference point for future experiments.

“Best baseline” means:

- reproducible;
- easy to rerun locally and on Kaggle;
- produces valid Kaggle submissions;
- has a trustworthy local validation strategy;
- logs experiments with MLflow;
- has clear documentation;
- avoids data leakage;
- is simple enough to debug;
- is strong enough to become a meaningful starting point for leaderboard improvement.

---

# 2. Confirmed user decisions

## Development workflow

Confirmed:

```text
Local development -> GitHub repository -> Kaggle execution/tests/submissions
```

Meaning:

- local machine is the main development environment;
- GitHub is the source of truth for code, configs and documentation;
- Kaggle is used as a remote execution environment for full-data runs, validation on full competition data and submission generation;
- Kaggle notebooks should be thin runners, not the place where core logic lives.

## Repository status

Confirmed:

```text
No existing repository.
No existing project structure.
No downloaded data assumed.
```

The agent should initialize a new clean repository.

## Agent/tool

Confirmed:

```text
OpenCode
```

The project must include agent-facing instructions, likely via `AGENTS.md` and optional `.agents/skills/` files.

## Public Kaggle notebooks

Confirmed:

```text
Allowed as references.
```

Rules:

- public notebooks may be used for orientation, feature ideas, validation hints and comparison;
- do not blindly copy code;
- if an idea comes from a public notebook, document it in `docs/DECISIONS.md` or `docs/EXPERIMENT_LOG.md`;
- do not overfit to public leaderboard by copying unexplained notebooks.

## Experiment tracking

Confirmed:

```text
MLflow is required.
```

Minimum requirement:

- log parameters;
- log metrics;
- log feature set name;
- log model type;
- log fold scores;
- log public LB score manually after submission;
- log artifacts such as submission file, feature importance and config snapshot.

Markdown experiment log is still useful as a human-readable layer, but MLflow is part of the baseline architecture from the start.

---

# 3. Domain context

## Competition domain

The competition is about wellbore geology prediction. The expected task is to predict TVT / geological position along horizontal wells.

The problem is related to geosteering: when drilling horizontal wells, engineers need to understand where the wellbore is relative to the target geological layer. The model should help infer the geological position from historical well data, measurements and spatial/depth context.

## Important domain terms

### Wellbore

The physical drilled path of a well. In ML terms, a wellbore is likely one high-level grouped object that contains many depth-indexed measurement rows.

### Horizontal well

A well that starts vertically or near-vertically and then turns horizontally through a geological formation. The horizontal section is usually the important part for production and prediction.

### TVT

True Vertical Thickness or a related target describing vertical position/thickness relative to geological layers. The agent must verify exact target definition from Kaggle’s official Evaluation/Data pages and actual files.

### Typewell / reference well

A reference well with known geological interpretation. In this competition, typewell data may be used to infer the geology around horizontal wells. The agent must inspect the actual data files before assuming the schema.

### Gamma ray / GR

A common well log measurement. It is often useful for identifying lithology and geological layer changes.

### MD / measured depth

Distance measured along the wellbore trajectory.

### TVD / true vertical depth

Vertical depth from surface/reference point to a point in the well.

---

# 4. Business/product context

## Goal

Build a strong baseline for Kaggle competition work.

This is not a production software project. The product is a reproducible competition pipeline.

## Primary user

The project owner / Kaggle participant.

## Secondary user

Future coding agents working in OpenCode.

## Core user scenarios

1. User starts a new repository from this intake dossier.
2. Agent creates a clean Kaggle project structure.
3. Agent downloads or connects Kaggle competition data.
4. Agent creates a data inventory and schema map.
5. Agent confirms target, ID column, submission format and evaluation metric.
6. Agent builds a valid naive submission.
7. Agent builds local validation.
8. Agent trains the first classical ML baseline.
9. Agent tracks the run in MLflow.
10. Agent generates a valid Kaggle submission.
11. Agent records local CV, public LB score, assumptions and next experiments.

## Non-goals

- Do not build a production drilling/geosteering service.
- Do not create a web API.
- Do not create a UI.
- Do not optimize infrastructure before the first baseline works.
- Do not start with neural networks.
- Do not use public leaderboard as the only source of truth.
- Do not rely on copied public notebooks without understanding.
- Do not commit raw Kaggle data, secrets, trained models or large artifacts to GitHub.

---

# 5. Success criteria

## Product success criteria

- The repo is initialized and understandable.
- Data is mapped and documented.
- A valid `submission.csv` can be generated.
- At least one naive baseline exists.
- At least one model baseline exists.
- MLflow logs the experiment.
- Local validation score is produced.
- Public leaderboard score can be recorded after Kaggle submission.
- The user can explain what the baseline does.

## Technical success criteria

- Pipeline can run locally on a small sample.
- Pipeline can run on Kaggle on full data.
- Kaggle runner notebook/script uses GitHub code.
- No core training logic lives only inside a notebook.
- Submission schema matches `sample_submission.csv` exactly.
- IDs are in correct order.
- Predictions are finite.
- Validation split is group-aware if well IDs exist.
- Feature columns are leakage-audited.
- Experiment configs are saved and logged.

## Quality success criteria

- Code is modular but not over-engineered.
- Documentation is precise and task-specific.
- Decisions are logged.
- Assumptions are explicit.
- The baseline is reproducible.
- Future experiments can branch from the baseline cleanly.

---

# 6. Functional scope

## Must have

- Repository skeleton.
- `README.md`.
- `AGENTS.md`.
- Project docs in `docs/`.
- Kaggle thin runner notebook or script.
- Data inventory script.
- Data schema summary.
- Submission validator.
- Naive baseline.
- First ML baseline.
- Validation strategy.
- MLflow tracking.
- Experiment log.
- Git ignore rules for data/secrets/artifacts.

## Should have

- Feature importance report.
- Fold-level metrics.
- Well-level error analysis.
- Public notebook reference log.
- Config-driven training.
- Small-sample smoke tests.
- Clear Kaggle/local path handling.

## Could have later

- LightGBM + CatBoost ensemble.
- Rolling depth features.
- Lag features.
- Spatial nearest-neighbor features.
- Typewell alignment features.
- Postprocessing / smoothing.
- Sequence model.
- Polars optimization.
- Automated Kaggle API submission.

## Explicitly out of scope for first baseline

- Complex deep learning.
- Heavy GPU training.
- Large ensembling.
- Production deployment.
- API service.
- Frontend.
- Full MLOps stack beyond MLflow tracking.

---

# 7. Existing assets

## Existing code

None.

## Existing repository

None.

## Existing data

None assumed locally.

The agent must support both:

- local data directory after download;
- Kaggle input directory inside Kaggle Notebook.

## Existing documentation

A bootstrap master prompt exists and defines how the agent should collect project context and execute `steps/01_AGENT_PROMPT.md` through `steps/11_AGENT_PROMPT.md` only after explicit user confirmation.

---

# 8. Technology preferences

## Language

Python.

## Core libraries

Recommended initial stack:

```text
pandas
numpy
scikit-learn
lightgbm
catboost
xgboost
matplotlib
pyyaml
mlflow
pytest
```

Optional later:

```text
polars
pyarrow
shap
optuna
```

## Models

Initial order:

1. naive baseline;
2. simple rule/interpolation baseline if data structure allows;
3. LightGBM baseline;
4. CatBoost baseline;
5. XGBoost baseline;
6. feature-enhanced tree baseline;
7. ensemble later.

## Experiment tracking

MLflow required from the start.

Minimum run metadata:

- run name;
- git commit hash;
- model type;
- feature set name;
- config path;
- data version/path;
- fold scores;
- mean CV score;
- public LB score after submission;
- runtime;
- notes;
- artifacts.

## Development tools

- Local IDE/editor.
- OpenCode agent.
- GitHub.
- Kaggle Notebooks / Kaggle API.
- MLflow UI locally.

---

# 9. Recommended repository structure

```text
rogii-wellbore-baseline/
  README.md
  AGENTS.md
  requirements.txt
  pyproject.toml                 # optional, if project packaging is useful
  .gitignore
  .env.example

  configs/
    baseline_naive.yaml
    baseline_lgbm.yaml
    paths.local.yaml.example
    paths.kaggle.yaml

  src/
    rogii/
      __init__.py
      config.py
      paths.py
      data_loading.py
      data_inventory.py
      schema.py
      features.py
      validation.py
      metrics.py
      models.py
      train.py
      predict.py
      submission.py
      mlflow_utils.py
      utils.py

  scripts/
    make_data_inventory.py
    run_smoke.py
    run_naive_baseline.py
    run_train.py
    run_predict.py
    validate_submission.py
    kaggle_runner.py

  notebooks/
    00_kaggle_thin_runner.ipynb
    01_data_map_eda.ipynb

  tests/
    test_submission_contract.py
    test_validation_split.py
    test_metrics.py
    test_no_target_leakage.py
    test_smoke_pipeline.py

  docs/
    PROJECT_CONTEXT.md
    CONTEXT_MAP.md
    DATA_MAP.md
    METRICS.md
    VALIDATION_STRATEGY.md
    BASELINE_PLAN.md
    EXPERIMENT_LOG.md
    PUBLIC_NOTEBOOK_REFERENCES.md
    DECISIONS.md
    TASKS.md
    KNOWN_ISSUES.md
    CHANGELOG.md

  outputs/                         # gitignored
  data/                            # gitignored
  models/                          # gitignored
  mlruns/                          # usually gitignored unless explicitly needed
  submissions/                     # gitignored
```

---

# 10. GitHub policy

## GitHub should contain

- source code;
- configs without secrets;
- tests;
- docs;
- lightweight notebooks;
- experiment summaries;
- runner scripts;
- `.env.example`.

## GitHub must not contain

- raw Kaggle competition data;
- `kaggle.json`;
- `.env`;
- trained models;
- large parquet/csv artifacts;
- MLflow local artifacts unless intentionally exported in small form;
- leaderboard probing outputs;
- private tokens.

## Required `.gitignore`

```gitignore
data/
outputs/
models/
submissions/
mlruns/
*.csv
*.parquet
*.feather
*.pkl
*.joblib
*.model
*.cbm
*.lgb
*.xgb
kaggle.json
.env
__pycache__/
.ipynb_checkpoints/
.pytest_cache/
```

Exception:

- tiny sample files may be committed only if they are synthetic or explicitly allowed.

---

# 11. Kaggle execution strategy

## Principle

Kaggle is a remote executor, not the source of truth.

Core training logic must live in `src/` and `scripts/`.

The Kaggle notebook should only:

1. clone/pull the GitHub repo;
2. install requirements;
3. point the code to `/kaggle/input/...`;
4. run the selected script;
5. save outputs to `/kaggle/working`;
6. optionally upload or expose `submission.csv`.

## Kaggle runner pattern

The runner should look conceptually like:

```python
!git clone <repo-url>
%cd rogii-wellbore-baseline
!pip install -r requirements.txt
!python scripts/run_train.py --config configs/baseline_lgbm.yaml --env kaggle
!python scripts/validate_submission.py --submission /kaggle/working/submission.csv
```

The actual repo URL must be filled after GitHub repository creation.

## Kaggle data paths

Use configuration:

```yaml
data_dir: /kaggle/input/rogii-wellbore-geology-prediction
output_dir: /kaggle/working
```

Local config example:

```yaml
data_dir: ./data/raw/rogii-wellbore-geology-prediction
output_dir: ./outputs
```

Do not hardcode paths inside model logic.

---

# 12. MLflow strategy

## Why MLflow is required

The user explicitly requested MLflow.

The purpose is to keep a reliable experiment history while iterating on baselines and feature sets.

## Minimum MLflow setup

For each run, log:

- params:
  - model type;
  - model hyperparameters;
  - feature set;
  - validation split;
  - number of folds;
  - seed;
  - data path;
  - config name;
- metrics:
  - fold scores;
  - mean CV score;
  - std CV score;
  - naive baseline score;
  - public LB score if available;
- artifacts:
  - config snapshot;
  - submission file;
  - feature importance;
  - OOF predictions if not too large;
  - validation report;
  - data inventory summary;
- tags:
  - git commit;
  - environment: `local` or `kaggle`;
  - competition name;
  - run type: `naive`, `baseline`, `feature_test`, `submission`.

## Local vs Kaggle MLflow

Local:

- use local `mlruns/` directory;
- open MLflow UI locally.

Kaggle:

- log to `/kaggle/working/mlruns` if needed;
- export selected artifacts;
- do not assume persistent MLflow storage unless explicitly configured.

Optional later:

- remote MLflow tracking server.

Not needed for first baseline unless the user specifically sets one up.

---

# 13. Data context

## Data source

Kaggle competition data.

The agent must inspect actual files before assuming exact schema.

## Expected entities

Likely entities:

- horizontal wells;
- typewells/reference wells;
- depth-indexed points;
- target TVT;
- measured logs such as gamma ray;
- sample submission IDs.

## Required first data inspection

The agent must produce `docs/DATA_MAP.md` containing:

- file list;
- file sizes;
- row counts;
- column names;
- dtypes;
- missing values;
- unique counts for likely IDs;
- target column confirmation;
- submission schema;
- train/test overlap checks;
- group/well identifier candidates;
- depth/order column candidates;
- possible leakage columns.

## Data risks

- Millions of rows may require careful memory handling.
- Row-level random split may leak well-level information.
- Some columns may be target-derived.
- Typewell information may require special alignment.
- Test data may contain intervals different from train data.
- Public LB may be misleading.
- The exact official metric must be confirmed.

---

# 14. Validation strategy

## Default recommendation

Start with group-aware validation.

If a well identifier exists, use:

```text
GroupKFold by well_id / horizontal well id
```

Do not use random KFold as the primary CV unless the agent proves rows are independent.

## Required checks

- no well/group appears in both train and validation fold;
- fold target distribution is reasonable;
- fold well counts are reasonable;
- fold row counts are reasonable;
- metric is computed the same way as Kaggle, or the difference is documented;
- local CV and public LB relationship is recorded.

## Validation documents

Create/update:

- `docs/VALIDATION_STRATEGY.md`;
- `docs/METRICS.md`;
- `docs/KNOWN_ISSUES.md` if there are validation doubts.

---

# 15. Baseline strategy

## Stage 1 — valid submission baseline

Goal:

- produce first valid `submission.csv`;
- verify schema and Kaggle submission contract.

Possible methods:

- constant prediction;
- mean target prediction;
- median target prediction;
- simple per-group fallback if possible.

Do not overvalue this score. This stage exists to test the pipeline.

## Stage 2 — naive local baseline

Goal:

- establish a lower bound for local validation.

Possible methods:

- global mean/median;
- per-well known-zone extrapolation if data allows;
- simple interpolation/extrapolation by depth if valid.

## Stage 3 — first ML baseline

Goal:

- train a simple model using safe features.

Recommended models:

- LightGBM first;
- CatBoost second;
- XGBoost third.

Safe initial features:

- numeric measurements available in both train and test;
- depth/coordinate features;
- no target-like columns;
- no post-target leakage;
- categorical IDs only if validation design allows them and leakage risk is considered.

## Stage 4 — stronger baseline

Add:

- depth-normalized features;
- per-well aggregates;
- rolling features;
- lag features;
- local slopes;
- GR gradients;
- typewell alignment features;
- distance to typewell/reference features;
- model ensemble.

---

# 16. Testing strategy

## Local tests

The local environment should run:

```bash
pytest tests/
python scripts/run_smoke.py --config configs/baseline_lgbm.yaml
python scripts/validate_submission.py --submission outputs/submission.csv
```

Local tests should use:

- tiny sample;
- synthetic minimal sample;
- or first N rows if competition data is available locally.

## Kaggle tests

Kaggle should run:

```bash
python scripts/run_train.py --config configs/baseline_lgbm.yaml --env kaggle
python scripts/validate_submission.py --submission /kaggle/working/submission.csv
```

## Submission validation checks

- row count equals sample submission;
- id column exactly matches sample submission;
- prediction column exists;
- predictions are numeric;
- predictions are finite;
- no NaN;
- no inf;
- output file name is correct;
- target scale seems plausible.

## Leakage tests

- ensure target column is not in features;
- ensure target-derived columns are excluded;
- ensure group split has no overlap;
- ensure train/test feature columns match.

---

# 17. Documentation strategy

## Required docs

### `docs/PROJECT_CONTEXT.md`

Human-readable overview of the competition, goal, workflow and project rules.

### `docs/CONTEXT_MAP.md`

Where everything lives and what future agents should read first.

### `docs/DATA_MAP.md`

Schema, files, columns, target, IDs, missing values and data risks.

### `docs/METRICS.md`

Official Kaggle metric and local implementation.

### `docs/VALIDATION_STRATEGY.md`

How validation is designed and why.

### `docs/BASELINE_PLAN.md`

Baseline stages and acceptance criteria.

### `docs/EXPERIMENT_LOG.md`

Human-readable run history complementary to MLflow.

### `docs/PUBLIC_NOTEBOOK_REFERENCES.md`

Public notebooks reviewed, ideas extracted, and whether used.

### `docs/DECISIONS.md`

Important decisions and rationale.

### `docs/TASKS.md`

Current backlog.

### `docs/KNOWN_ISSUES.md`

Known problems, risks and unresolved questions.

### `docs/CHANGELOG.md`

Chronological project changes.

---

# 18. Agent workflow rules

## Core agent rules

The agent must:

- use this dossier as the initial source of truth;
- create a new repository structure if none exists;
- create docs before complex code;
- keep Kaggle notebook as a thin runner;
- put reusable logic in `src/` and `scripts/`;
- use GitHub as source of truth;
- keep raw data and artifacts out of Git;
- add MLflow tracking from the first model baseline;
- ask before Kaggle submission unless user gives blanket permission;
- ask before adding heavy dependencies;
- ask before using paid compute;
- document every major assumption;
- update docs after architecture, validation or baseline changes.

## Forbidden agent actions

The agent must not:

- commit raw data;
- commit secrets;
- delete raw data;
- overwrite previous submissions without saving/renaming;
- build complex neural architecture before baseline;
- use public notebooks as black-box copied code;
- optimize only for public leaderboard;
- perform broad refactoring without justification;
- invent schema details before inspecting files;
- silently change validation strategy;
- run steps out of order unless explicitly instructed.

## Completion report format

After each major action, report:

```md
## Step completed

### Step
[step or task name]

### Created/updated files
- ...

### Commands run
- ...

### Decisions made
- ...

### Assumptions used
- ...

### Validation/tests
- ...

### Open questions
- ...

### Ready for next step?
Yes / No
```

---

# 19. First vertical slice

## Objective

Create a clean end-to-end baseline path:

```text
GitHub repo -> local smoke test -> Kaggle full run -> MLflow logged baseline -> valid submission.csv
```

## End-to-end steps

1. Initialize repository.
2. Add `.gitignore`, `README.md`, `AGENTS.md`, docs skeleton.
3. Add dependency files.
4. Add Kaggle/local path configs.
5. Add data inventory script.
6. Add submission validator.
7. Add naive baseline script.
8. Add MLflow utility wrapper.
9. Add first model baseline script.
10. Add Kaggle thin runner notebook/script.
11. Run local smoke test.
12. Run Kaggle full test.
13. Save `submission.csv`.
14. Log run in MLflow.
15. Record results in `docs/EXPERIMENT_LOG.md`.

## Acceptance criteria

- repo can be cloned;
- dependencies install;
- docs exist;
- data paths are configurable;
- data inventory runs;
- submission validator works;
- naive baseline creates valid submission;
- model baseline creates valid submission;
- MLflow logs the model run;
- Kaggle runner works;
- first LB score can be recorded.

---

# 20. Roadmap

## Bootstrap roadmap

1. Create repository structure.
2. Create source-of-truth docs.
3. Create agent instructions.
4. Create configs and path handling.
5. Create data inventory tool.
6. Create submission validator.
7. Create MLflow tracking utilities.
8. Create Kaggle thin runner.

## Baseline roadmap

1. Inspect data files.
2. Confirm target and metric.
3. Confirm submission schema.
4. Confirm well/group identifier.
5. Build naive baseline.
6. Build validation strategy.
7. Build LightGBM baseline.
8. Log experiment in MLflow.
9. Run on Kaggle.
10. Submit with user approval.
11. Record CV/LB gap.

## Improvement roadmap

1. Feature importance analysis.
2. Better validation diagnostics.
3. Depth-based features.
4. Rolling and lag features.
5. Typewell/reference features.
6. Spatial features.
7. CatBoost/XGBoost comparison.
8. Ensemble.
9. Postprocessing.
10. Private leaderboard robustness strategy.

---

# 21. Open questions

No blocking questions remain for project bootstrap.

Non-blocking questions that can be answered later:

1. What is the GitHub repository URL after creation?
2. Will Kaggle submissions always require explicit user approval, or can the agent submit automatically after validation?
3. Should MLflow artifacts be stored only locally/Kaggle, or should a remote MLflow tracking server be configured later?
4. What is the target time budget for the first strong baseline?
5. What are the local machine constraints: RAM, CPU, disk, OS?

Default recommendations until answered:

- use explicit user approval for Kaggle submissions;
- use local/Kaggle MLflow storage first;
- target first strong baseline within 1–3 days;
- optimize for CPU-friendly training first.

---

# 22. Readiness checklist

## Project understanding

- [x] Project goal is clear.
- [x] Primary user is clear.
- [x] First useful version is clear.
- [x] Non-goals are clear.

## Technical direction

- [x] Project type is classified.
- [x] Stack preferences are known.
- [x] Main workflow is known.
- [x] Architecture risks are listed.

## Contracts

- [ ] Data contracts must be confirmed after data inspection.
- [ ] Submission contract must be confirmed from `sample_submission.csv`.
- [ ] Official metric must be confirmed from Kaggle Evaluation page.
- [x] ML/evaluation contract is planned.
- [x] Security constraints are planned.

## Agent workflow

- [x] Source-of-truth documents are planned.
- [x] Context map strategy is planned.
- [x] `AGENTS.md` requirements are clear.
- [x] Required skills are listed.
- [x] Documentation update policy is defined.
- [x] Forbidden agent actions are defined.

## Execution readiness

- [x] First vertical slice is proposed.
- [x] Bootstrap roadmap is ready.
- [x] Open questions are listed.
- [x] Assumptions are clearly marked.

---

# 23. Recommended execution mode

Recommended mode:

```text
balanced
```

Meaning:

- agent can use reasonable defaults for repository structure, configs, docs and scripts;
- agent must ask before changing validation strategy;
- agent must ask before adding heavy dependencies;
- agent must ask before using paid resources;
- agent must ask before Kaggle submission unless explicitly authorized;
- agent must mark assumptions clearly.

---

# 24. First command to give the future agent

Use this when ready:

```text
Use ROGII_PROJECT_INTAKE_DOSSIER.md as the project intake dossier.
Start in balanced mode.
Do not implement model complexity yet.
Initialize the repository and begin with steps/01_AGENT_PROMPT.md.
After each step, stop and report created/updated files, decisions, assumptions, tests and open questions.
```
