# Data Map

## Purpose

Document competition files, schemas, target, IDs, missing values and data risks after actual data inspection.

## Owns

File inventory, row counts, column names, dtypes, missing values, unique counts, target confirmation, submission schema, train/test overlap checks, group ID candidates and leakage candidates.

## Update when

- Kaggle data is downloaded or mounted.
- Data inventory script output changes.
- Target, ID or submission schema is confirmed.
- Leakage risks are discovered.

## Do not store here

- Raw data rows.
- Large tables or copied datasets.
- Model metrics.
- Validation rationale beyond data facts.

## Current content

Data has been lightly inspected from local files under `data/`. Full inventory is still pending implementation.

Observed local structure:

| Path | Observed content |
|---|---|
| `data/train/` | Train well files, including `*_horizontal_well.csv`, `*_typewell.csv` and `.png` files |
| `data/test/` | Test well files, including `*_horizontal_well.csv` and `*_typewell.csv` |
| `data/sample_submission.csv` | Submission template |
| `data/AI_wellbore_geology_prediction_task_en.pptx` | Task presentation deck |

Observed counts from quick inspection:

| Item | Count |
|---|---:|
| Total CSV files | 1553 |
| Train CSV files | 1546 |
| Test CSV files | 6 |
| Sample submission rows | 14151 |

Observed sample submission columns:

| Column | Status |
|---|---|
| `id` | Submission ID column observed |
| `tvt` | Prediction column observed |

Observed example train horizontal columns from `data/train/000d7d20__horizontal_well.csv`:

`MD`, `X`, `Y`, `Z`, `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`, `TVT`, `GR`, `TVT_input`

Observed example test horizontal columns from `data/test/000d7d20__horizontal_well.csv`:

`MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`

Observed example train typewell columns from `data/train/000d7d20__typewell.csv`:

`TVT`, `GR`, `Geology`

Observed example test typewell columns from `data/test/000d7d20__typewell.csv`:

`TVT`, `GR`

Preliminary data contract notes:

- Target/prediction column appears to be `tvt` in submission and `TVT` in train horizontal files, but the official metric and target definition still need confirmation.
- File prefix such as `000d7d20` is a likely well/group identifier candidate.
- `TVT_input` appears in both train and test horizontal files and must be leakage-audited before feature use.

Expected first inventory checks:

- File list and sizes.
- Row counts.
- Column names and dtypes.
- Missing values.
- Unique counts for likely IDs.
- Target column confirmation.
- Submission schema from `sample_submission.csv`.
- Train/test overlap checks.
- Group/well identifier candidates.
- Depth/order column candidates.
- Possible leakage columns.

## Open questions

- What is the official metric?
- Is `TVT_input` a permitted input feature or a leakage-adjacent column?
- Which column should be used for group-aware validation?
