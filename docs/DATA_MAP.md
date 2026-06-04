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

Data inventory has been implemented and run locally with `python scripts/make_data_inventory.py --data-dir data`.

Observed local structure:

| Path | Observed content |
|---|---|
| `data/train/` | Train well files, including `*_horizontal_well.csv`, `*_typewell.csv` and `.png` files |
| `data/test/` | Test well files, including `*_horizontal_well.csv` and `*_typewell.csv` |
| `data/sample_submission.csv` | Submission template |
| `data/AI_wellbore_geology_prediction_task_en.pptx` | Task presentation deck; confirms prediction task and RMSE metric |

Observed counts from quick inspection:

| Item | Count |
|---|---:|
| Total files | 2327 |
| Total CSV files | 1553 |
| PNG files | 773 |
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

- Target/prediction column is `tvt` in submission and `TVT` in train horizontal files.
- File prefix such as `000d7d20` is the well/group identifier candidate for validation and loading.
- `TVT_input` appears in both train and test horizontal files.
- Task deck states `TVT_input` contains known geology values until Prediction Start (PS); values after PS are missing and must not be used as labels/features.
- Sample submission IDs encode `<well_id>_<row_index>` and point to post-PS rows in the corresponding test horizontal file.

Inventory checks implemented:

- File list and sizes.
- Row counts.
- Column names and dtypes.
- Unique counts for likely IDs.
- Target column confirmation.
- Submission schema from `sample_submission.csv`.
- Group/well identifier candidates.
- Depth/order column candidates.
- Possible leakage columns.

Still pending deeper inventory checks:

- Full missing-value profile by column.
- Train/test overlap diagnostics beyond file-prefix observations.
- Typewell alignment contract details.

## Open questions

- Is `TVT_input` a permitted input feature or a leakage-adjacent column?
- Which column should be used for group-aware validation?
