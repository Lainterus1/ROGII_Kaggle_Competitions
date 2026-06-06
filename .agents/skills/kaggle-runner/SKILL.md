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
- Kaggle input root: `/kaggle/input`.
- Kaggle output root: `/kaggle/working`.
- Runner files under `scripts/` and `notebooks/`.
- Candidate train/predict/validate commands.
- Current code-competition submit path: validated kernel-version submit, not direct file upload.

## Source-of-truth files

- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_MAP.md`
- `docs/DECISIONS.md`
- `configs/paths.kaggle.yaml`
- `src/rogii/kaggle_runtime.py`
- `scripts/kaggle_offline_inference.py`
- `scripts/kaggle_runner.py`
- `notebooks/00_kaggle_inference.ipynb`
- `notebooks/01_kaggle_train.ipynb`
- `notebooks/02_kaggle_update_repo.ipynb`
- `notebooks/kernel-metadata.json`
- `AGENTS.md`

## Resource mapping

All procedures use placeholders. The actual values are defined here — a single point to update when names change:

| Placeholder          | Current value                              |
|----------------------|--------------------------------------------|
| `<inference-kernel>` | `daniilgonchar/00-rogii-inference-r1`      |
| `<repo-dataset>`     | `daniilgonchar/rogii-repo-v2`              |
| `<model-dataset>`    | `daniilgonchar/rogii-models-v2`            |
| `<model-file>`       | `baseline_lgbm.pkl`                         |
| `<competition>`      | `rogii-wellbore-geology-prediction`         |
| `<kernel-meta-dir>`  | `notebooks`                                 |

To change a dataset or kernel name, update this table and the corresponding
`kernel-metadata.json`. All commands below use only the placeholders.

## Offline submit flow (ADR-013)

Works for any kernel — R1 fallback or new stage.

1. Make code changes locally. Commit and push to GitHub.
2. If code changes affect `<repo-dataset>`: update it
   (see "How to update `<repo-dataset>`" below). Otherwise skip.
3. Update the inference notebook with needed CLI flags.
4. Run `python -m pytest tests`.
5. Push kernel: `kaggle kernels push -p <kernel-meta-dir>`.
6. Wait ~15 seconds for kernel to auto-run, then download output:
   `kaggle kernels output <inference-kernel> -p <temp_dir> --file-pattern submission.csv -o`.
7. Validate: `python scripts/validate_submission.py --data-dir data --submission <temp_dir>/submission.csv`.
8. After explicit user approval, submit:
   ```
   kaggle competitions submit -c <competition> -k <inference-kernel> \
     -v <version> -f submission.csv -m "<message>"
   ```
9. Record LB score in `docs/EXPERIMENT_LOG.md` only after available.

## How to update `<repo-dataset>`

The repo dataset contains the full repository as individual files with
directory structure (e.g. `scripts/run_predict.py`, `src/rogii/smoothing.py`).

**This structure can ONLY be created or updated through Kaggle notebook
output.** There is no CLI-only way. The Kaggle API (`kaggle datasets version`)
cannot preserve directory structure — it either skips subdirectories or
converts them to archives. Both result in a broken dataset.

### Correct way

1. Push code changes to GitHub.
2. Open `02_kaggle_update_repo.ipynb` on Kaggle (internet ON).
3. Run it → clones latest code from GitHub.
4. Save notebook output as a new version of `<repo-dataset>`:
   Kaggle UI → **Save Version** → **Create Dataset** → select existing `<repo-dataset>`.

### When to update

Only when `src/`, `scripts/`, `configs/` or `requirements.txt` have changed
AND the change is needed at inference time (new module, new CLI flag,
new dependency). If only the notebook cell changed, no dataset update needed —
the notebook is pushed via `kaggle kernels push` separately.

## Stable contract — do NOT modify

These parts of the inference pipeline are fixed and proven. Changing them
breaks the submit flow:

| Component                     | What                                                                          | Why stable                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Notebook repo discovery       | `find_repo_root()` — searches for `scripts/run_predict.py` + `src/rogii/`    | Works across all dataset layouts. Do NOT rewrite as zip-extraction, git-clone, or hardcoded paths. |
| `kernel-metadata.json`        | `dataset_sources: ["<repo-dataset>", "<model-dataset>"]`                      | Points to the stable repo and model. Do NOT change slugs unless the dataset itself was rebuilt. |
| Notebook command builder      | `cmd = [..., '--data-dir', ..., '--model', ..., '--output', ...]`             | The ONLY place to add CLI flags (e.g. `--savgol-smooth`). Do NOT rewrite the command logic.     |
| Model file name               | `<model-file>` inside `<model-dataset>`                                        | Inference script and notebook both reference this name.                                         |

## New stage = new kernel

Every new experiment (feature change, post-processing change, model change)
gets its own Kaggle kernel. Do NOT overwrite an existing kernel even if
"only the CLI flags changed."

**Why:**
- Fallback kernel (e.g. R1) remains working for re-submission.
- Kernel version history stays clean — one purpose per kernel.
- Debugging is straightforward — logs show exactly what ran.

**How:**
1. Copy `<kernel-meta-dir>/` to `<kernel-meta-dir>/kernels/<stage-slug>/`.
2. Update `kernel-metadata.json` in the new directory:
   - `id`: `<owner>/00-rogii-inference-<stage-slug>`
   - `title`: `00 - ROGII Inference (<stage-slug>)`
   - `dataset_sources`: keep `<repo-dataset>` and `<model-dataset>` unchanged.
3. Edit the notebook: add/change CLI flags as needed. Do NOT change the
   `find_repo_root()` logic or the command builder structure.
4. Push: `kaggle kernels push -p <kernel-meta-dir>/kernels/<stage-slug>`.
5. Follow the standard offline submit flow (validate → submit with `-k <new-kernel>`).

Example: PrP3 (Savgol post-processing) → kernel `00-rogii-inference-prp3`.

## Candidate build workflow

Use this flow for any new model or non-trivial code change. Do not overwrite
fallback kernel or model dataset unless explicitly asked.

| Artifact | Rule |
|---|---|
| Repo dataset | If `src/`, `scripts/`, `configs/` or `requirements.txt` changed AND needed at inference: update `<repo-dataset>` via the 02 notebook (Kaggle, internet ON). |
| Model dataset | Use a candidate-specific dataset, e.g. `<owner>/rogii-models-<stage-slug>`, containing the trained model file. |
| Dependency dataset | If inference imports packages not available offline, create a candidate-specific wheels dataset, e.g. `<owner>/rogii-wheels-<stage-slug>`. |
| Kernel metadata | Use candidate-specific kernel metadata and slug (see "New stage = new kernel" above). |
| Submit command | Standard kernel-version submit with `-k`, `-v` and `-f submission.csv`. |

Candidate steps:

1. Confirm the candidate feature/model flags and whether inference needs extra packages.
2. Update `<repo-dataset>` if code changed (see "How to update `<repo-dataset>`").
3. Train the candidate model and upload it to a candidate-specific model dataset.
4. If extra packages are needed at inference, package wheels locally and upload them to a dependency dataset.
5. Create a candidate kernel folder (see "New stage = new kernel").
6. Add dependency install in the candidate notebook only from attached offline inputs, e.g. `python -m pip install --no-index --find-links <wheels_dir> pywavelets`.
7. Push the candidate kernel: `kaggle kernels push -p <candidate_kernel_dir>`.
8. Check logs for resolved paths, row count and non-empty output.
9. Download `submission.csv` to a temp directory and validate it.
10. Submit only after explicit user approval.
11. Record the public LB score only after available.

## Why not direct file submit

Direct `kaggle competitions submit -f <submission.csv>` can return `400 Bad Request` for this code competition. Use kernel-version submit with `-k`, `-v` and `-f submission.csv`.

## One-time setup (in Kaggle)

Only needed when rebuilding datasets manually:

| Notebook | Purpose | Internet |
|---|---|---|
| `02_kaggle_update_repo` | Clone GitHub repo, then create/update `<repo-dataset>` | ON |
| `01_kaggle_train` | Train stable model and create/update `<model-dataset>` with `<model-file>` | ON |
| Inference kernel | Offline inference and code-competition submit | OFF |

1. Keep Kaggle notebooks thin: clone repo, run scripts.
2. Put reusable logic in `src/rogii/` or `scripts/`, not notebook cells.
3. Write outputs to `/kaggle/working`.
4. Do not require GitHub auth, tokens or Kaggle Secrets for cloning the public repo.
5. Ensure notebook commands match pushed repository files.
6. `src/rogii/kaggle_runtime.py` owns marker-based path discovery; do NOT reintroduce hardcoded paths.
7. `00_kaggle_inference.ipynb` must fail loudly if repo/model/data cannot be found or if `submission.csv` is missing/empty.
8. Submit only after explicit user approval.
9. For new candidates, keep artifact slugs explicit in the completion report: repo dataset, model dataset, dependency dataset, kernel slug and submitted version.

## Documentation updates

- Update `docs/ARCHITECTURE.md` if runtime flow changes.
- Update `docs/DECISIONS.md` for meaningful runner strategy changes.
- Update `docs/TASKS.md` when Kaggle execution readiness changes.
- Update `README.md` if user-facing commands change.
- Update `docs/EXPERIMENT_LOG.md` only after a meaningful run or after the user reports a real public LB score.

## Validation

- Run the relevant local script with local config when possible.
- For offline inference, verify Kaggle logs show `Repo root`, `Model path`, `Data dir`, `Submission rows` and non-empty output bytes.
- Always validate `submission.csv` before any user-approved submit.
- For candidates with offline dependencies, verify logs show successful offline package installation before prediction.
- Run `python -m pytest tests` after code changes.

## Completion checklist

- [ ] Notebooks remain thin.
- [ ] Kaggle paths use `/kaggle/input` and `/kaggle/working`.
- [ ] `notebooks/kernel-metadata.json` points to the intended kernel, datasets and competition source.
- [ ] Inference notebook is offline, loads from `<repo-dataset>` + `<model-dataset>`, no training.
- [ ] Kernel output contains non-empty `submission.csv`.
- [ ] For candidate builds, repo/model/dependency datasets and kernel slug are candidate-specific and do not overwrite the fallback.
- [ ] Extra inference dependencies, if any, are installed only from attached offline inputs.
- [ ] No secrets are required for cloning the public repo.
- [ ] Runners invoke repository scripts.
- [ ] `submission.csv` was validated before any approved submission.

## Forbidden actions

- Do not put core training, feature or validation logic in notebooks.
- Do not write outputs to `/kaggle/input`.
- Do not assume unpublished local code is available on Kaggle.
- Do not implement scheduled or approval-free Kaggle submission.
- Do not submit to Kaggle without explicit user approval.

- Do NOT try to update `<repo-dataset>` via `kaggle datasets version` with
  tar, zip, or flat directories. The CLI cannot preserve directory structure.
  This WILL break the dataset (corrupted/missing files).

- Do NOT create new dataset slugs to work around upload limitations.

- Do NOT change `<repo-dataset>` or `<model-dataset>` references in
  `kernel-metadata.json` unless the dataset itself was intentionally
  rebuilt under a new name.

- Do NOT replace the `find_repo_root()` discovery logic with zip-extraction,
  git-clone, or hardcoded-path approaches.

- Do NOT overwrite an existing inference kernel for a new experiment.
  Create a new kernel (see "New stage = new kernel").
