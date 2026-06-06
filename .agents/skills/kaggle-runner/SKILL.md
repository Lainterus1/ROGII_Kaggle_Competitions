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
- Current stable offline inference notebook uses `rogii-repo-v2` + `rogii-models-v2`; inspect metadata before changing inputs.
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

## Procedure (metadata-driven offline inference, ADR-013)

The stable recovery path is R1 offline inference. `00` is updated through Kaggle CLI metadata and uses mounted `rogii-repo-v2` + `rogii-models-v2` Datasets with internet OFF. `01` and `02` remain optional helpers for rebuilding model/repo datasets.

### Current fixed inputs

| Resource | Value |
|---|---|
| Inference kernel | `daniilgonchar/00-rogii-inference-r1` |
| Kernel metadata | `notebooks/kernel-metadata.json` |
| Repo dataset | `daniilgonchar/rogii-repo-v2` |
| Model dataset | `daniilgonchar/rogii-models-v2` |
| Model file | `baseline_lgbm.pkl` |
| Competition source | `rogii-wellbore-geology-prediction` |
| Internet | OFF for `00` |

### R1 offline submit flow

1. Make code changes locally.
2. Run `python -m pytest tests`.
3. Push/update the inference kernel: `kaggle kernels push -p notebooks`.
4. Check logs: `kaggle kernels logs daniilgonchar/00-rogii-inference-r1`.
5. Download output to a temp directory, not the repo: `kaggle kernels output daniilgonchar/00-rogii-inference-r1 -p <temp_dir> --file-pattern submission.csv -o`.
6. Validate output locally when local sample data is available: `python scripts/validate_submission.py --data-dir data --submission <temp_dir>/submission.csv`.
7. After explicit user approval, submit the kernel version output for this code competition: `kaggle competitions submit -c rogii-wellbore-geology-prediction -k daniilgonchar/00-rogii-inference-r1 -v <version> -f submission.csv -m "<message>"`.
8. Do not wait indefinitely for the score; record it in `docs/EXPERIMENT_LOG.md` only after the user or CLI reports a public LB score.

### Candidate build workflow

Use this flow for A2a or any future model/build. Do not overwrite the R1 recovery kernel or R1 model dataset unless the user explicitly asks.

| Artifact | Rule |
|---|---|
| Repo dataset | If `src/`, `scripts/`, `configs/`, `requirements.txt` or notebooks changed, update `rogii-repo-v2` before candidate submit. |
| Model dataset | Use a candidate-specific dataset, for example `rogii-models-a2a`, containing the trained model file. |
| Dependency dataset | If inference imports packages not available offline, create a candidate-specific wheels/dependency dataset, for example `rogii-wheels-a2a`. |
| Kernel metadata | Use candidate-specific kernel metadata and kernel slug, for example `00-rogii-inference-a2a`; include repo, model, dependency and competition sources. |
| Submit command | Submit the candidate kernel version output with `-k`, `-v` and `-f submission.csv`. |

Candidate steps:

1. Confirm the candidate feature/model flags and whether inference needs extra packages.
2. Update `rogii-repo-v2` if code changed since the dataset was last created.
3. Train the candidate model and upload it to a candidate-specific model dataset.
4. If extra packages are needed at inference, package wheels locally or through an internet-ON helper and upload them to a dependency dataset.
5. Create or update a candidate inference notebook/metadata pair. Keep it thin and call repository scripts; do not move core logic into the notebook.
6. Add dependency installation in the candidate notebook only from attached offline inputs, for example `python -m pip install --no-index --find-links <wheels_dir> pywavelets`.
7. Push the candidate kernel with `kaggle kernels push -p <candidate_kernel_dir>`.
8. Check logs for resolved repo/model/data/dependency paths, row count and non-empty output bytes.
9. Download `submission.csv` to a temp directory and validate it.
10. Submit only after explicit user approval.
11. Record the public LB score only after the user or CLI reports it.

For A2a specifically, the candidate needs `pywavelets` offline at inference time. A2a should use a separate model dataset and either a separate A2a inference kernel or updated candidate metadata that includes the wheels dataset.

### Why not direct file submit

Direct `kaggle competitions submit -f <submission.csv>` can return `400 Bad Request` for this code competition. Use kernel-version submit with `-k`, `-v` and `-f submission.csv`.

### A2a DWT limitation

A2a uses `pywavelets`. Kaggle Submit reruns with internet OFF, so A2a cannot replace R1 in the stable offline submit path until an offline dependency/model packaging solution is created.

### One-time setup (in Kaggle)

Only needed when rebuilding datasets manually:

| Notebook | Purpose | Internet |
|---|---|---|
| `02_kaggle_update_repo` | Clone GitHub repo, then create/update `rogii-repo-v2` Dataset | ON |
| `01_kaggle_train` | Train stable R1 and create/update `rogii-models-v2` with `baseline_lgbm.pkl` | ON |
| `00-rogii-inference-r1` | Offline inference and code-competition submit | OFF |

1. Keep Kaggle notebooks thin: clone repo, run scripts.
2. Put reusable logic in `src/rogii/` or `scripts/`, not notebook cells.
3. Write outputs to `/kaggle/working`.
4. Do not require GitHub auth, tokens or Kaggle Secrets for cloning the public repo.
5. Ensure notebook commands match pushed repository files.
6. `src/rogii/kaggle_runtime.py` owns marker-based path discovery; do not reintroduce hardcoded `/kaggle/input/datasets/*/.../ROGII_Kaggle_Competitions*` paths.
7. `00_kaggle_inference.ipynb` must fail loudly if repo/model/data cannot be found or if `submission.csv` is missing/empty.
8. Submit only after explicit user approval.
9. Document any Kaggle-specific runtime limitation.
10. For new candidates, keep artifact slugs explicit in the completion report: repo dataset, model dataset, dependency dataset, kernel slug and submitted version.

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
- [ ] `00_kaggle_inference.ipynb` is offline, loads from `rogii-repo-v2` + `rogii-models-v2`, no training.
- [ ] Kernel output contains non-empty `submission.csv`.
- [ ] For candidate builds, repo/model/dependency datasets and kernel slug are candidate-specific and do not overwrite R1 fallback.
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
