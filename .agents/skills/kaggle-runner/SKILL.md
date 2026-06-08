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
| `<repo-dataset>`     | `daniilgonchar/rogii-repo-<slug>` (candidate-specific) |
| `<model-dataset>`    | `daniilgonchar/rogii-models-<slug>` (candidate-specific) |
| `<model-file>`       | `baseline_lgbm.pkl`                         |
| `<competition>`      | `rogii-wellbore-geology-prediction`         |
| `<kernel-meta-dir>`  | `notebooks`                                 |

Each candidate gets its own `<repo-dataset>` and `<model-dataset>`.
Do NOT reuse across candidates.

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

## How to create a repo dataset for a candidate

The repo dataset contains the full repository as individual files with
directory structure (e.g. `scripts/run_predict.py`, `src/rogii/train.py`).

**Use `kagglehub.dataset_upload`.** It preserves directory structure when
Kaggle extracts files server-side. The "creating a zip archive" message
during upload is misleading — paths are preserved.

**Do NOT use `kaggle datasets version -p` or `kaggle datasets create -p`.**
They skip subdirectories and break the dataset.

### Procedure (fully automated, no Kaggle UI)

```python
import kagglehub, os, shutil, subprocess

repo_root = "/path/to/project"
tmp_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "repo-upload")
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)

# Copy only git-tracked files (no data/, models/, outputs/, *.pkl)
result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=repo_root)
for f in result.stdout.strip().split("\n"):
    if not f:
        continue
    dst = os.path.join(tmp_dir, f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(repo_root, f), dst)

# Upload — Kaggle extracts the zip server-side, paths preserved
kagglehub.dataset_upload(
    handle="daniilgonchar/rogii-repo-<candidate-slug>",
    local_dataset_dir=tmp_dir,
    version_notes="commit message or change description"
)
```

### Verification (mandatory)

```bash
kaggle datasets files daniilgonchar/rogii-repo-<candidate-slug>
```

Must show files with full paths: `src/rogii/train.py`, `scripts/run_predict.py`.
If only flat file names appear (no slashes), the dataset is broken. Re-upload.

### When to create

For every candidate where `src/`, `scripts/`, `configs/` or `requirements.txt`
changed AND the change is needed at inference time. Create a NEW dataset
(`rogii-repo-<slug>`) — never overwrite an existing one. If code has not
changed since the last candidate, reuse the same repo dataset.

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
|---|---|---|
| Repo dataset | Create candidate-specific: `daniilgonchar/rogii-repo-<slug>` via `kagglehub.dataset_upload()`. Never reuse for different candidates. |
| Model dataset | Candidate-specific: `daniilgonchar/rogii-models-<slug>`, containing `baseline_lgbm.pkl`. |
| Dependency dataset | If inference needs packages not available offline, create `daniilgonchar/rogii-wheels-<slug>`. |
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

## Dataset creation summary

| Dataset type | Method | Tool |
|---|---|---|
| Repo (multi-file, dirs) | `kagglehub.dataset_upload(handle, local_dir)` | Python |
| Model (single .pkl file) | `kaggle datasets create -p <dir>` | CLI |
| Dependency (wheels) | `kaggle datasets create -p <dir>` | CLI |

Always verify with `kaggle datasets files <dataset>` after creation.

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

- Do NOT use `kaggle datasets version -p` or `kaggle datasets create -p`
  for multi-file repo datasets. They skip subdirectories and WILL break
  the dataset. Use `kagglehub.dataset_upload()` instead.

- Do NOT overwrite an existing repo dataset. Each candidate must create
  its own repo dataset (`rogii-repo-<slug>`) to avoid breaking fallbacks.

- Do NOT change `<repo-dataset>` or `<model-dataset>` references in
  `kernel-metadata.json` unless the dataset itself was intentionally
  rebuilt under a new name.

- Do NOT replace the `find_repo_root()` discovery logic with zip-extraction,
  git-clone, or hardcoded-path approaches.

- Do NOT overwrite an existing inference kernel for a new experiment.
  Create a new kernel (see "New stage = new kernel").
