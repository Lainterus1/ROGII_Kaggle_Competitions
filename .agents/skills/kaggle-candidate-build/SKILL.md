---
name: kaggle-candidate-build
description: Use when packaging any ROGII model candidate for Kaggle submission, including new baselines, feature variants, ensembles, CNNs, or future builds such as A2b/B23.
---

# Kaggle Candidate Build

## When to use

Use this skill whenever a new model candidate must be packaged for Kaggle LB submission.

Use it for any build name or stage, not only A2a.

Examples: `r1`, `a2a-dwt`, `a2b-spatial`, `b23-ensemble`, `cnn-v1`.

Use `.agents/skills/kaggle-runner/SKILL.md` together with this skill when pushing Kaggle kernels, creating Kaggle Datasets, checking logs, or submitting kernel versions.

## Goal

Every candidate must have the same strict structure:

| Artifact | Rule |
|---|---|
| Candidate slug | Short stable slug, for example `a2b-spatial` |
| Source code | Comes from a candidate-specific repo dataset (`rogii-repo-<slug>`) created via `kagglehub.dataset_upload` |
| Model artifact | Stored in a candidate-specific Kaggle Dataset |
| Dependencies | Stored in a candidate-specific offline dependency dataset if needed |
| Inference kernel | Candidate-specific Kaggle kernel with metadata |
| Submission | Generated as `/kaggle/working/submission.csv` |
| Submit mode | Kaggle kernel-version submit, not direct file upload |

## Fixed rules

- Do not overwrite R1 fallback artifacts unless explicitly requested.
- Do not invent a new architecture for transferring builds to Kaggle.
- Do not put model logic in notebooks.
- Do not rely on internet during Kaggle Submit rerun.
- Do not use direct file submit for this competition.
- Do not commit data, models, wheels, submissions or runtime outputs.
- Submit only after explicit user approval.
- Record LB score only after it is actually available.

## Required candidate naming

Use this naming pattern unless the user gives a better slug:

| Item | Pattern | Example |
|---|---|---|
| Candidate slug | `<stage>-<short-name>` | `a2b-spatial` |
| Model dataset | `rogii-models-<slug>` | `rogii-models-a2b-spatial` |
| Dependency dataset | `rogii-wheels-<slug>` | `rogii-wheels-a2b-spatial` |
| Inference kernel | `00-rogii-inference-<slug>` | `00-rogii-inference-a2b-spatial` |
| Kernel folder | `notebooks/kernels/<slug>/` | `notebooks/kernels/a2b-spatial/` |
| Model file | `baseline_lgbm.pkl` or explicit documented name | `baseline_lgbm.pkl` |

## Procedure

1. Define candidate slug, purpose and expected feature/model flags.
2. Check whether source code changed since the last candidate. If unchanged, reuse the same repo dataset.
3. If source changed: create a NEW candidate-specific repo dataset (`rogii-repo-<slug>`) via `kagglehub.dataset_upload()`. Never overwrite an existing repo dataset.
4. Verify: `kaggle datasets files daniilgonchar/rogii-repo-<slug>` must show files with full paths like `src/rogii/train.py`.
5. Train or locate the candidate model.
6. Confirm the model payload stores feature flags, target mode and feature columns.
7. Upload the model file to the candidate model dataset via `kaggle datasets create -p <dir>` (single file, no directories).
8. Check whether inference imports any package unavailable in offline Kaggle.
9. If extra packages are needed, create a candidate dependency dataset with wheels.
10. Create a candidate inference kernel folder with notebook and `kernel-metadata.json`.
11. Metadata must include repo dataset, model dataset, dependency dataset if needed, and competition source.
12. Notebook must run with internet OFF.
13. Notebook may install dependencies only from attached offline inputs.
14. Notebook must call repository scripts and write `/kaggle/working/submission.csv`.
15. Push candidate kernel with `kaggle kernels push -p <candidate_kernel_folder>`.
16. Wait for auto-run, download output: `kaggle kernels output <kernel> -p <tmp> --file-pattern submission.csv`.
17. Validate submission with `scripts/validate_submission.py`.
18. Submit using kernel-version mode only after explicit user approval.
19. Update docs after run and after LB score is known.

## Submit command

Use:

```bash
kaggle competitions submit -c rogii-wellbore-geology-prediction -k <owner>/<candidate-kernel> -v <version> -f submission.csv -m "<message>"
```

Do not use:

```bash
kaggle competitions submit -c rogii-wellbore-geology-prediction -f submission.csv -m "<message>"
```

## Kernel metadata requirements

`kernel-metadata.json` must include:

```json
{
  "id": "daniilgonchar/00-rogii-inference-<slug>",
  "title": "00 - ROGII Inference (<slug>)",
  "code_file": "00_kaggle_inference.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "false",
  "dataset_sources": [
    "daniilgonchar/rogii-repo-<slug>",
    "daniilgonchar/rogii-models-<slug>"
  ],
  "competition_sources": [
    "rogii-wellbore-geology-prediction"
  ],
  "kernel_sources": [],
  "model_sources": []
}
```

If offline dependencies are required, add the dependency dataset to `dataset_sources`.

## Validation checklist

- [ ] Candidate slug is defined.
- [ ] R1 fallback artifacts are untouched.
- [ ] Source repo dataset is current.
- [ ] Candidate model dataset exists.
- [ ] Candidate dependency dataset exists if needed.
- [ ] Candidate kernel has its own metadata.
- [ ] Kernel internet is OFF.
- [ ] Notebook stays thin.
- [ ] Notebook writes `/kaggle/working/submission.csv`.
- [ ] Kaggle logs show correct repo/model/data paths.
- [ ] Kaggle logs show successful offline dependency install if needed.
- [ ] Output file is non-empty.
- [ ] Submission validates against sample.
- [ ] Submit uses kernel-version mode.
- [ ] LB score is recorded only after available.

## Documentation updates

Update only relevant docs:

| Event | Update |
|---|---|
| New candidate packaged | `docs/TASKS.md`, `docs/CHANGELOG.md` |
| New workflow decision | `docs/DECISIONS.md` |
| New blocker | `docs/KNOWN_ISSUES.md` |
| Real CV/LB result | `docs/EXPERIMENT_LOG.md` |
| User-facing command changed | `README.md` |

## Related skills

Use together with:

- `kaggle-runner` for Kaggle kernel/dataset/submit commands.
- `submission-validation` before any submit.
- `experiment-logging` after CV/LB results.
- `leakage-review` before feature/target changes.
- `documentation-maintenance` before finishing.

## Forbidden actions

- Do not use direct file submit for this competition.
- Do not rely on internet during Kaggle Submit rerun.
- Do not overwrite R1 fallback model/kernel/repo dataset.
- Do not update an existing repo dataset for a new candidate — always create a new one.
- Do not use `kaggle datasets version -p` for repo datasets — it skips directories. Use `kagglehub.dataset_upload()`.
- Do not commit model files, submissions, data, wheels, or runtime artifacts.
