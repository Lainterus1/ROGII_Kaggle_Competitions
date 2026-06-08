# Decisions

## Purpose

Record important project decisions, alternatives considered and rationale.

## Owns

Architecture decisions, workflow decisions, validation decisions, dependency decisions and competition-specific trade-offs.

## Update when

- A meaningful technical or workflow choice is made.
- Validation strategy changes.
- A major dependency is added or removed.
- Public notebook ideas are adopted.

## Do not store here

- Raw experiment logs.
- Full task backlog.
- Detailed data inventory tables.
- Secrets or private credentials.

## Current content

## ADR-000: Initial workflow constraints

Date: 2026-06-04  
Status: Accepted

### Context

The project starts from a bootstrap dossier and targets a reproducible Kaggle baseline for `ROGII - Wellbore Geology Prediction`.

### Decision

- Use local development as the main working environment.
- Use public GitHub repo `https://github.com/Lainterus1/ROGII_Kaggle_Competitions` as source of truth.
- Use Kaggle as a remote full-data executor.
- Keep Kaggle submissions manual unless the user changes this rule.
- Require MLflow for model baseline experiment tracking.

### Consequences

#### Positive

- Core logic remains versioned and reusable.
- Kaggle notebooks stay thin.
- Submission risk is reduced by manual approval.

#### Negative

- Manual submission adds a human step to the loop.

#### Follow-up

- Push project skeleton to GitHub after local bootstrap.
- Confirm Kaggle metric and data contract after data inspection.

## ADR-001: Initial architecture

Date: 2026-06-04  
Status: Accepted

### Context

The project needs a reproducible Kaggle baseline architecture that supports local development, GitHub source control, Kaggle execution, MLflow tracking, validation checks, data inventory and valid submission generation.

At bootstrap time, the actual Kaggle data schema, target, submission contract and official metric are not yet confirmed. The architecture therefore must avoid hardcoding schema assumptions while still being ready for a vertical slice.

### Decision

Use a balanced Python package plus scripts architecture:

- `src/rogii/` for reusable project logic.
- `scripts/` for command-line entry points.
- `configs/` for environment paths and baseline settings.
- `tests/` for smoke and contract tests.
- `docs/` for source-of-truth documentation.
- `notebooks/` for thin Kaggle runners and lightweight exploration only.

### Alternatives considered

- Option A — Simple: script-first layout. Fast to bootstrap but likely to scatter data loading, validation, MLflow and submission logic.
- Option B — Balanced: small reusable package plus scripts and configs. Accepted because it matches the project dossier and first-baseline needs.
- Option C — Scalable: heavier pipeline/orchestration architecture. Rejected because it would over-engineer before data inspection.

### Consequences

#### Positive

- Core logic remains reusable locally and on Kaggle.
- Notebooks stay thin and do not own training logic.
- Submission validation, MLflow logging and validation strategy can be tested independently.
- Future baselines can extend the same structure without broad refactoring.

#### Negative

- More setup than a scripts-only project.
- Some modules may start as skeletons until real data is inspected.
- Kaggle runner depends on the latest local code being pushed to GitHub.

#### Follow-up

- Create project skeleton in Step 04.
- Initialize local git and push to `https://github.com/Lainterus1/ROGII_Kaggle_Competitions` after skeleton creation.
- Confirm official metric and data schema using Kaggle official sources and actual files.
- Use public unauthenticated clone in Kaggle: `git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git`.

## ADR-002: Public GitHub repository and Kaggle clone strategy

Date: 2026-06-04  
Status: Accepted

### Context

The Kaggle notebook is configured with internet enabled and competition data attached as Kaggle Input. The repository is public and initially empty.

### Decision

- Use public GitHub repository `https://github.com/Lainterus1/ROGII_Kaggle_Competitions`.
- Kaggle notebook will clone the repository with `git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git`.
- No GitHub auth, token or Kaggle Secrets are needed for clone.
- Kaggle data source is `/kaggle/input`.
- Kaggle outputs are written to `/kaggle/working`.

### Alternatives considered

- Private repo with Kaggle Secrets: rejected because the user confirmed a public repo and no auth requirement.
- Notebook-only development: rejected because core logic must live in `src/` and `scripts/`.

### Consequences

#### Positive

- Kaggle runner setup is simple.
- No token management is needed for clone.
- Local-to-GitHub-to-Kaggle workflow is straightforward.

#### Negative

- Repository contents are public, so no competition data, secrets, submissions, trained models or sensitive notes can be committed.
- Kaggle runner only sees code that has been pushed.

#### Follow-up

- Keep `.gitignore` strict.
- Verify `data/` is ignored before every commit.
- Push bootstrap skeleton before using the Kaggle notebook.

## ADR-003: Freeze Stage 4 baseline and create post-baseline roadmap

Date: 2026-06-05
Status: Accepted

### Context

The project has a valid Kaggle-submitted baseline: LightGBM with `safe_numeric_v1 + last_tvt_input`, 5-fold `GroupKFold` CV RMSE `20.58 +/- 3.99` on Kaggle and public LB RMSE `24.114`.

Two public notebooks were reviewed for improvement ideas. They suggest residual target modeling, GR rolling/lag features, geometry features, typewell alignment, CatBoost/LightGBM ensembles, beam search, particle filters, TabICL artifact stacks and exact coordinate-overlap blending.

### Decision

- Freeze Stage 4 as the reference baseline.
- Track post-baseline development in `docs/ROADMAP.md`.
- Prioritize clean, reproducible stages: residual GR/geometry features, simple typewell features, then model upgrades.
- Do not include public saved artifacts, TabICL artifact stacks or exact train/test coordinate-overlap blending in the clean mainline roadmap.

### Alternatives considered

- Continue editing `docs/BASELINE_PLAN.md` as the only plan document. Rejected because the baseline is now fixed and future work needs promotion gates and deferred-risk tracking.
- Adopt the strongest public artifact route directly. Rejected because it is opaque, dependency-heavy and conflicts with the reproducible baseline goal.

### Consequences

#### Positive

- The current baseline remains a stable comparison point.
- Future experiments have a clear order and promotion criteria.
- High-risk public-notebook ideas are documented instead of silently copied.

#### Negative

- Maintaining a separate roadmap adds one more source-of-truth document.
- Some public LB-improving tricks are intentionally deferred or rejected for mainline use.

#### Follow-up

- Implement Roadmap R1 before R2/R3.
- Ask the user before adding CatBoost to project dependencies.

## ADR-004: Residual target and forward-looking GR features for Roadmap R1

Date: 2026-06-05
Status: Accepted

### Context

Roadmap R1 adds residual target (predict `TVT - last_tvt_input` instead of raw `TVT`) and deterministic geometry/GR features. Decisions were needed on: (1) whether `last_tvt_input` stays as a feature in residual mode, (2) definition of `frac_after_ps`, (3) whether GR features may use centered rolling windows and forward-looking leads.

### Decision

- In residual mode, `last_tvt_input` is NOT a model feature. It is used only as the reconstruction base: `pred_tvt = last_tvt_input + pred_delta`.
- `frac_after_ps` is row-level progress through the post-PS section: `(i - ps_idx) / (n - ps_idx)`, ranging from 0 at PS to 1 at the last row.
- GR features use full well context (centered rolling windows, forward-looking leads) because all GR values are available in test data at prediction time.
- Trained model is saved as a dict `{"model": ..., "residual_target": True, "include_geometry": True, "include_gr": True}` so that `run_predict.py` auto-detects the feature configuration.
- Backward compatibility: `run_predict.py` handles both bare `LGBMRegressor` pickles (old format) and dict payloads (new format).

### Alternatives considered

- Keep `last_tvt_input` as a feature alongside residual target. Rejected to keep the model focused on learning deviation patterns.
- Centered vs backward-only GR windows. Centered chosen because test data contains full GR profiles.

### Consequences

#### Positive

- Clean separation: model learns delta, reconstruction is deterministic.
- Feature config auto-detection eliminates mismatch between train and predict.
- 32% CV improvement over frozen Stage 4 baseline.

#### Negative

- Old models saved as bare pickles don't carry feature config; user must pass correct CLI flags manually.
- `frac_after_ps` definition differs from some public notebooks; may need adjustment if comparing to external baselines.

#### Follow-up

- R2: typewell features require separate leakage review.

## ADR-005: Typewell V1 features rejected due to CV degradation

Date: 2026-06-05
Status: Accepted

### Context

Roadmap R2 added 15 typewell-reference features: 11 anchor-offset GR residuals (`tw_gr_residual_{offset} = horizontal_GR - typewell_GR(last_tvt_input + offset)`) and 4 summary statistics (`tw_range`, `tw_gr_mean`, `tw_gr_std`, `tw_gr_at_last_tvt`). All typewell data is available in both train and test.

### Decision

R2 typewell features are not promoted to the current best baseline.

### Rationale

R2 CV RMSE `14.75 ± 0.77` is 0.66 worse than R1 (`14.09 ± 0.88`). The anchor-offset residual features are highly correlated with the base `GR` feature (since `tw_gr_residual_{o} = GR - const`), adding redundancy without new signal. Summary features may be partially captured by the residual target approach already.

### Alternatives considered

- Keep typewell features with feature selection. Rejected for now — structure of features (GR minus constant) inherently limits added value.
- Different typewell alignment strategy (DTW, beam search). Deferred to a future roadmap stage after more fundamental improvements.

### Consequences

#### Positive

- R1 remains the cleanest and best-performing baseline.
- No unnecessary feature bloat in production model.

#### Negative

- Typewell data is not yet leveraged.
- May need a fundamentally different approach to typewell alignment.

#### Follow-up

- Submit R1 to Kaggle.
- Revisit typewell in a future roadmap stage with feature selection or alternative alignment.

## ADR-006: Replace old roadmap with staged geoscience feature plan

Date: 2026-06-05
Status: Accepted

### Context

R1 optimized is the current best clean baseline: LightGBM with 18 features, residual target `TVT - last_tvt_input`, GroupKFold by well, CV RMSE around `14.19` and public LB RMSE `12.247`.

The previous pending roadmap still contained a generic model-upgrade stage and a separate CatBoost dependency decision. The user provided a new technical direction focused on physically and geologically motivated features: trajectory kinematics, causal GR DWT, strict OOF spatial KNN, DTW typewell alignment, target engineering and structural blending.

### Decision

- Supersede the old pending roadmap with stages A0-A4 in `docs/ROADMAP.md`.
- Keep R1 optimized as the active comparison point for new experiments.
- Keep Stage 4 as the frozen historical reference baseline in `docs/BASELINE_PLAN.md`.
- Approve staged dependencies for this roadmap: `PyWavelets`, `scipy`, `catboost` and optionally `torch` for a later 1D CNN branch.
- For spatial KNN, use a strict default test-time contract: build the test reference tree from train pre-PS rows only, excluding test pre-PS rows.
- Keep Kaggle submissions manual only. The code may generate and validate `submission.csv`, but no automatic Kaggle submission path will be implemented.
- After any code push intended for Kaggle execution, provide the user with exact instructions for what to change in the competition notebook.

### Alternatives considered

- Continue with the old standalone R3 ensemble roadmap. Rejected because it conflicts with the new feature-first development sequence.
- Include test pre-PS rows in the spatial KNN reference tree. Rejected as the default because the stricter train-only reference makes validation/test behavior easier to reason about.
- Add automatic Kaggle submission support. Rejected by user policy; submissions remain manual.
- Implement all feature families at once. Rejected because A2/A3 have high leakage and rollback risk.

### Consequences

#### Positive

- Future work is ordered by risk: inspect, implement, verify, review and only then promote.
- Spatial features get an explicit OOF leakage contract before implementation.
- Kaggle workflow stays safe and user-controlled.
- Dependency additions are approved but still tied to stage-specific verification.

#### Negative

- More stages and gates add overhead before leaderboard submissions.
- The stricter spatial KNN test-time contract may leave signal unused if test pre-PS rows would have been safe and useful.
- Optional neural modeling remains deferred until tabular and stacking work is exhausted.

#### Follow-up

- Complete Stage A0 contracts before implementing A1 features.
- Update Kaggle notebook instructions after each push intended for Kaggle runs.
- Record every meaningful CV/LB result in `docs/EXPERIMENT_LOG.md`.

## ADR-007: Separate training and inference Kaggle notebooks with model artifact dataset

Date: 2026-06-05
Status: Accepted

### Context

The previous single notebook `00_kaggle_thin_runner.ipynb` ran both training and prediction every time, requiring ~10-20 min of 5-fold CV training for every submission even when the model was unchanged. This is a standard Kaggle workflow problem.

### Decision

- Split into two notebooks: training (`01_kaggle_train.ipynb`) and inference (`00_kaggle_inference.ipynb`).
- Training notebook saves the model to `/kaggle/working/baseline_lgbm.pkl`; the user creates a private Kaggle Dataset `rogii-models` from this output.
- Inference notebook loads the pre-trained model from the `rogii-models` dataset and runs only prediction — no training, no feature flags needed (model payload auto-detects feature config).
- Both notebooks remain offline (internet OFF), using the `rogii-repo` Kaggle Dataset for code.
- Model payload carries full metadata (feature flags, feature columns, residual target mode) so the inference notebook does not need any CLI feature flags.

### Alternatives considered

- Single notebook with conditional train/predict. Rejected — Kaggle notebook state is ephemeral; separating into two notebooks with a shared Dataset artifact is the standard Kaggle pattern.
- Git clone with internet ON. Rejected — competition rules may require offline notebooks for submission.

### Consequences

#### Positive

- Submission cycle drops from ~10-20 min to ~30 sec (prediction only).
- Training only runs when the model or features actually change.
- Clean separation of concerns: train produces artifact, inference consumes it.
- Compatible with offline execution requirements.

#### Negative

- Requires manual creation of `rogii-models` Kaggle Dataset after each training run.
- Two notebooks to maintain instead of one.

#### Follow-up

- Update `configs/paths.kaggle.yaml` with model input path.
- Update Kaggle notebook instructions in `docs/ROADMAP.md` after each code push.

## ADR-008: GitHub-linked Kaggle Notebooks with git clone for training

Date: 2026-06-05
Status: Accepted

### Context

ADR-007 split training and inference into separate notebooks, but code updates still required manual Import File or copying JSON between GitHub and Kaggle. This was friction for every feature change.

### Decision

- `01_kaggle_train.ipynb` and `02_kaggle_update_repo.ipynb` are linked to GitHub via Kaggle's built-in File → Link to GitHub feature (one-time setup per notebook).
- Before each run: File → Pull from GitHub fetches the latest notebook code and commands.
- Both use `!git clone` (internet ON) to get the full repo, eliminating the need for `rogii-repo` Dataset in training.
- `00_kaggle_inference.ipynb` remains NOT linked to GitHub and offline — code comes from `rogii-repo` Dataset (produced by `02`).
- `rogii-repo` Dataset is created by `02_kaggle_update_repo.ipynb` (Save Version → Create Dataset, one click in Kaggle UI after the notebook finishes).

### After git push workflow

1. `02`: Pull from GitHub → Run → Create Dataset `rogii-repo`
2. `01` (if model changed): Pull from GitHub → Run → Create Dataset `rogii-models`
3. `00`: Run (offline) → Submit

### Consequences

#### Positive

- Zero manual file downloads/uploads between GitHub and Kaggle.
- Notebook code stays in sync with repository via one button (Pull from GitHub).
- Inference stays offline for competition compliance.
- Training uses live git clone — always runs the latest pushed code without intermediate Dataset.

#### Negative

- Two notebooks require internet ON (training + repo update).
- Pull from GitHub is manual (not triggered by push).
- First-time setup requires linking two notebooks to GitHub.

## Open questions

- None for the current roadmap. Metric and schema are documented in `docs/METRICS.md` and `docs/DATA_MAP.md`.

## ADR-020: Multi-Seed LightGBM Averaging

Date: 2026-06-07
Status: Accepted

### Context

Stage A4 (Roadmap) planned multi-seed LightGBM averaging: train N models with different random states on same folds/features, average predictions. Expected reduction in CV variance and small mean improvement. Single seed (42) baseline: CV 14.191 ± 0.887.

### Decision

- Implement multi-seed training in `run_train()`: N models per fold, ensemble prediction = mean across seeds.
- Default seed list: `[42]` (single-seed backward compatible). Multi-seed: `[42, 7, 123]`.
- CV metrics computed on ensemble predictions (not individual seeds).
- Final model saves ALL models in a list. `build_model_payload()` stores list with backward compat for single models.
- `resolve_prediction_contract()` wraps single model in list if payload has `model` (not `models`).
- CLI: `--seed-list "42,7,123"` or `--n-seeds 3 --seed 42`.
- `TrainResult.model` → `TrainResult.models: list`. `PredictionContract.model` → `PredictionContract.models: list`.

### Results

3-seed ensemble: CV 14.052 ± 0.868. Mean improvement +0.139 vs single-seed (14.191). Std reduction −0.019 (0.887 → 0.868). Seed correlation is high (~0.95+) — models benefit from averaging but are not independent.

### Consequences

#### Positive
- +0.14 CV mean improvement with 3x training cost.
- Backward compatible: old single-model payloads load and predict correctly.
- `is_multi_seed` flag in `PredictionContract` for downstream awareness.

#### Negative
- Training time scales linearly with number of seeds (3 seeds = 3x time).
- Std reduction smaller than theoretical maximum (1/√3 ≈ 0.58x) due to high seed correlation.
- Model payload size increases linearly with seeds.

#### Follow-up
- Test on Kaggle LB: expected gain 0.1-0.3 RMSE.
- 9 new tests in `tests/test_train.py`, `tests/test_model_io.py`, `tests/test_validation_split.py`.

## ADR-021: StratifiedGroupKFold Validation Strategy

Date: 2026-06-07
Status: Accepted

### Context

Current GroupKFold (by well_id) produces optimistic CV (std 0.89) and LB/CV gap of −2.0 (LB better than CV — test wells easier than train). Matteo Niccoli's StratifiedGroupKFold approach balances folds by:
1. signed_azimuth_bin (updip/downdip)
2. median_tvt_quintile (target-stratified)  
3. spatial_cluster (KMeans X/Y, ~5 clusters)

### Decision

- Implement `build_stratification_labels()` in `validation.py` to compute per-well combined strat labels.
- Implement `create_cv_splitter()` returning `GroupKFold` or `StratifiedGroupKFold` based on strategy.
- Integrate into `run_train()` with `cv_strategy` parameter (`"group"` | `"stratified"`).
- CLI: `--cv-strategy stratified` with optional `--strat-tvt-bins` and `--strat-spatial-clusters`.
- Merge rare classes (< 5 wells) into nearest neighbor to satisfy StratifiedGroupKFold requirements.
- Default strat params: n_tvt_bins=4, n_spatial_clusters=5 → 33 classes, min=5 wells.

### Parameter sweep results

| Config | CV Mean | CV Std |
|---|---|---|
| GroupKFold (baseline) | 14.191 | 0.887 |
| Stratified (3,3) | 14.195 | 1.010 |
| Stratified (5,5) | 13.990 | 1.150 |
| Stratified (4,5) | 13.806 | 1.522 |
| Stratified (4,5) + multi-seed [42,7,123] | 13.763 | 1.523 |

### Analysis

- (4,5) is the best single-seed strat config: mean 13.806 (−0.385 vs GroupKFold 14.191).
- Higher std (1.522) is expected and potentially better for CV/LB correlation — folds are more heterogeneous.
- (3,3) is too coarse (18 classes) — identical to GroupKFold.
- (5,5) is in between (38 classes).
- Multi-seed + strat (4,5) is best overall: 13.763 (−0.428).

### Decision on default strategy

StratifiedGroupKFold (4,5) is NOT set as default — it produces higher std (1.522 vs 0.887) which is possibly more realistic but unvalidated on LB. Default remains GroupKFold. StratifiedGroupKFold is available as an experimental option for CV/LB correlation studies.

### Consequences

#### Positive
- Stratified CV available for experiments via `--cv-strategy stratified`.
- (4,5) config produces ~0.4 lower CV mean — potentially better calibrated.
- All configs validated with 5-fold StratifiedGroupKFold, no class below 5 wells.

#### Negative
- Higher CV std (1.5 vs 0.9) may not translate to better LB correlation.
- `median_tvt_quintile` uses train target for binning — standard for StratifiedKFold, but worth noting.
- Compute cost identical to GroupKFold (strat labels computed once).

#### Follow-up
- Validate StratifiedGroupKFold vs GroupKFold CV/LB correlation on Kaggle.
- If Stratified CV → LB gap is smaller, promote to default.

## ADR-018: Post-Processing Pipeline (PrP3) — Savgol Smoothing + TVT Clipping

Date: 2026-06-06
Status: Accepted

### Context

The tabular feature ceiling (CV ~14.1, LB ~12.2) is confirmed after 14+ rejected experiments. All feature families are exhausted. The remaining path for improvement within tabular models is post-processing: per-well smoothing of predicted TVT sequences and clipping to a reasonable physical range.

The Savgol smoothing code already existed in `src/rogii/smoothing.py` (ADR-015, B2b) but was never tested on real predictions. TVT clipping was a new idea — out-of-bounds predictions harm RMSE disproportionately.

### Decision

- Implement TVT clipping (`clip_predictions`) and AUX function (`compute_tvt_clip_bounds`) in `src/rogii/smoothing.py`.
- Implement unified `apply_postprocessing()` that chains clip → smooth.
- Add `--eval-postproc` flag to `run_train.py` that collects per-well OOF predictions and evaluates all post-processing configs (Savgol windows [5,11,17,25,31], polyorders [2,3], clip bounds [p0.1-p99.9, p0.5-p99.5, p1-p99]).
- Store `clip_lower` / `clip_upper` in model payload so predict can apply clipping without re-scanning train data.
- Add `--tvt-clip`, `--savgol-window`, `--savgol-polyorder` flags to `run_predict.py`.
- Order of operations: clip → smooth (remove outliers before filter).
- Default Savgol params: window=17, polyorder=3 (original hardcoded values from ADR-015).
- Scott Weeden v13 params (window=11, polyorder=2) included in grid search.
- Continuity check: max jump ≤ 30 ft between adjacent TVT predictions.

### Alternatives considered

- Smooth → clip order: rejected because outliers would distort the Savgol filter.
- Hardcoded clip bounds [11700, 12500] (plagiagia): rejected; use data-driven percentiles from train TVT.
- Separate grid search script: rejected; `--eval-postproc` in `run_train.py` is sufficient and avoids code duplication.

### Consequences

#### Positive

- Post-processing is orthogonal to feature/model development — it can improve any model's predictions.
- CV evaluation of post-processing uses honest OOF predictions (per-well grouping, same folds as raw CV).
- Clip bounds auto-detected from train data, no magic numbers.
- Model payload carries clip bounds for reproducible prediction-time post-processing.

#### Negative

- Savgol may smooth real formation boundaries (step changes in TVT). Per-well visualization needed as safeguard.
- Post-processing cannot fix systematic bias — only noise reduction.
- Expected improvement is marginal (0.05–0.2 RMSE) given the tabular ceiling.

#### Follow-up

- Ran `inspect_tvt_range.py` → clip bounds p0.1-p99.9 = [9851.80, 12860.23].
- Ran `run_train --eval-postproc` → Savgol w=31 p=2 best OOF RMSE 14.2123 vs raw 14.2187 (−0.0064). All Savgol configs beat raw. Clipping degrades (14.2208, +0.002).
- Ran `visualize_postproc.py` → 3/3 wells improved, raw max jumps 1.6-4.1 ft (noise, not geology), continuity excellent.
- **Kaggle LB `53428554`: 12.239** — −0.008 vs R1 (12.247). Improvement confirmed on both CV and LB. OOF→LB gap −1.97 (consistent with R1 −1.94). **Savgol w=31 p=2 is now the active baseline — designated as R2 pipeline (R1 model + Savgol).**
- Updated defaults: window=31, polyorder=2.

## ADR-014: B1 Beam Search rejected — typewell alignment adds no net signal

Date: 2026-06-06  
Status: Accepted

### Context

Stage B1 implemented Numba JIT beam search stratigraphic alignment: 7 beam configs with diverse move_cost / emit_scale, producing 19 features (per-config TVT estimates, consensus, std, consensus differences at 11 offsets). 9 tests verified JIT compilation, causal construction, and shape checks.

### Decision

Reject B1. Full CV 14.43 (5-fold) vs R1 14.19 (+0.24 worse). Beam-only (no geometry/GR): CV 16.02 (worse than naive 15.91).

Root cause: `beam_std` is #2 feature by importance (6.2%) but cannibalizes X/Y/Z spatial coordinate importance:

| Feature | R1 % | B1 % | Delta |
|---|---|---|---|
| X | 17.0% | 5.6% | -11.3% |
| gr_energy | 15.3% | 7.5% | -7.8% |
| Y | 14.9% | 5.5% | -9.4% |
| Z | 12.5% | 6.0% | -6.5% |

Beam features act as noisy proxies for spatial coordinates — they predict TVT through a roundabout pathway (GR → beam path → TVT estimate → prediction) that is less reliable than direct X/Y/Z → TVT.

This is consistent with all previous typewell-referenced experiments:
- R2 typewell residuals: CV 14.75 (+0.66 vs R1)
- A3a DTW alignment: CV 14.63 (+0.50 vs A2a)
- A4 geology v1/v2: degraded or flat
- B1 beam search: CV 14.43 (+0.24 vs R1)

### Consequences

- Code kept in `src/rogii/beam_search.py` behind `include_beam` feature flag for future compound experiments.
- Tabular feature ceiling at CV ~14.1 reconfirmed. 8 feature families tested, 2 promoted (geometry/GR → R1, DWT → A2a), 6 rejected/deferred.
- Future improvement requires architectural change: CNN for sequence modeling or ensemble methods (A4+ deferred stages).

## ADR-015: B2b Slope-Based Baseline Methods Rejected

Date: 2026-06-06  
Status: Accepted

### Context

Stage B2b tested replacing the flat `last_tvt_input` baseline with slope-based extrapolation from the known zone: `slope_md` (global MD trend), `slope_recent` (last 200 MD rows), `wls` (exponential-weighted recent MD), and `slope_z` (Z-based). The hypothesis was that providing the model with a smarter baseline (instead of a flat constant) would reduce residual magnitude and improve CV.

### Decision

Reject all slope-based baselines. Results:

| Method | CV (3-fold) | vs R1 flat (14.19) |
|---|---|---|
| `flat` (R1) | 14.19 | reference |
| `slope_recent` | 14.16 | flat |
| `slope_md` | 284 | +19.9x worse |
| `wls` | 130 | +9.2x worse |

Root cause: TVT-vs-MD trend in the known zone does NOT linearly continue into the evaluation zone. After Prediction Start, wells frequently change direction (horizontal section, upward trajectory), making the global slope from the known zone a catastrophically wrong predictor. Even the local slope (`slope_recent`, using last 200 rows) does not outperform the flat baseline — the MD→TVT relationship at the PS boundary is not a reliable predictor of the evaluation zone trajectory.

### Consequences

- Code kept in `src/rogii/baseline.py` (5 methods via `compute_baseline()`) and `src/rogii/smoothing.py` (Savgol per-well smoothing).
- `baseline_method` added to model payload v2 contract. Existing models default to `"flat"` for backward compatibility.
- `--baseline-method` CLI flag available in `run_train.py` and `run_predict.py`.
- Savgol smoothing (`--savgol-smooth`) available as post-processing in `run_predict.py` (not yet tested on real predictions — may provide marginal noise reduction independently of baseline method).
- 10 new tests in `tests/test_baseline.py`. 132 tests total pass.
- The flat baseline (`last_tvt_input`) remains the optimal residual base for this problem.

## ADR-016: B3 Formation Plane KNN Rejected

Date: 2026-06-06  
Status: Accepted

### Context

Formation columns (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA) exist in train horizontal wells but are absent in test. 765/773 wells have all 6 formations present. Formation depths vary per-row (within-well std 35-64m) but thicknesses between formations are constant within a well. The approach: impute formation depths for test wells via KNN (k=10) in (X, Y) plane, using well-median coordinates. 21 features: 6 raw depths, 6 Z-relative, 5 thicknesses, 1 nearest-formation distance, 3 KNN uncertainty. Implemented as fold-aware OOF, same pattern as spatial_features.

### Decision

Reject B3. CV 14.99 (3-fold) vs R1 14.19 (+0.80 worse).

Root cause: same cannibalization pattern as B1:

| Feature | R1 % | B3 % | Delta |
|---|---|---|---|
| X | 17.0% | 3.7% | -13.3% |
| Y | 14.9% | 3.4% | -11.5% |
| Z | 12.5% | 3.3% | -9.2% |

FP features (especially `fp_knn_mean_dist` #5 at 5.0%, `fp_nearest_dist` #6 at 4.5%) take ~35% of model importance from spatial coordinates without adding net signal. The KNN-imputed formation depths act as noisy proxies for the direct X/Y/Z → TVT pathway.

### Consequences

- Code kept in `src/rogii/formation_plane.py` (3 functions), feature flag `include_formation_plane` in payload v2.
- 7 new tests in `tests/test_formation_plane.py`. 139 total tests pass.
- 12+ feature families tested, 2 promoted (geometry/GR → R1, DWT → A2a superseded), 10+ rejected/deferred.
- Tabular ceiling at CV ~14.1 and LB ~12.2 is now strongly confirmed.
- Remaining path: architecture change (CNN, ensemble) — A4+ deferred stages.

## ADR-009: A2a DWT promoted as active baseline

Date: 2026-06-05  
Status: Accepted

### Context

Stage A2a added causal GR DWT features (PyWavelets db4, trailing window 256): `gr_dwt_approx` + `gr_dwt_detail_energy`. Runtime 1.4 min full train, no leakage (causal verified by test).

### Decision

Promote A2a as active baseline. CV 14.13 vs R1 14.19 (+0.06). Feature flag `include_gr_dwt` integrated into payload contract.

### Consequences

- Active baseline: 20 features (6 base + 9 geometry + 3 GR + 2 DWT).
- Marginal CV improvement; DWT is a better GR filter, not a new information source.
- Superseded by R1: LB 12.558 worse than R1 12.247 (+0.311). DWT does not generalize. R1 remains active baseline.

## ADR-010: A3 DTW, signed-log, derivative — all rejected

Date: 2026-06-05  
Status: Accepted

### Context

Three target/alignment improvements tested: DTW typewell alignment (A3a), signed-log residual target (A3b.1), derivative dTVT/dMD target (A3b.2).

### Decision

Reject all three. DTW CV 14.63 (+0.50 vs A2a) — GR cross-correlation only 0.43. Signed-log CV 14.64 (+0.51) — residuals not heavy-tailed (bounded ±40, skew 0.74). Derivative CV 14.32 (+0.19) — integration error accumulation.

### Consequences

- Code kept behind feature flags (`include_dtw`, `--signed-log-target`). No runtime cost when disabled.
- Target remains plain residual `TVT − last_tvt_input`.

## ADR-011: Geology features — rejected

Date: 2026-06-05  
Status: Accepted

### Context

Two versions of Geology features tested: v1 (well-level constants: formation at PS, next formation, boundaries) and v2 (per-row GR z-scores to 8 formations). 43 unique formations in typewells.

### Decision

Reject both. v1 CV 14.57 (+0.44) — well-level constants cause unstable overfitting. v2 CV 14.17 (−0.04 flat) — per-row signal adds no net benefit.

### Consequences

- Code kept in `src/rogii/geology_features.py`. Feature flag `include_geology` available.
- GR statistics for 8 formations hard-coded in module.

## ADR-012: Tabular feature ceiling at CV ~14.13

Date: 2026-06-05  
Status: Accepted

### Context

8 experiments (A1–A4) over 2 sessions. All feature additions produced flat or degraded CV. 69% model importance from 4 features (X, Y, Z, gr_energy). Remaining 25+ features share 31% importance.

### Decision

Acknowledge tabular feature ceiling at CV ~14.13 for LightGBM with features derived from X, Y, Z, GR, TVT_input. Further improvement requires architectural diversity (CNN for sequence modeling) or ensemble methods.

### Consequences

- Active feature development paused. Focus shifts to A4+: CNN, multi-model ensemble.
- Existing feature flags preserved for future compound experiments.

## ADR-013: Metadata-driven offline Kaggle inference submit

Date: 2026-06-06
Status: Accepted

### Context

Kaggle Submit reruns code with internet OFF. The previous `00-rogii-inference-r1` notebook searched for a single nested `rogii-repo-v2/ROGII_Kaggle_Competitions*` layout, but the active `rogii-repo-v2` Dataset is mounted as a flat repository. This caused notebook rerun failures and zero-byte submissions without LB scores.

### Decision

- Keep the active recovery path as R1 inference with `rogii-repo-v2` and `rogii-models-v2` until A2a offline dependencies are packaged.
- Use marker-based Kaggle path discovery: repo root must contain `scripts/run_predict.py` and `src/rogii`; data root must contain `sample_submission.csv` and `test/`; model path is selected by `baseline_lgbm.pkl` with a preference for `rogii-models-v2`.
- Store `notebooks/kernel-metadata.json` so the inference kernel can be updated with `kaggle kernels push -p notebooks` and fixed inputs/internet settings.
- Submit through Kaggle's code-competition kernel-version mode after explicit approval: `kaggle competitions submit -k <kernel> -v <version> -f submission.csv`.
- Treat R1 as the recovery/fallback submit path. Candidate builds such as A2a must use explicit candidate artifacts: repo dataset, model dataset, dependency dataset when needed, kernel metadata and kernel slug.
- If a candidate needs packages unavailable in internet-OFF submit reruns, package them as an attached offline dependency dataset instead of relying on `pip install` from the internet.

### Consequences

- Offline R1 inference no longer depends on fragile dataset nesting or manual notebook cell edits.
- The kernel output is validated before submission and fails loudly if `submission.csv` is missing or empty.
- Direct file submission remains unsupported for this competition; kernel-version submit is the supported agent path.
- A2a DWT submission is now unblocked: `pywavelets` packaged as offline dependency dataset `rogii-wheels-a2a-dwt`; kernel `00-rogii-inference-a2a-dwt` v1 validated.
- New candidates can follow the same systematic flow without changing R1 fallback artifacts.

## ADR-017: PrP2 Z-Drift Physics Features Not Promoted

Date: 2026-06-06  
Status: Accepted

### Context

Scott Weeden v13 suggested a TVT-Z coupling physics prior: in the lateral section, TVT ≈ Z + local_offset. The idea is to compute `offset = last_tvt_input − Z_at_PS` and use it to derive `implied_tvt = Z + offset` and `resid = implied_tvt − last_tvt_input` as features. Matteo Niccoli confirmed dTVT/dZ = +0.057 in lateral — formation nearly flat, offset stable.

### Decision

Reject PrP2. 5-fold CV 14.20 vs R1 14.19 (+0.01, flat). Fold inconsistency: fold 2 improved by 0.80, fold 3 degraded by 0.62 — no consistent signal.

### Root Cause

Only 1 of 3 features carries new signal:
- `z_drift_offset_at_anchor` — new (well-level constant, encodes Z→TVT shift)
- `z_drift_implied_tvt` = Z + offset — linear duplicate of Z (r=1.0, offset is well-constant)
- `z_drift_implied_tvt_resid` = Z − Z_at_PS = dz_since_ps — identical duplicate (r=1.0)

The one novel feature (offset_at_anchor) is a well-level constant — LightGBM cannot split on it effectively because its value is identical for all rows in a well.

### Consequences

- Code kept in `src/rogii/features.py` (`Z_DRIFT_FEATURES`, `build_z_drift_features()`), feature flag `include_z_drift` integrated into payload v2.
- 12 new tests in `tests/test_feature_engineering.py`. 151 total tests pass.
- CLI flags `--include-z-drift` available in `run_train.py` and `run_predict.py`.
- Tabular ceiling at CV ~14.1 reconfirmed: 14 feature families tested, 2 promoted, 12 rejected/not promoted.

## ADR-019: PoP2 — Prediction-Time 3-Strategy Blend

Date: 2026-06-07
Status: Accepted (Rejected post-evaluation)

### Context

PrP2 (Z-Drift Physics as features) was flat (CV 14.20 vs R1 14.19) — the offset is a well-level constant, `implied_tvt` = Z+const and `implied_tvt_resid` = dz_since_ps are linear duplicates. However Scott Weeden v13 and plagiagia v2.8 both use physics/alignment-based TVT estimates as **independent predictors** blended with model output — not as model features. This is a fundamentally different approach: the physics signal, while too weak to help LightGBM split on, may have enough independent accuracy to improve blended output via ensemble averaging.

### Decision

- Implement 3-strategy prediction-time blend: **model + Z-physics + DTW GR matching**.
- Blend strategy: **median** by default. Weighted average as optional CLI parameter.
- Z-physics: `TVT_pred = Z_row + (last_tvt_input − Z_at_PS)` — flat formation assumption.
- DTW GR matching: for each row, take GR window ±25 points, slide across typewell between `[anchor − 300, anchor + 300]`, find TVT with minimum SAD. Uses `numpy.lib.stride_tricks.sliding_window_view` for vectorized computation.
- Order: blend → Savgol/clip. Savgol smooths the blended output.
- Missing typewell: fall back to 2-strategy blend (model + Z-physics).
- No model retraining: all strategies are prediction-time only.

### Alternatives considered

- Z-physics + DTW as **features** for model. Rejected — PrP2 already showed Z-drift features flat (CV 14.20). DTW features (A3a) also flat (CV 14.63). The signal is too weak for tree splitting but may still help in an ensemble average.
- Beam search + particle filter (plagiagia v2.8). Deferred — more complex, additional dependencies.
- Weighted average only. Rejected as default — median is more robust, especially when one strategy produces an outlier.

### Consequences

#### OOF Evaluation (5-fold GroupKFold, 773 wells, 3.78M rows)

| Strategy | OOF RMSE | vs Model |
|---|---|---|
| Model (R1, delta target) | 14.2187 | reference |
| Z-physics only | 111.29 | +97.07 |
| DTW matching only | 145.06 | +130.84 |
| PoP2 Median | 53.94 | +39.72 |
| Best weighted (0.8/0.1/0.1) | 22.79 | +8.57 |

Z-physics and DTW are 7-10× weaker than the model as standalone predictors. Any blend degrades accuracy — the model has no error pattern that physics/alignment can correct. This is identical to all previous physics/alignment experiments (PrP2 CV 14.20, A3a CV 14.63, B1 CV 14.43).

#### Positive

- Orthogonal feature test — confirmed that physics/alignment as independent predictors is not viable.
- Code kept behind `--postprocess-blend` flag for future experiments with stronger auxiliary strategies.

#### Negative

- No improvement. Tabular ceiling at CV ~14.2 reconfirmed.
- DTW matching is computationally expensive for OOF evaluation (773 wells).

#### Follow-up

- Rejected. No Kaggle submission.
- Code preserved: `src/rogii/z_physics.py`, `src/rogii/gr_matcher.py`, `src/rogii/postprocess.py`, `scripts/eval_pop2_oof.py`, `tests/test_postprocess.py`.

## ADR-022: Kaggle dataset upload via kagglehub (replaces notebook-only approach)

Date: 2026-06-07
Status: Accepted

### Context

ADR-008 specified that repo datasets can ONLY be created through Kaggle notebook output (02 → Save Version → Create Dataset). This required Kaggle UI interaction for every code change. The agent discovered that `kagglehub.dataset_upload()` preserves directory structure — the "creating a zip archive" message during upload is misleading; Kaggle extracts files server-side and paths are preserved.

The Kaggle CLI (`kaggle datasets version -p`) SKIPS subdirectories and breaks multi-file datasets. Confirmed by breaking `rogii-repo-v2`.

### Decision

- `kagglehub.dataset_upload(handle, local_dataset_dir)` is the primary method for creating repo datasets. Fully automated, no Kaggle UI needed.
- Always verify with `kaggle datasets files <dataset>` after creation — files must show full paths.
- For each candidate, create a NEW repo dataset (`rogii-repo-<slug>`). Never overwrite an existing one.
- Model datasets (single binary file) can use `kaggle datasets create -p <dir>` since no directories are involved.

### Consequences

#### Positive

- Zero Kaggle UI steps needed for repo dataset creation.
- Each candidate is fully self-contained (code + model datasets).
- R1 fallback protected — its datasets are never touched by new candidates.
- Errors caught early: always verify dataset file listing after creation.

#### Negative

- More Kaggle datasets to manage (one repo dataset per candidate).
- `kagglehub` must be installed locally (`pip install kagglehub`).

#### Follow-up

- Remove all references to manual notebook-based dataset creation from skills and docs.
- Add `kagglehub` to project dependencies if not already present.

## ADR-023: Strategic shift from feature engineering to architecture diversity

Date: 2026-06-07
Status: Accepted

### Context

All feature engineering experiments (Stages A1-A3, B1-B3, PrP2) resulted in flat or degraded CV vs R1 (14.19). Tabular ceiling at CV ~14.1 confirmed. However, top leader solutions achieve LB 5.99–7.5 — a 4.7–6.2 point gap vs R3 (12.177). Analysis of leader solutions and public notebooks was conducted to understand the gap.

### Analysis of Leader Methods (without participant names)

Methods observed across top 20 leaderboard solutions:

1. **Sequence models:** TCN (Temporal Convolutional Network) with causal convolutions on raw X,Y,Z,GR,MD + diff/lag/roll features. Architecture: [64,128,256] channels, kernel=3, GroupKFold 5, GPU training with AMP. Target normalization via StandardScaler.
2. **Beam Search + Particle Filter as independent predictors:** NOT as tabular features for gradient boosting. Blend ratio: `0.5×ML + 0.25×Beam + 0.25×PF`. Numba JIT beam search, 250-particle filter with GR-likelihood updates.
3. **Multi-strategy ensemble:** 3+ independent prediction strategies blended via Ridge stacking or weighted averaging. Strategies: ML model + deep learning + physics-based alignment.
4. **Seq-CNN with geology prediction:** PyTorch CNN + geology classifier (leave-one-well-out CV) + TVT projection methods (js_anchor, linear, gr_adjusted).
5. **Typewell matching:** merge_asof on Z↔TVT for per-row typewell features.
6. **Post-processing:** Z ± margin clipping, TVT range clipping, Savgol smoothing.

### Our Gap Analysis

| Capability | Leaders | We have | Action |
|---|---|---|---|
| Sequence model (TCN/CNN) | Yes | No | A5a |
| Physics predictors (Beam/PF) | Yes, as predictors | Yes, as features → degraded | A5b (repurpose) |
| Multi-strategy ensemble | Yes (3+ strategies) | Multi-seed LGBM only | A5c |
| OOF infrastructure | Implicit | In-memory only | A5.0 |

### Root Cause of B1 Failure

Stage B1 (Beam Search as LightGBM features) degraded CV to 14.43 because beam-derived features (`beam_std` was #2 by importance at 6.2%) cannibalized X/Y/Z spatial importance (X: 17%→5.6%, Y: 15%→5.5%, Z: 12.5%→6.0%) — zero-sum redistribution without net gain. LightGBM cannot combine beam alignment signal with spatial coordinates; it picks one. Leader approach avoids this by treating beam as a separate predictor blended at the output level, not the feature level.

### Decision

- **Abandon feature-engineering-first roadmap.** Stages A1-B3 confirmed diminishing returns on tabular features for LightGBM.
- **Adopt architecture-diversity roadmap (Stage A5):** OOF infrastructure → TCN sequence model → standalone physics predictors → multi-strategy ensemble → optional HPO.
- **Dependency `torch` is approved** for GPU sequence model training. GPU available locally.
- **Dependency `catboost` deferred to A6** — lower priority than TCN for architecture diversity.
- **No `weight` column** in competition data (DATA_MAP confirms 8 columns: MD,X,Y,Z,ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA,TVT,GR,TVT_input) — plain MSE loss, not weighted.

### Consequences

#### Positive

- Addresses the actual gap with leader solutions (architecture, not features).
- Reuses existing beam search code (`src/rogii/beam_search.py`) for A5b predictor mode.
- GPU enables TCN training at leader-comparable speed.
- Each stage independently verifiable via OOF infrastructure.

#### Negative

- `torch` is a heavy dependency (~800 MB). Kaggle GPU kernels support it natively.
- TCN training time: 1-2 hours for 5-fold CV on full data vs ~2 min for LightGBM.
- OOF infrastructure adds complexity to the training pipeline.

#### Follow-up

- Implement A5.0 (OOF infrastructure) first as gate for A5a-A5c.
- Implement A5a (TCN) as highest priority.
- Re-evaluate after TCN CV result: if CV < 12, continue A5b-A5c; if CV flat, re-analyze.

### Rejected alternatives

- **More feature engineering** — 10+ experiments all flat or degraded. Diminishing returns proven.
- **CatBoost first** — Ordered Boosting addresses grouped data but doesn't add sequential signal. TCN has larger potential upside.
- **XGBoost first** — same architecture family as LightGBM (tree-based). No architecture diversity.
- **Foundation models (TimesFM, PatchTST)** — promising in literature but heavy dependencies, unproven on this exact problem, higher risk than TCN.
- **Soft-DTW differentiable loss** — complex, requires tslearn, no leader evidence of necessity.
