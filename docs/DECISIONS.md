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

## Open questions

- None for the current roadmap. Metric and schema are documented in `docs/METRICS.md` and `docs/DATA_MAP.md`.
