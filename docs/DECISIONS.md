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

## Open questions

- What is the official Kaggle metric?
- What are the exact train/test/sample submission schemas?
