# Context Map

## Purpose

Tell future agents which project files to read first and which document owns each kind of fact.

## Owns

Documentation ownership boundaries, agent reading order and location map for key project context.

## Update when

- New source-of-truth documents are added.
- Project structure changes.
- Agent onboarding flow changes.
- A document's ownership changes.

## Do not store here

- Full project context copied from `PROJECT_CONTEXT.md`.
- Detailed architecture decisions.
- Experiment logs.
- Data inventory details.

## Current content

Recommended reading order for future agents:

1. `ROGII_PROJECT_INTAKE_DOSSIER.md` for original project intake.
2. `docs/PROJECT_CONTEXT.md` for current project goal, constraints and success criteria.
3. `docs/CONTEXT_MAP.md` for navigation and document ownership.
4. `docs/ARCHITECTURE.md` for selected architecture and component boundaries.
5. `docs/DECISIONS.md` for accepted decisions.
6. `docs/TASKS.md` for current backlog.
7. `AGENTS.md` after it is created in Step 05.

Repository and runtime locations:

| Need | Location |
|---|---|
| Public GitHub repo | `https://github.com/Lainterus1/ROGII_Kaggle_Competitions` |
| Kaggle clone command | `git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git` |
| Local competition data | `data/` |
| Kaggle competition data | `/kaggle/input` |
| Kaggle outputs | `/kaggle/working` |

Document ownership:

| File | Owns |
|---|---|
| `docs/PROJECT_CONTEXT.md` | Goal, users, scenarios, constraints, non-goals, success criteria |
| `docs/ARCHITECTURE.md` | Architecture, components, boundaries, runtime assumptions |
| `docs/DECISIONS.md` | Accepted decisions and rationale |
| `docs/TASKS.md` | Current backlog and next actions |
| `docs/TASK_TEMPLATE.md` | Copy-paste contract for future implementation tasks |
| `docs/CHANGELOG.md` | Chronological project changes |
| `docs/KNOWN_ISSUES.md` | Known risks, blockers and unresolved issues |
| `docs/DATA_MAP.md` | Data files, schema, target, IDs and leakage risks |
| `docs/METRICS.md` | Official metric and local implementation |
| `docs/VALIDATION_STRATEGY.md` | Validation design and leakage checks |
| `docs/BASELINE_PLAN.md` | Baseline stages and acceptance criteria |
| `docs/EXPERIMENT_LOG.md` | Human-readable experiment history |
| `docs/PUBLIC_NOTEBOOK_REFERENCES.md` | Reviewed public notebooks and reused ideas |

Documentation maintenance policy:

| Location | Owns |
|---|---|
| `AGENTS.md` | Concise documentation update rules and completion-report requirements |
| `.agents/skills/documentation-maintenance/SKILL.md` | Detailed documentation maintenance workflow and update matrix |
| `docs/TASK_TEMPLATE.md` | Per-task documentation impact section |

Planned architecture locations:

| Path | Purpose |
|---|---|
| `src/rogii/` | Reusable package for data, features, validation, models, submissions and MLflow |
| `scripts/` | Thin executable entry points |
| `configs/` | Local/Kaggle paths and run settings |
| `tests/` | Contract, validation, metric and smoke tests |
| `notebooks/` | Thin Kaggle runner and lightweight EDA only |
| `.agents/skills/` | Future reusable project-specific agent skills |

## Open questions

- None for current bootstrap navigation.
