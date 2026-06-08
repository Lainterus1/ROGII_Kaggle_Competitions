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

1. `docs/PROJECT_CONTEXT.md` for project goal, constraints, domain terms and success criteria.
2. `docs/CONTEXT_MAP.md` for navigation and document ownership.
3. `docs/ARCHITECTURE.md` for selected architecture and component boundaries.
4. `docs/DECISIONS.md` for accepted decisions.
5. `docs/TASKS.md` for current backlog.
6. `docs/ROADMAP.md` for post-baseline development direction.
7. `AGENTS.md` for agent operating rules, forbidden actions and completion-report format.

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
| `docs/REVIEW_CHECKLIST.md` | Review criteria, optimization boundaries and refactoring approval rules |
| `docs/CHANGELOG.md` | Chronological project changes |
| `docs/KNOWN_ISSUES.md` | Known risks, blockers and unresolved issues |
| `docs/DATA_MAP.md` | Data files, schema, target, IDs and leakage risks |
| `docs/HOW_IT_WORKS.md` | Beginner-friendly explanation of the model, features and pipeline |
| `docs/METRICS.md` | Official metric and local implementation |
| `docs/VALIDATION_STRATEGY.md` | Validation design and leakage checks |
| `docs/BASELINE_PLAN.md` | Baseline stages and acceptance criteria |
| `docs/ROADMAP.md` | Post-baseline development roadmap and promotion gates |
| `docs/EXPERIMENT_LOG.md` | Human-readable experiment history |
| `docs/PUBLIC_NOTEBOOK_REFERENCES.md` | Reviewed public notebooks and reused ideas |

Documentation maintenance policy:

| Location | Owns |
|---|---|
| `AGENTS.md` | Concise documentation update rules and completion-report requirements |
| `.agents/skills/documentation-maintenance/SKILL.md` | Detailed documentation maintenance workflow and update matrix |
| `docs/TASK_TEMPLATE.md` | Per-task documentation impact section |

Review and optimization protocol:

| Location | Owns |
|---|---|
| `.agents/skills/code-review/SKILL.md` | Detailed review workflow and findings format |
| `docs/REVIEW_CHECKLIST.md` | Concise checklist and optimization boundaries |
| `AGENTS.md` | High-level review protocol rules |

Handoff and context compaction:

| Location | Owns |
|---|---|
| `.agents/skills/handoff/SKILL.md` | Handoff format, produce and resume procedures |

Implemented architecture locations:

| Path | Purpose |
|---|---|
| `src/rogii/` | Reusable package for data, features, validation, models, post-processing, OOF, diagnostics, submissions and Kaggle runtime helpers |
| `scripts/` | Thin executable entry points for inventory, validation, training, prediction, diagnostics, tuning and Kaggle inference |
| `configs/` | Local/Kaggle paths and run settings |
| `tests/` | Contract, validation, metric and smoke tests |
| `notebooks/` | Thin Kaggle runner metadata and candidate inference kernels only |
| `.agents/skills/` | Reusable project-specific agent skills |
| `.opencode/opencode.json` | Project OpenCode configuration |
| `.opencode/plugin/rogii-guards.ts` | Project guard hooks for artifacts, Kaggle submission, leakage-sensitive edits and verification reminders |

OpenCode guard locations:

| Path | Purpose |
|---|---|
| `.opencode/opencode.json` | Loads the project-specific OpenCode plugin |
| `.opencode/plugin/rogii-guards.ts` | Blocks high-risk Git/Kaggle actions and emits low-noise reminders for validation, leakage review and experiment logging |

Kaggle runner locations:

| Path | Purpose |
|---|---|
| `src/rogii/kaggle_runtime.py` | Marker-based offline Kaggle repo/model/data discovery |
| `scripts/kaggle_offline_inference.py` | Offline Kaggle inference CLI wrapper |
| `notebooks/kernel-metadata.json` | Kaggle CLI metadata for `00-rogii-inference-r1` |
| `notebooks/kernels/a2a-dwt/` | Historical A2a DWT candidate kernel metadata and notebook |
| `notebooks/kernels/a4-multiseed/` | Current R3/A4 multi-seed candidate kernel metadata and notebook |
| `.agents/skills/kaggle-runner/SKILL.md` | Current Kaggle runner workflow and validation checklist |
| `.agents/skills/kaggle-candidate-build/SKILL.md` | Standard packaging contract for any new Kaggle candidate build |

A5/TCN and OOF locations:

| Path | Purpose |
|---|---|
| `src/rogii/sequence_features.py` | Causal sequence feature construction for TCN |
| `src/rogii/sequence_data.py` | Per-well sequence dataset/windowing utilities |
| `src/rogii/tcn_model.py` | TCN model implementation |
| `src/rogii/oof.py` | OOF persistence contract |
| `src/rogii/diagnostics.py` | OOF diagnostics and error analysis |
| `scripts/tune_tcn.py` | Fold-selectable TCN tuning/validation helper |
| `scripts/diagnose_tcn.py` | TCN vs LGBM OOF diagnostics |
| `configs/a5_tcn.yaml` | Current A5 TCN control/Phase 2 config |

## Open questions

- None for current bootstrap navigation.
