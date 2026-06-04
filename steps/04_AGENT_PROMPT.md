# Prompt 04 — Create project skeleton


## Step Execution Contract

You are executing this step as part of a sequential project bootstrap workflow.

Primary input:

- current conversation;
- `Project Intake Dossier`;
- `Agent Bootstrap Roadmap`;
- outputs from previous completed steps;
- existing project files, if they already exist.

Do not ask the user to manually fill placeholders if the information already exists in the dossier.

If information is missing:

1. ask only targeted clarification questions;
2. mark unknowns explicitly;
3. do not silently invent facts.

Execution mode:

1. First produce a step proposal.
2. Wait for the user to approve, edit, or say `идём дальше`, unless the user explicitly requested direct execution.
3. After approval, create or update the relevant files for this step.
4. At the end, produce the required step report.
5. Do not move to the next step until the user confirms.

Step report format:

```md
## Step completed

### Step

`steps/XX_AGENT_PROMPT.md`

### Created/updated files

- ...

### Key decisions

- ...

### Assumptions used

- ...

### Open questions

- ...

### Documentation impact

- ...

### Ready for next step?

Yes / No
```


## Goal of this step

Create the initial project skeleton according to the selected architecture.

Use:

- `Project Intake Dossier`;
- `docs/PROJECT_CONTEXT.md`;
- `docs/ARCHITECTURE.md`;
- `docs/DECISIONS.md`;
- `docs/CONTEXT_MAP.md`.

## Stage 1 — Propose tree only

First output only the proposed project tree and rationale.

Do not create files yet.

Use this format:

```md
## Step 04 proposal

### Proposed project tree

```text
project/
  ...
```

### Directory rationale

| Path | Purpose | Why needed now |
|---|---|---|
| ... | ... | ... |

### Files intentionally not created yet

| File/path | Reason |
|---|---|
| ... | ... |

### Assumptions

- ...

### Open questions

- ...
```

Wait for user approval.

## Stage 2 — Create skeleton after approval

Create the skeleton files/directories.

Baseline files:

```text
README.md
AGENTS.md                  # temporary seed; final version comes in Step 05
docs/PROJECT_CONTEXT.md
docs/ARCHITECTURE.md
docs/CONTEXT_MAP.md
docs/DECISIONS.md
docs/TASKS.md
docs/CHANGELOG.md
docs/KNOWN_ISSUES.md
.agents/skills/
.env.example               # only if env vars are expected
```

Add stack-specific files only if the stack is known:

```text
pyproject.toml
package.json
Dockerfile
docker-compose.yml
Makefile
pytest.ini
ruff.toml
mypy.ini
tsconfig.json
```

## Update after creation

Update:

```text
docs/CONTEXT_MAP.md
docs/TASKS.md
README.md
```

## Rules

- Do not implement business logic.
- Do not add production dependencies without justification.
- Do not create excessive abstractions.
- Do not create generated files, logs, datasets, lock files, or large binary assets unless required.
- If a command is unknown, mark it as `TBD`.
