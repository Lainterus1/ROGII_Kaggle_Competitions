# Prompt 07 — Propose and implement the first vertical slice


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

Define and implement the first minimal vertical slice.

This is the first step where minimal production code may be created.

Use:

- `Project Intake Dossier`;
- `AGENTS.md`;
- relevant skills from `.agents/skills/`;
- `docs/PROJECT_CONTEXT.md`;
- `docs/ARCHITECTURE.md`;
- `docs/CONTEXT_MAP.md`;
- `docs/TASKS.md`;
- relevant contracts.

## Stage 1 — Propose only

First propose the vertical slice.
Do not implement it yet.

Use this format:

```md
## Step 07 proposal

### User/business scenario

[...]

### End-to-end path

[...]

### Files likely involved

- ...

### Acceptance criteria

1. ...
2. ...
3. ...

### Tests/checks

- ...

### Documentation updates expected

- ...

### Out of scope

- ...

### Assumptions

- ...

### Open questions

- ...
```

Wait for user approval.

## Stage 2 — Implement after approval

Implement the smallest useful version.

Required:

1. Make focused code changes.
2. Add or update tests/smoke checks.
3. Run the smallest relevant checks.
4. Update only relevant documentation.
5. Update `docs/TASKS.md`.
6. Update `docs/CHANGELOG.md` only if the change is user-visible or operationally relevant.

## Rules

- Do not implement adjacent features.
- Do not refactor unrelated code.
- Do not add dependencies without justification.
- Do not change architecture without a new ADR.
- Do not invent data fields, APIs, env vars, or business rules.
