# Prompt 01 — Materialize `docs/PROJECT_CONTEXT.md` from Project Intake Dossier


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

Create or update:

```text
docs/PROJECT_CONTEXT.md
```

This file must become the primary source of truth for the project goal, users, scenarios, constraints, non-goals, and success criteria.

## Do not restart discovery

Use the `Project Intake Dossier` created by `PROJECT_BOOTSTRAP_MASTER_PROMPT.md`.

Do not ask the user to describe the project again.
Do not use placeholders if the dossier already contains the answer.
Do not silently invent facts.

## Before writing the file

Show a concise preview:

```md
## Step 01 proposal

### Project goal

[...]

### Users

[...]

### Core scenarios

1. ...
2. ...
3. ...

### Success criteria

- ...

### Constraints

- ...

### Non-goals

- ...

### Assumptions

- ...

### Open questions

- ...
```

Ask the user to approve or correct it.

## After approval

Create or update `docs/PROJECT_CONTEXT.md` with this structure:

```md
# Project Context

## One-sentence summary

[...]

## Goal

[...]

## Users

[...]

## Core scenarios

1. ...
2. ...
3. ...

## Success criteria

### Product criteria

- ...

### Technical criteria

- ...

### Quality criteria

- ...

## Constraints

### Business constraints

- ...

### Technical constraints

- ...

### Data constraints

- ...

### Security and privacy constraints

- ...

### Cost and infrastructure constraints

- ...

## Non-goals

- ...

## Initial assumptions

- ...

## Open questions

- ...
```

## Quality bar

The document must be:

- specific to this project;
- concise enough to be read by an agent before work;
- explicit about unknowns;
- free of invented requirements;
- useful as source-of-truth context.
