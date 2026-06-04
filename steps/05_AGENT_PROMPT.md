# Prompt 05 — Generate project-specific `AGENTS.md`


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

Create the final project-specific `AGENTS.md`.

This file is the stable operating contract for future coding agents.

Use:

- `Project Intake Dossier`;
- `docs/PROJECT_CONTEXT.md`;
- `docs/ARCHITECTURE.md`;
- `docs/CONTEXT_MAP.md`;
- `docs/DECISIONS.md`;
- current project skeleton;
- known setup/test/lint commands.

## Before writing the file

Show a proposal:

```md
## Step 05 proposal

### Agent role

[...]

### Source-of-truth files

- ...

### Context retrieval policy

- ...

### Architecture rules

- ...

### Testing policy

- ...

### Documentation update matrix summary

- ...

### Forbidden actions

- ...

### Unknown commands or paths

- ...
```

Wait for user approval.

## After approval

Create or update:

```text
AGENTS.md
```

## Required structure

```md
# AGENTS.md

## Role

[...]

## Project summary

[...]

## Source of truth

| Need | File | Notes |
|---|---|---|
| ... | ... | ... |

## Context retrieval policy

1. ...
2. ...
3. ...

## Work protocol

1. Plan
2. Implement
3. Test
4. Review
5. Document

## Architecture rules

- ...

## Code quality rules

- ...

## Testing policy

- ...

## Documentation update matrix

| Change type | Required documentation update |
|---|---|
| ... | ... |

## Skills policy

- ...

## Forbidden actions

- ...

## Completion report format

```md
## Summary

[...]

## Changed files

- ...

## Tests/checks

- ...

## Documentation impact

- ...

## Assumptions

- ...

## Risks/follow-up

- ...
```
```

## Rules

- Do not write generic instructions.
- Use only real or explicitly planned files.
- Do not reference unknown commands as facts.
- Mark unknown commands as `TBD`.
- Do not duplicate the full content of all docs.
- Keep it operational and concise.
