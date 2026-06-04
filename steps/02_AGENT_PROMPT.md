# Prompt 02 — Create source-of-truth documentation structure


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

Create the project-specific source-of-truth documentation structure.

Use:

- `Project Intake Dossier`;
- `Agent Bootstrap Roadmap`;
- `docs/PROJECT_CONTEXT.md`;
- planned document structure from the intake phase.

Do not guess from scratch if the dossier already planned the documents.

## Before creating files

Show a proposal:

```md
## Step 02 proposal

### Project type

[...]

### Always required docs

| File | Purpose | Update trigger |
|---|---|---|
| ... | ... | ... |

### Project-specific docs

| File | Why needed | Owns which facts | Update trigger |
|---|---|---|---|
| ... | ... | ... | ... |

### Not needed yet

| File | Reason |
|---|---|
| ... | ... |
```

Wait for the user's approval.

## After approval

Create or update the selected docs as lightweight source-of-truth skeletons.

Every document must contain:

```md
# <Document Title>

## Purpose

[...]

## Owns

[Which facts this document owns]

## Update when

- ...

## Do not store here

- ...

## Current content

[...]

## Open questions

- ...
```

## Required baseline documents

Unless there is a strong project-specific reason, create:

```text
docs/ARCHITECTURE.md
docs/CONTEXT_MAP.md
docs/DECISIONS.md
docs/TASKS.md
docs/CHANGELOG.md
docs/KNOWN_ISSUES.md
```

`docs/PROJECT_CONTEXT.md` already exists from Step 01.

## Rules

- Do not create excessive documentation.
- Do not duplicate the same information across multiple documents.
- Keep documents short and operational.
- Prefer ownership boundaries: each fact should have one primary source.
- If a document is not needed yet, say why.
