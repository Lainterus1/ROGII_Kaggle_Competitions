# Prompt 03 — Design architecture options and select one


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

Choose the initial architecture through explicit alternatives and record the accepted decision.

Use:

- `Project Intake Dossier`;
- `docs/PROJECT_CONTEXT.md`;
- source-of-truth docs from Step 02;
- user constraints and preferences from the current dialogue.

## Before finalizing architecture

Propose 3 options:

```md
## Step 03 proposal

## Option A — Simple

### Structure

[...]

### Data/control flow

[...]

### Pros

- ...

### Cons

- ...

### Risks

- ...

### Cost of implementation

Low / Medium / High

### When this option becomes bad

[...]

## Option B — Balanced

[...]

## Option C — Scalable

[...]

## Recommendation

Recommended option: [...]

Why:
- ...

Trade-offs:
- ...

Assumptions:
- ...

Open questions:
- ...
```

Wait for user confirmation before writing final architecture decisions.

## After approval

Create or update:

```text
docs/ARCHITECTURE.md
docs/DECISIONS.md
docs/CONTEXT_MAP.md
```

## `docs/ARCHITECTURE.md` must include

```md
# Architecture

## Overview

[...]

## Selected architecture

[...]

## Components

| Component | Responsibility | Main files/directories |
|---|---|---|
| ... | ... | ... |

## Boundaries

- ...

## Data/control flow

1. ...
2. ...
3. ...

## Dependencies

[...]

## Deployment/runtime assumptions

[...]

## Architecture risks

- ...

## Open questions

- ...
```

## `docs/DECISIONS.md` must include an ADR

```md
## ADR-001: Initial architecture

Date: [current date if known, otherwise TBD]  
Status: Accepted

### Context

[...]

### Decision

[...]

### Alternatives considered

- Option A:
- Option B:
- Option C:

### Consequences

#### Positive

- ...

#### Negative

- ...

#### Follow-up

- ...
```

## Rules

- Do not write production code.
- Do not choose scalable architecture by default.
- Do not ignore user preferences from the dossier.
- Do not add infrastructure without a reason.
- If the user has strong architecture preferences, treat them as constraints but still compare alternatives.
