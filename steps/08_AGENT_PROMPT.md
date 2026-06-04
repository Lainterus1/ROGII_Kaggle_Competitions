# Prompt 08 — Create project-specific task contract template


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

Create a project-specific template for all future implementation tasks.

This step does not execute a random task.
It creates the standard that future tasks must follow.

Use:

- `Project Intake Dossier`;
- `AGENTS.md`;
- `docs/PROJECT_CONTEXT.md`;
- `docs/ARCHITECTURE.md`;
- `docs/CONTEXT_MAP.md`;
- `docs/TASKS.md`;
- project-specific risks and workflows.

## Before creating files

Show a proposal:

```md
## Step 08 proposal

### Future task workflow

1. ...
2. ...
3. ...

### Required task sections

- ...

### Project-specific constraints to include

- ...

### Required checks

- ...

### Documentation impact section

- ...
```

Wait for user approval.

## After approval

Create or update one or both:

```text
docs/TASK_TEMPLATE.md
.agent/templates/TASK_CONTRACT.md
```

Use the path that best fits the project. If unsure, prefer:

```text
docs/TASK_TEMPLATE.md
```

## Required template structure

```md
# Task Contract Template

## Task

[What needs to be done]

## Why

[Why this matters for the project]

## Context

Use:

- `AGENTS.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_MAP.md`
- `docs/TASKS.md`
- [project-specific docs]

## Scope

Implement:

- ...

Do not implement:

- ...

## Acceptance criteria

1. ...
2. ...
3. ...

## Constraints

- Do not invent fields, APIs, env vars, or business rules.
- Do not change public contracts unless necessary.
- Do not add dependencies without justification.
- Do not perform broad refactoring.
- Keep changes minimal and testable.
- [project-specific constraints]

## Required workflow

1. Read relevant docs.
2. Identify minimal files.
3. State a short plan.
4. Implement.
5. Add/update tests.
6. Run relevant checks.
7. Check documentation impact.
8. Report assumptions and risks.

## Documentation impact

Choose one:

- Updated docs:
  - ...
- No docs update required because:
  - ...
- Docs update deferred because:
  - ...

## Completion report

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

## Also update

```text
docs/CONTEXT_MAP.md
AGENTS.md
```

only if needed to reference the task template.

## Rules

- Do not execute a new feature task in this step.
- Do not make the template generic if project-specific constraints are known.
- Keep the template copy-paste friendly.
