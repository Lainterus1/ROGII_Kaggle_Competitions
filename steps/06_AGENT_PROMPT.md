# Prompt 06 — Generate project-specific agent skills


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

Create reusable project-specific skills in:

```text
.agents/skills/*/SKILL.md
```

Use:

- `AGENTS.md`;
- `Project Intake Dossier`;
- `docs/PROJECT_CONTEXT.md`;
- `docs/ARCHITECTURE.md`;
- `docs/CONTEXT_MAP.md`;
- `docs/TASKS.md`;
- expected recurring workflows.

## Principle

Create a skill only if it is likely to be reused at least 3 times or protects a high-risk workflow.

Do not create skills just because they sound useful.

## Before creating files

Show a proposal:

```md
## Step 06 proposal

### Required skills

| Skill | Why needed | Expected reuse | Risk reduced |
|---|---|---:|---|
| ... | ... | ... | ... |

### Not needed skills

| Skill | Reason |
|---|---|
| ... | ... |

### Skill directory plan

```text
.agents/skills/
  task-implementation/
    SKILL.md
  documentation-maintenance/
    SKILL.md
  ...
```
```

Wait for user approval.

## Candidate skills

Choose only relevant ones.

### Baseline candidates

```text
task-implementation
documentation-maintenance
architecture-decision
code-review
bugfix
dependency-change
configuration-change
```

### API/backend candidates

```text
api-contract-change
database-migration
security-sensitive-change
```

### ML/DS candidates

```text
data-contract-change
experiment-logging
metric-change
feature-pipeline-change
model-training-change
```

### NLP/LLM/RAG/agent candidates

```text
prompt-change
retrieval-change
evaluation-change
tool-change
agent-orchestration-change
```

## Required format for each skill

Each skill file must be:

```text
.agents/skills/<skill-name>/SKILL.md
```

Each `SKILL.md` must contain:

```md
---
name: <skill-name>
description: <when the agent should use this skill>
---

# <Skill Name>

## When to use

[...]

## Inputs

[...]

## Source-of-truth files

- ...

## Procedure

1. ...
2. ...
3. ...

## Documentation updates

- ...

## Validation

- ...

## Completion checklist

- [ ] ...
- [ ] ...

## Forbidden actions

- ...
```

## Rules

- Do not duplicate `AGENTS.md`.
- Do not create generic skills unrelated to the project.
- Do not reference nonexistent files as existing.
- If a skill depends on a future document, mark it as planned.
- Keep each skill focused on one workflow.
