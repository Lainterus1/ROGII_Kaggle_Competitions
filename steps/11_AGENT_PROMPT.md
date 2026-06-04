# Prompt 11 — Create project handoff and context compaction


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

Create a compact handoff document for future agents and future conversations.

Use:

- full current conversation;
- `Project Intake Dossier`;
- `Agent Bootstrap Roadmap`;
- all created project docs;
- current project skeleton;
- completed step reports.

## Before creating the handoff

Show a proposal:

```md
## Step 11 proposal

### What the next agent must know

- ...

### What should be compressed

- ...

### What should not be duplicated

- ...

### Files to update

- `docs/HANDOFF.md`
- `docs/CONTEXT_MAP.md`
- optionally `README.md`
```

Wait for user approval.

## After approval

Create or update:

```text
docs/HANDOFF.md
docs/CONTEXT_MAP.md
```

Optionally update:

```text
README.md
```

only if it should point to `docs/HANDOFF.md`.

## Required `docs/HANDOFF.md` structure

```md
# Project Handoff

## Current project state

[...]

## Source-of-truth files

| Need | File |
|---|---|
| Project goal | ... |
| Architecture | ... |
| Tasks | ... |
| Data contracts | ... |
| API contracts | ... |
| Agent rules | ... |
| Skills | ... |

## Accepted decisions

- ADR-001: ...

## Current architecture

[Compact summary]

## Current commands

```bash
[setup/test/run commands or TBD]
```

## Completed bootstrap steps

| Step | Result |
|---|---|
| 01 | ... |
| 02 | ... |

## Current tasks

- ...

## First vertical slice status

[...]

## Open questions

- ...

## Known issues

- ...

## Risks

- ...

## Rules for next agent

- Read `AGENTS.md` first.
- Use `docs/CONTEXT_MAP.md` before searching the repository.
- Use relevant `.agents/skills/*/SKILL.md`.
- Do not change architecture without updating `docs/DECISIONS.md`.
- Do not invent data fields, API contracts, env vars, or business rules.
- Update documentation according to the documentation matrix.
```

## Rules

- Keep the handoff compact.
- Do not duplicate full documents.
- Do not copy the entire changelog.
- Do not hide unresolved contradictions.
- Mark assumptions clearly.
