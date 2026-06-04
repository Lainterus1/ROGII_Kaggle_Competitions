# Prompt 09 — Create automatic documentation maintenance policy


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

Create a precise project-specific documentation maintenance system.

Future agents must update documentation automatically when needed, but not create documentation noise.

Use:

- `AGENTS.md`;
- `.agents/skills/documentation-maintenance/SKILL.md` if it exists;
- `docs/CONTEXT_MAP.md`;
- all source-of-truth documents;
- `Project Intake Dossier`.

## Before changing files

Show a proposal:

```md
## Step 09 proposal

### Documentation update matrix

| Change type | Update required | Do not update when |
|---|---|---|
| ... | ... | ... |

### Files to update

- `AGENTS.md`
- `.agents/skills/documentation-maintenance/SKILL.md`
- `docs/CONTEXT_MAP.md`
- ...

### Completion report documentation section

[...]

### Risks of documentation noise

- ...
```

Wait for user approval.

## After approval

Create or update:

```text
AGENTS.md
.agents/skills/documentation-maintenance/SKILL.md
docs/CONTEXT_MAP.md
```

Optionally create:

```text
docs/DOCUMENTATION_POLICY.md
```

only if the documentation policy is too large for `AGENTS.md`.

## Required documentation update rules

Include:

```md
## Documentation update rules

### Update docs when

- API behavior changes.
- Data schema changes.
- Architecture boundaries change.
- Configuration or env vars change.
- Public behavior changes.
- Experiment/evaluation protocol changes.
- Known limitation appears.
- Architectural decision is accepted.
- Project-specific contract changes.

### Do not update docs when

- Internal implementation changes without contract changes.
- Pure formatting changes.
- Test-only refactoring.
- Temporary debugging changes.
- Dead code removal that does not affect behavior.
- Minor code cleanup with no user-visible or operational impact.

### Update style

- Update the smallest relevant section.
- Do not duplicate information across docs.
- Do not append generic summaries.
- Prefer tables for contracts.
- Prefer ADR entries for decisions.
- Keep changelog factual and short.
```

## Required completion-report section

Every future task must include:

```md
## Documentation impact

- Updated:
  - ...
- Not updated because:
  - ...
- Deferred:
  - ...
```

## Rules

- Do not create duplicate policies in many files.
- Keep `AGENTS.md` concise.
- Put detailed procedure into the documentation-maintenance skill.
- If no documentation update is needed, the agent must explain why.
