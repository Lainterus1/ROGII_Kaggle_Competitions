# Prompt 10 — Create review and optimization protocol


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

Create a safe project-specific review and optimization protocol.

Do not optimize the whole codebase in this step.
Do not perform broad refactoring.

Use:

- `AGENTS.md`;
- `docs/ARCHITECTURE.md`;
- `docs/CONTEXT_MAP.md`;
- `.agents/skills/*`;
- `Project Intake Dossier`;
- project-specific quality risks.

## Before changing files

Show a proposal:

```md
## Step 10 proposal

### Review criteria

- ...

### Optimization boundaries

- ...

### Refactoring approval rules

- ...

### Files to create/update

- ...

### Review report format

- ...
```

Wait for user approval.

## After approval

Create or update:

```text
.agents/skills/code-review/SKILL.md
```

Optionally create:

```text
docs/REVIEW_CHECKLIST.md
```

only if the checklist is useful enough to be referenced by future agents.

Update:

```text
AGENTS.md
docs/CONTEXT_MAP.md
```

only if needed.

## Required `code-review` skill structure

```md
---
name: code-review
description: Use when reviewing a module, diff, pull request, or completed agent task for correctness, architecture, test coverage, maintainability, and documentation impact.
---

# Code Review

## When to use

[...]

## Inputs

- Review target:
- Relevant docs:
- Recent task contract:
- Test results:

## Source-of-truth files

- `AGENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_MAP.md`
- `docs/DECISIONS.md`
- [project-specific contracts]

## Procedure

1. Read the relevant docs.
2. Identify intended behavior.
3. Inspect only relevant files.
4. Check architecture boundaries.
5. Check contracts.
6. Check tests.
7. Check error handling and edge cases.
8. Check performance-sensitive paths.
9. Check dependency changes.
10. Check documentation impact.
11. Produce findings before suggesting changes.

## Findings format

| Priority | Issue | Evidence | Suggested fix | Risk |
|---|---|---|---|---|
| High | ... | ... | ... | ... |

## Refactoring rules

- Prefer minimal safe changes.
- Do not change public contracts without approval.
- Do not introduce abstractions unless they remove clear duplication or risk.
- Do not refactor unrelated modules.
- Do not change architecture without ADR.

## Completion checklist

- [ ] Relevant docs inspected.
- [ ] Architecture boundaries checked.
- [ ] Contracts checked.
- [ ] Tests checked.
- [ ] Documentation impact checked.
- [ ] Findings prioritized.
```

## Rules

- Do not rewrite code during protocol creation.
- Do not create generic review advice.
- Make review criteria specific to this project.
