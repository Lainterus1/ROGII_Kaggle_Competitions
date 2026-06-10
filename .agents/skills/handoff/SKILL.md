---
name: handoff
description: Use when finishing a session, handing off to another agent, or compressing context before switching tasks.
---

# Handoff & Context Compaction

## When to use

Use this skill at the end of a working session, before switching to an unrelated task, or when context is about to be lost.

Do not use it for trivial continuations where the next session shares full context.

## Inputs

- Current Linear issue and `docs/CHANGELOG.md`.
- Last commit hash and message.
- Current git status.
- Open questions and risks from `docs/KNOWN_ISSUES.md`.
- Recent decisions from `docs/DECISIONS.md`.
- Any in-progress work not yet committed.

## Source-of-truth files

- Linear MCP (`ROG-*` issues) for current task state
- `docs/TASKS.md` only for historical pre-Linear context
- `docs/KNOWN_ISSUES.md`
- `docs/DECISIONS.md`
- `docs/CHANGELOG.md`
- `AGENTS.md`

## Procedure — produce handoff

1. Collect current goal: the Linear issue or step being worked on.
2. List completed items with last commit hash.
3. Identify in-progress work: what the next immediate action is, and whether anything is blocked.
4. Note key decisions made in this session that are not yet in ADRs.
5. List active risks and open questions that affect continuation.
6. List the files most relevant to resuming work.
7. Output the handoff in the compact format below.

## Handoff format

```md
## Goal

[One-line current objective]

## Progress

- Issue: [ROG-<id> and status]
- Completed: [steps done, last commit hash]
- In progress: [what was being worked on, blocked?]
- Next: [immediate next action]

## Key decisions

- [brief notes; add ADRs later if significant]

## Risks / open questions

- [active concerns that affect next steps]

## Key files

- [files most relevant to continuation]
```

Keep it under 20 lines. Do not repeat the full project docs — reference them by file name. The handoff is a pointer, not a duplicate.

## Procedure — resume from handoff

1. Read the handoff block.
2. Open `docs/CONTEXT_MAP.md` and follow the recommended reading order for unfamiliar areas.
3. Run `git log --oneline -5` to verify last commit.
4. Run `git status --short --branch` to confirm clean state or understand pending work.
5. Read any key files listed in the handoff.
6. Continue from the "Next" action.

## Where to store

Handoffs are inline in the agent's final message. No persistent handoff file is required. The format is the contract — any agent that sees this block in a previous message should consume it as a handoff.

If a durable handoff is needed for long gaps, write it to a temporary file in the workspace root (not committed) and reference it in the final message.

## Completion checklist

- [ ] Goal, progress, next action, risks and key files captured.
- [ ] Last commit hash included.
- [ ] Handoff is concise (under 20 lines of prose).
- [ ] No duplicate of full docs — references only.

## Forbidden actions

- Do not commit handoff files unless explicitly requested.
- Do not duplicate entire docs in a handoff.
- Do not produce a handoff mid-task unless context is about to be lost.
- Do not use handoffs to skip reading source-of-truth docs.
