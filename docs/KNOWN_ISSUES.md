# Known Issues

## Purpose

Track known risks, blockers, unresolved questions and validation doubts.

## Owns

Current issues that could affect correctness, reproducibility, data leakage, Kaggle execution or submission validity.

## Update when

- A blocker or risk is discovered.
- A validation concern appears.
- A known issue is resolved.
- Data inspection reveals schema or leakage risks.

## Do not store here

- General backlog tasks.
- Full experiment logs.
- Public notebook references.

## Current content

| Status | Issue | Impact | Next action |
|---|---|---|---|
| Open | Kaggle data is only lightly inspected | Full schema, missing values and target contract are not documented yet | Implement and run data inventory |
| Open | Bootstrap code not pushed yet | Kaggle runner cannot clone current local skeleton until push | Initialize git, commit and push to public repo |
| Open | Validation group column unknown | Leakage-safe CV cannot be finalized | Find well/group ID candidates during data inventory |
| Open | Kaggle official pages require better access than anonymous HTML fetch | Metric could not be confirmed from fetched page text | Use Kaggle API, task deck or competition page during data inspection |
| Open | `TVT_input` exists in train and test horizontal well files | It may be allowed input or leakage-adjacent; usage must be justified | Audit task deck and official docs before using as a feature |

## Open questions

- What is the official metric?
- What are the exact train/test/sample submission schemas?
