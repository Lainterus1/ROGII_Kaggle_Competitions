# Metrics

## Purpose

Define the official Kaggle evaluation metric and the local implementation used for validation.

## Owns

Metric definition, local metric code contract, score direction, edge cases and differences between local validation and Kaggle evaluation.

## Update when

- Official Kaggle metric is confirmed.
- Metric implementation changes.
- Validation reports expose metric mismatch risks.

## Do not store here

- Full experiment history.
- Fold assignments.
- Submission files.

## Current content

The task deck `data/AI_wellbore_geology_prediction_task_en.pptx` states that prediction quality is measured as RMSE over all predicted TVT values.

Metric contract:

| Item | Value |
|---|---|
| Metric | RMSE |
| Target/prediction | `TVT` in train horizontal files, `tvt` in submissions |
| Direction | Lower is better |
| Local implementation | `src/rogii/metrics.py::rmse` |
| Validation scope for naive baseline | Train post-PS rows where `TVT_input` is missing |

Formula:

```text
RMSE = sqrt(mean((manualTVT - predictedTVT)^2))
```

Known limitations:

- The Kaggle Evaluation page text was not accessible through anonymous HTML fetch during bootstrap.
- Metric confirmation currently comes from the official task deck included in the competition data.

## Open questions

- Confirm the same RMSE wording from the Kaggle Evaluation page if accessible later.
