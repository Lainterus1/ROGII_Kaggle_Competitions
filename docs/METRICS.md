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

Official metric is not confirmed yet.

Required future content:

- Official metric name and formula.
- Whether lower or higher is better.
- Local implementation reference.
- Known differences from Kaggle scoring, if any.
- Sanity checks for finite predictions and target scale.

## Open questions

- What metric is specified on the Kaggle Evaluation page?
