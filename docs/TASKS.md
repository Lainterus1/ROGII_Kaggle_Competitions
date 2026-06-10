# Tasks

## Purpose

Preserve the historical pre-Linear project backlog and completed-task timeline.

## Owns

Historical task records created before Linear became the centralized task tracker.

## Update when

- The archival policy changes.
- A historical entry needs factual correction.
- A migration note needs to be clarified.

## Do not store here

- Long technical explanations.
- Experiment result details.
- Architecture rationale that belongs in `DECISIONS.md`.
- Current tasks, statuses, blockers or next actions. Use Linear MCP (`ROG-*` issues) instead.

## Migration note

As of 2026-06-10, current task tracking is centralized in Linear through MCP. Do not add new tasks or status updates here, and do not backfill old completed tasks into Linear. This file remains useful as a compact historical timeline for agents reconstructing project context.

## Current content

| Status | Priority | Task |
|---|---|---|
| Done | High | Step 01: create `docs/PROJECT_CONTEXT.md` |
| Done | High | Step 02: create source-of-truth documentation skeletons |
| Done | High | Step 03: select initial architecture and record ADR |
| Done | High | Step 04: create project skeleton |
| Done | High | Initialize local git and push bootstrap to `ROGII_Kaggle_Competitions` |
| Done | High | Step 05: create final `AGENTS.md` |
| Done | Medium | Step 06: create reusable project-specific agent skills |
| Done | High | Step 07: implement data inventory, RMSE metric, submission validator and naive baseline |
| Done | Medium | Step 08: create task contract template |
| Done | Medium | Step 09: create documentation maintenance policy |
| Done | Medium | Step 10: create review and optimization protocol |
| Done | Medium | Step 11: create handoff and context compaction |
| Done | High | Freeze Stage 4 LightGBM + `last_tvt_input` as the reference baseline |
| Done | High | Roadmap R1: implement residual target plus deterministic GR and geometry features |
| Done | High | Roadmap R2: implement simple typewell-reference features after leakage review (degraded CV, not promoted) |
| Done | High | R1 feature ablation: remove 20 zero/low-importance features, finalize 18-feature set with identical CV |
| Done | High | Submit R1 optimized model to Kaggle (LB: 12.247, 49% improvement over Stage 4) |
| Done | High | Replace old pending roadmap with the staged A0-A4 development plan |
| Cancelled | Medium | Old Roadmap R3 standalone model-upgrade task; superseded by Stage A4 structural blending |
| Cancelled | Medium | Old CatBoost decision task; dependency is now approved and belongs to Stage A4 |
| Done | High | Stage A0: sync CLI/config/README/model-payload contracts before new feature implementation |
| Rejected | High | Stage A1: trajectory features (CV flat, LB worse: 12.487 vs R1 12.247, all features are geometry duplicates) |
| Not promoted | High | Stage A2a: add causal GR DWT features — CV 14.13 (+0.06 vs R1), but LB 12.558 worse than R1 12.247; code kept |
| Done | High | Stage A2b: implement strict OOF spatial KNN — CV 14.21 (−0.02 flat), not promoted, code kept |
| Done | Medium | Stage A2 combined (DWT + spatial): CV 14.19 (flat), spatial neutralizes DWT gain |
| Rejected | High | Stage A3a: DTW typewell alignment — CV 14.63 (+0.50 vs A2a), insufficient GR correlation |
| Rejected | High | Stage A3b.1: signed-log residual target — CV 14.64 (+0.51), residuals not heavy-tailed |
| Rejected | High | Stage A3b.2: derivative dTVT/dMD target — CV 14.32 (+0.19), integration error accumulation |
| Rejected | High | Geology v1 (well-level): CV 14.57 (+0.44 vs A2a), signal unstable across folds |
| Not promoted | High | Geology v2 (per-row GR z-scores): CV 14.17 (−0.04 flat vs A2a), code kept |
| Superseded | Medium | 1D CNN sequence model — replaced by TCN in A5a |
| Superseded | Medium | Multi-seed LGBM + CatBoost + Ridge stacking — multi-seed promoted as R3, CatBoost deferred to A6 |
| Superseded | Medium | Stage A4: standardize OOF artifacts — replaced by A5.0 OOF infrastructure |
| Done | High | **Stage A5.0**: implement OOF infrastructure — `src/rogii/oof.py`, `TrainResult.oof_df`, `--save-oof` CLI flag; OOF contract includes `well_id,row_idx,fold,y_true,y_pred,baseline` |
| Done | High | **Stage A5a v0**: TCN raw implementation — `sequence_features.py`, `sequence_data.py`, `tcn_model.py`, `train_tcn()`, `predict_tcn()`, `configs/a5_tcn.yaml`, `torch` in `requirements.txt`, `--model-type tcn` in CLI, AMP support |
| Done | High | **Stage A5a v0**: TCN tests — `test_sequence_features.py`, `test_tcn_model.py`, `test_tcn_pipeline.py` |
| Done | High | **A5a Phase 0 (Diagnostics)**: TCN OOF by fold/position/well_length, prediction variance, RMSE by `frac_after_ps`, TCN vs LGBM OOF. Created `src/rogii/diagnostics.py`, `scripts/diagnose_tcn.py`, `tests/test_diagnostics.py` (14 tests). OOF contract + `fold` column. Key findings: RMSE 7.2→20.0 monotonic, std_ratio 0.42 flattening, error corr 0.76. Priority re-ranked: P2→P3→P4→P1→P5→P6. |
| Done | High | **A5a Fixed Control Config**: run `channels=[32,64,128] window=64 kernel=5 lr=3e-4` — CV 15.03 (4 folds, fold 5 aborted), OOF saved to `outputs/oof/a5_tcn_control_tcn_oof.parquet` |
| Done | High | **A5a Phase 1 (#1 priority)**: dual input normalization — per-well normalized (65 cols) + global-standardized absolute X/Y/Z/MD (4 cols). Fold-local `StandardScaler` fit on train folds only, final scaler saved in `tcn_metadata["x_scaler"]`, `predict_tcn()` applies saved scaler via `input_scaler` param. `tcn_input_size` stored in payload. `WellSequence.X_abs` added. 6 tests (2 new, 4 modified). Target: std_ratio 0.42 → >0.7. |
| Done | Medium | **A5a TCN tuning validation alignment**: `scripts/tune_tcn.py` now uses fold-selectable 5-fold `GroupKFold`, dense validation RMSE monitoring and prints the final `run_train.py` command. Best tuned small TCN CV recorded as `15.036 ± 0.848`; not promoted. |
| Planned | High | **A5a Phase 2 (#2 priority)**: R1 sequence channels — PS geometry, local trajectory, GR stable features, anchor/baseline, position masks. All causal/test-available, leakage audit. Target: screening folds close to 14.2–14.5. |
| Planned | High | **A5a Phase 3 (#3 priority)**: unified TCN eval path — extract common CV/eval logic, `tune_tcn.py` delegates to evaluator, `run_train.py` uses same evaluator, TCN OOF saved. Fixes fold 5 abort. |
| Planned | Low | **A5a Phase 4 (deprioritized)**: full-well `WellSequence` with target only on post-PS, window ending at target row, left-padding for short context, same windowing in `predict_tcn()`. Correctness fix — early post-PS rows already best (RMSE 7.2). |
| Planned | Medium | **A5a Phase 5 (Verify Loss)**: MSE baseline, `SmoothL1Loss` option, RMSE in TVT/delta scale, prediction dispersion check |
| Planned | Medium | **A5a Phase 6 (Postprocess & Blend)**: TCN OOF saved, Savgol/clipping on TCN OOF, LGBM vs TCN error correlation, weighted blend LGBM+TCN OOF |
| Planned | Medium | **Stage A5b**: implement standalone Beam/PF predictors — `beam_predictor.py`, `particle_filter.py`, OOF evaluation |
| Planned | Medium | **Stage A5c**: implement ensemble — `src/rogii/ensemble.py`, `scripts/run_ensemble.py`, Ridge stacking + weighted blending |
| Planned | Low | **Stage A5d**: Optuna HPO for TCN + LightGBM hyperparameters |
| Planned | Low | **Stage A6**: CatBoost + XGBoost for ensemble diversity |
| Done | High | Split Kaggle runner into separate training and inference notebooks (ADR-007) |
| Done | High | Repair R1 offline Kaggle inference workflow: marker-based path discovery, kernel metadata, version 3 output validation |
| Done | High | Generalize Kaggle workflow docs for A2a and future candidate builds with separate model/dependency/kernel artifacts |
| Done | High | Add `kaggle-candidate-build` skill for strict packaging of future Kaggle candidate builds |
| Done | Medium | Record public LB for R1 fixed workflow submission `53410572`: `12.247` |
| Done | High | Submit A2a to Kaggle: LB `12.558` (+0.311 vs R1 12.247), DWT doesn't generalize — not promoted |
| Done | High | Analyse public notebooks, define Stage B1 (Beam Search Stratigraphic Alignment), implement `src/rogii/beam_search.py` with 9 tests |
| Done | High | Stage B1: train beam search model (`configs/b1_lgbm.yaml`), evaluate CV — 5-fold 14.43 vs R1 14.19 (worse by +0.24). Beam features cannibalize X/Y/Z importance. |
| Cancelled | High | Stage B1: Kaggle candidate build — CV degraded, not promoted. |
| Rejected | High | Stage B1: Beam Search Stratigraphic Alignment — CV 14.43 (worse than R1 14.19). beam_std #2 feature but redistributes spatial importance without net gain. Same pattern as typewell/DTW/geology. |
| Done | High | Stage B2b: implement slope-based baseline methods (slope_md, slope_recent, wls, slope_z) + Savgol smoothing — tests pass (132), smoke OK |
| Rejected | High | Stage B2b: train baseline variants — best CV 14.16 (slope_recent) flat vs R1 14.19. slope_md CV 284, wls CV 130. TVT-vs-MD trend does not extrapolate after PS. Flat baseline remains optimal. |
| Done | High | Stage B3: implement Formation Plane KNN — `formation_plane.py`, 7 tests, fold-aware OOF |
| Not promoted | High | **PrP2: Z-Drift Physics Features** — 3 TVT-Z coupling features (offset_at_anchor, implied_tvt, resid). CV 14.20 (flat vs R1 14.19, +0.01). 2/3 features are linear duplicates of dz_since_ps and Z. Fold inconsistency: fold 2 improved 0.80, fold 3 degraded 0.62. Same pattern as geology v1. Code kept behind `include_z_drift` flag. |
| Done | High | **PrP3 Phase 1**: Create `scripts/inspect_tvt_range.py` — scans train TVT, prints percentiles and suggested clipping bounds |
| Done | High | **PrP3 Phase 2**: Add `clip_predictions()`, `compute_tvt_clip_bounds()`, `apply_postprocessing()` to `src/rogii/smoothing.py`. 7 new tests (17 total). |
| Done | High | **PrP3 Phase 3**: Modify `train.py` — collect per-well OOF predictions during CV, add `evaluate_postprocessing()` grid search, add `--eval-postproc` to `run_train.py`. |
| Done | High | **PrP3 Phase 4**: Wire postproc into `run_predict.py` — add `--savgol-window`, `--savgol-polyorder`, `--tvt-clip` flags. Store clip bounds in `model_io.py` payload. |
| Done | Medium | **PrP3 Phase 5**: Grid search integrated via `--eval-postproc` in `run_train.py` (tests Savgol windows [5,11,17,25,31] × polyorders [2,3] × clip bounds [none, p0.1-p99.9, p0.5-p99.5, p1-p99]). |
| Done | Medium | **PrP3 Phase 6**: Create `scripts/visualize_postproc.py` — per-well raw vs smoothed vs true TVT plots with continuity checks. |
| Done | High | **PrP3 Phase 7**: Document ADR-018 (DECISIONS.md), Stage PrP3 (ROADMAP.md), tasks (TASKS.md), update KNOWN_ISSUES.md. |
| Done | High | **PrP3 Evaluation**: Run `--eval-postproc` CV to determine best Savgol window + clip config on OOF predictions; completed by the following PrP3 Evaluate task |
| Done | High | **PrP3 Evaluate**: Savgol w=31 p=2 best (OOF 14.2123 vs raw 14.2187, −0.0064). All Savgol configs beat raw. Clipping rejected (+0.002). 3/3 wells improved in per-well viz. Defaults updated to w=31 p=2. |
| Done | High | **PrP3 Kaggle**: Submit Savgol w=31 p=2 → LB **12.239** (−0.008 vs R1 12.247). Promoted as R2 at the time; later superseded by R3. `ref 53428554`. |
| Done | High | **PoP2 Phase 1**: Create `src/rogii/z_physics.py` (`apply_z_physics`), `src/rogii/gr_matcher.py` (`apply_dtw_matching`), `src/rogii/postprocess.py` (`apply_postprocess_blend`). 16 tests pass. |
| Done | High | **PoP2 Phase 2**: Integrate into `scripts/run_predict.py` (`--postprocess-blend`, `--blend-weights` flags). Blend runs between `run_predict()` and Savgol/clip. |
| Rejected | High | **PoP2 Evaluate**: OOF CV 53.94 (+39.72 vs Model 14.22). Z-physics (111) and DTW (145) are weak standalone predictors — blend degrades model. Code behind `--postprocess-blend` flag. Same pattern as all physics/alignment experiments. |
| Done | High | **ROG-23 Phase 0**: Refactor `train.py` — extract `_fit_lgbm_single()`, `_build_lgbm_params()`, `_build_fold_features()`, `_make_validation_split()`. Code audit: removed hardcoded params, lazy LGBM import, typed helpers. |
| Done | High | **ROG-23 Phase 1**: Training progress monitoring — `tqdm` progress bars for CV folds + seeds, Rich Table for final summary (`scripts/run_train.py:_report_results`). |
| Done | High | **ROG-23 Phase 2**: Early stopping in `run_train()` — `early_stopping_rounds` param (default 50), `eval_set` in CV loop, `GroupShuffleSplit` holdout for final model, `best_iteration_` tracking. |
| Done | Medium | **ROG-23 Phase 3**: Custom objective support — `huber`, `quantile`, `tweedie` via `model_params["objective"]`. Config-driven, no new CLI flags needed. |
| Done | High | **ROG-23 Phase 4**: Optuna hyperparameter tuning — `src/rogii/tuning.py` (TPESampler + MedianPruner, 8-param search space), `scripts/run_tune.py` (CLI with MLflow), `configs/b4_tuned.yaml`. |
| Done | High | **ROG-23 Phase 5**: Tests + docs — `tests/test_tuning.py` (7 tests), `test_train.py` (+4 tests: ES, huber, quantile). ADR-020 in DECISIONS.md. All 237 tests pass. |
| Pending | High | **ROG-23 Training**: Run `python scripts/run_tune.py --config configs/b4_tuned.yaml --data-dir data` to find optimal params; retrain R3 with tuned params; compare CV/LB; update `docs/EXPERIMENT_LOG.md`. |
| Done | High | **ROG-23 Verify & Promote**: Top-3 verified on 5-fold CV. Trial 19 best: CV 13.948 ± 0.764 (−0.104 vs R3 14.052). Promoted as **B4** — new baseline. Params: lr=0.0664, leaves=48, min_child=60, subsample=0.716, colsample=0.733, reg_α=0.00154, reg_λ=0.000433, min_child_w=0.0096. Updated `configs/a4_multiseed.yaml`.

## Open questions

- A5 Phase 1 gate result will determine whether A5b/A5c remain planned or are re-prioritized.
- Stage-specific blockers are tracked in `docs/ROADMAP.md` and `docs/KNOWN_ISSUES.md`.
