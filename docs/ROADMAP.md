# Roadmap

## Purpose

Define the post-baseline development plan after the first Kaggle baseline has been frozen.

## Owns

Post-baseline improvement stages, priorities, promotion gates, high-risk deferred ideas and the short-term execution plan.

## Update when

- A roadmap stage is started, completed, promoted, rejected or reprioritized.
- A new public-notebook-derived idea is accepted into the project plan.
- A roadmap gate, constraint or verification rule changes.

## Do not store here

- Full experiment history; use `docs/EXPERIMENT_LOG.md`.
- Detailed baseline metrics and commands; use `docs/BASELINE_PLAN.md`.
- Raw public notebook code; use `docs/PUBLIC_NOTEBOOK_REFERENCES.md` for review notes.
- Data schema facts; use `docs/DATA_MAP.md`.

## Current content

Current best clean baseline:

| Item | Value |
|---|---|---|
| Stage | **B4** (3-seed ensemble [42,7,123] + Optuna-tuned params + Savgol w=31 p=2), CV 13.948, LB TBD |
| R3 | R1 model + 3-seed [42,7,123] + Savgol w=31 p=2, CV 14.052, LB 12.177 |
| R2 | R1 model + Savgol, CV 14.21, LB 12.239 |
| R1 | 18 tabular features, CV 14.22, LB 12.247 — model-only, no post-processing |
| Tabular ceiling | LB ~12.2, all A1-B3 experiments flat or degraded |

Active development: **Stage A5 — Architecture Diversity.** Analysis of top leader solutions (LB 5.99–7.5) reveals three critical gaps vs our approach:
1. **Sequence models** (TCN/CNN with causal convolutions on raw X,Y,Z,GR,MD) — leaders use them; we don't.
2. **Physics-based standalone predictors** (Beam Search + Particle Filter as independent strategies, NOT as tabular features for LightGBM) — we tried beam as features (degraded); leaders blend beam/PF as separate predictors.
3. **Multi-strategy blending** (ML model + physics + deep learning) — we have single-strategy multi-seed averaging.

Tabular ceiling at CV ~14.1 / LB ~12.2 is the LightGBM-only ceiling, not the problem ceiling. A5 targets CV < 10 via architecture diversity.

## Guiding constraints

- Keep core logic in `src/rogii/` and executable entry points in `scripts/`.
- Keep Kaggle notebooks thin runners.
- Keep `GroupKFold` by well as the primary validation strategy unless a documented decision changes it.
- Treat `TVT_input` as allowed only up to Prediction Start; never use post-PS labels or target-derived columns as features.
- Do not use public saved model artifacts, TabICL artifacts or exact train/test coordinate overlap blending in the clean mainline roadmap.
- Add new dependencies only as approved staged dependencies: `PyWavelets`, `scipy`, `catboost` and optionally `torch`.
- Promote a stage only after tests pass, leakage review passes, submission validation passes, and CV improves or the stage produces useful diagnostics.
- Kaggle submission requires explicit user approval. After approval, the agent may submit a validated kernel version through Kaggle CLI/API.
- Current offline submit path: inference kernel uses a candidate-specific repo dataset (`rogii-repo-<slug>`) + model dataset, internet OFF. Each candidate gets its own datasets — never update a shared repo dataset.
- For candidate builds, do not overwrite R1 fallback artifacts. Use candidate-specific repo/model/dependency datasets and kernel metadata. If source code changed, create a fresh candidate repo dataset via `kagglehub.dataset_upload()` — never update an existing one.

## Standard Gate After Each Stage

1. Run `python -m pytest tests`.
2. Run full 5-fold GroupKFold CV for the candidate stage.
3. Compare against R3 and the latest promoted stage.
4. Run leakage review for every new feature family or target transform.
5. Generate a candidate submission only in an ignored runtime path such as `outputs/submission.csv` or `/kaggle/working/submission.csv`.
6. Run `python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv` locally or the Kaggle-equivalent command in the notebook.
7. Update `docs/EXPERIMENT_LOG.md` after meaningful runs.
8. Update the Linear issue when stage status, blockers or next actions change. Update `docs/ROADMAP.md`, `docs/VALIDATION_STRATEGY.md` or `docs/KNOWN_ISSUES.md` only when stage plan, validation contracts or risks change.
9. Do not commit `data/`, `outputs/`, `models/`, `submissions/`, `mlruns/`, OOF artifacts or generated submissions.
10. For R1 recovery submissions: push `00` with `kaggle kernels push -p notebooks`, validate kernel output, then submit the kernel version with `kaggle competitions submit -k daniilgonchar/00-rogii-inference-r1 -v <version> -f submission.csv` after explicit approval.
11. For candidate submissions: if source code changed, create a new candidate-specific repo dataset via `kagglehub.dataset_upload()` (see `.agents/skills/kaggle-runner/SKILL.md`). Train/upload a candidate model dataset, attach offline dependency dataset if needed, push a candidate kernel version, validate output, then submit the candidate kernel version after explicit approval.

## Stage A0: Pipeline Contracts and Roadmap Reset

Goal: make the new staged roadmap executable before adding high-risk features.

Status: Done. Train can load `configs/baseline_lgbm.yaml`; saved model payloads include feature flags and exact feature columns; predict validates the generated feature matrix against the saved payload before writing a submission.

Scope:

- Replace old pending roadmap items with stages A1-A4.
- Keep R3 as the active baseline and Stage 4 as the historical frozen baseline.
- Ensure model payloads carry enough metadata to reproduce prediction: feature flags, target mode, feature columns and run name.
- Sync README and config examples with the currently supported CLI commands.
- Keep Kaggle workflow safe: stable R1 fallback remains available, while current and future candidates use candidate-specific repo/model/dependency datasets and kernel metadata (ADR-013, ADR-022).

Primary files:

- `docs/ROADMAP.md`
- Linear issue for task status and execution tracking
- `docs/DECISIONS.md`
- `docs/KNOWN_ISSUES.md`
- `README.md`
- `configs/baseline_lgbm.yaml`
- `src/rogii/config.py`
- `src/rogii/model_io.py`
- `src/rogii/train.py`
- `src/rogii/predict.py`
- `scripts/run_train.py`
- `scripts/run_predict.py`
- `tests/test_model_io.py`
- `tests/test_predict_contract.py`

Verification:

- `python -m pytest tests`
- `python scripts/run_train.py --help`
- `python scripts/run_predict.py --help`
- `PYTHONPATH=src python -c "from rogii.config import load_yaml_config; c=load_yaml_config('configs/baseline_lgbm.yaml'); assert c['features']['residual_target'] is True"` or PowerShell equivalent.
- `git diff -- docs/ROADMAP.md docs/DECISIONS.md docs/KNOWN_ISSUES.md README.md`

Promotion gate:

- New roadmap is the only active development plan and old pending tasks are explicitly cancelled or replaced.

## Stage A1: Spatial Kinematics and Trajectory Geometry

Status: **Rejected.** CV 14.24 (flat vs R1 14.19), Kaggle LB 12.487 (worse vs R1 12.247). All 5 trajectory features are near-perfect linear transforms of existing geometry features (r >= 0.99). No new signal — only importance redistribution across duplicates. Code remains in `features.py` for future experiments.

Goal: improve the tree model's representation of 3D trajectory curves without changing the model family.

Code remaining in repo (`TRAJECTORY_FEATURES`, `build_trajectory_features()`, `include_trajectory` flag). Rejection reason: all 5 features are linear duplicates of `GEOMETRY_FEATURES` (r >= 0.99 with `dz_since_ps`, `dzdmd`, `dxdmd`, `dydmd`). `dogleg_severity_10m` was ablated mid-experiment (importance 0.00).

## Stage A2: GR DWT and Strict OOF Spatial KNN

Status: **Implemented, not promoted.** A2a (DWT) improved local CV by `0.06` but worsened public LB to `12.558` vs R1 `12.247`; DWT does not generalize. A2b (spatial KNN) was flat on CV and not promoted. No leakage detected in either block.

Goal: extract deeper GR structure and add safe inter-well spatial context without target leakage.

### Stage A2a: Causal GR DWT — **Not promoted after LB**

CV: 14.13 (R1: 14.19, +0.06). Public LB: 12.558 (worse than R1 12.247). Runtime: ~1.4 min full train.

Feature scope:

- Added dependency `PyWavelets`.
- `gr_dwt_approx`: low-frequency GR approximation via trailing-window DWT (db4, window=256).
- `gr_dwt_detail_energy`: mean squared detail coefficient energy from same window.
- Causal only: feature at row `i` depends only on `GR[0:i+1]` (verified by test).

Primary files:

- `requirements.txt` — added `pywavelets`
- `src/rogii/gr_dwt.py` — new module
- `src/rogii/features.py` — `GR_DWT_FEATURES` constant, `build_gr_dwt_features()` integration
- `src/rogii/model_io.py` — `include_gr_dwt` flag
- `tests/test_feature_engineering.py` — 8 DWT tests including causal verification
- `configs/a2_lgbm.yaml` — DWT config

### Stage A2b: Strict OOF Spatial KNN — **Not promoted (flat CV)**

CV: 14.21 (R1: 14.19, −0.02). No leakage (CV not implausibly low). Spatial features add no new signal beyond what X/Y/Z coordinates already encode for LightGBM.

Feature scope implemented:

- KNN for `k = 5`, `10`, `50` in 3D space using `sklearn.neighbors.NearestNeighbors` (ball_tree).
- `spatial_nn{k}_mean_tvt`, `spatial_nn{k}_median_tvt`, `spatial_nn{k}_std_tvt`.
- Reference: pre-PS rows from other wells, `TVT_input` only.

OOF contract (all verified by tests):

- For validation fold K, reference tree built only from wells outside fold K.
- Reference rows are only pre-PS rows.
- Reference target is pre-PS known `TVT_input`, not post-PS `TVT`.
- Validation wells never appear in their own spatial reference tree.
- Test-time tree uses all train pre-PS rows.

Primary files:

- `src/rogii/spatial_features.py` — new module
- `src/rogii/train.py` — fold-aware spatial feature building
- `src/rogii/predict.py` — `_run_predict_with_spatial`
- `tests/test_spatial_oof.py` — 9 tests including OOF leakage checks

## Stage A3: DTW Typewell Alignment and Target Engineering

Status: **Rejected.** All three sub-stages degraded CV vs A2a (14.13).

- A3a DTW: CV 14.63 (+0.50). Median GR cross-correlation 0.43 — insufficient signal. Code kept in `src/rogii/typewell_alignment.py`.
- A3b.1 Signed-log: CV 14.64 (+0.51). Residuals not heavy-tailed (bounded ±40, skew 0.74).
- A3b.2 Derivative model: CV 14.32 (+0.19). Integration error accumulation negates any slope-prediction benefit.

## Stage A4: Multi-Seed Averaging and Validation Refinements

Status: **Partially promoted.** Multi-seed ensemble [42,7,123] promoted as **R3** (CV 14.052, LB 12.177). StratifiedGroupKFold explored but not promoted. CatBoost and OOF stacking deferred to A5.

### A4a: Multi-Seed LightGBM — **Promoted → R3**

CV: 14.052 ± 0.868 (GroupKFold 5) vs single-seed 14.191. LB: 12.177 (−0.062 vs R2 12.239). 3 seeds [42,7,123] + Savgol w=31 p=2.

### A4b: StratifiedGroupKFold — Not promoted

Best config Strat(4,5) + multi-seed: CV 13.763 ± 1.523. Higher std reflects fold heterogeneity — potentially better CV/LB correlation but not yet validated on LB. Code kept behind `--cv-strategy stratified`.

### A4c: CatBoost + Stacking — Deferred to A5

Multi-model ensemble requires OOF infrastructure and diverse model types first.

---

## Stage A5: Architecture Diversity — Sequence Models + OOF Infrastructure + Multi-Strategy Ensemble

Status: **Active.** TCN v0, OOF persistence, diagnostics and Phase 1 dual normalization are implemented. Phase 1 still needs the full/screening training gate before promotion.

Goal: close the gap between R3 (LB 12.177) and leader solutions (LB 5.99–7.5) through architecture diversity. Analysis of top 20 leader solutions and public notebooks (plagiagia v2.8, stpeteishii TCN, Scott Weeden v13 seq-CNN, adarsh5harma Stacker v2) confirms three critical gaps.

### Root Cause Analysis (Leader Methods vs Our Approach)

| Method | Leaders (LB 5.99–7.5) | Our approach (LB 12.177) |
|---|---|---|
| **Model** | TCN, Seq-CNN, Ridge stacker, HGB | LightGBM only |
| **Sequence modeling** | Causal convolutions with diff/lag/roll on raw X,Y,Z,GR,MD | None — 18 hand-crafted tabular features |
| **Physics predictions** | Beam Search + Particle Filter as **independent predictors** | Beam Search as **features** for LightGBM → degraded (B1: CV 14.43) |
| **Blending** | 3+ strategies: ML + physics + deep learning | Single-strategy multi-seed averaging |
| **Feature approach** | Raw coordinates + automatic diff/lag/roll → ~50-70 features | 18 manually engineered features |

**Critical insight:** Beam Search and Particle Filter work, but NOT as tabular features for LightGBM. They must be independent predictors blended with ML model output. We confirmed this negatively — B1 beam features CV 14.43 (worse than R1). Leader approach: `0.5×ML + 0.25×Beam + 0.25×PF`.

### Dependency Graph

```
                         ┌─→ A5a (TCN) ──────────────┐
OOF-infrastructure (A5.0) ┤                             ├──→ A5c (Ensemble) ──→ A5d (HPO, optional)
                         ├─→ A5b (Beam/PF predictors) ─┤
                         └─→ R3 (LightGBM, done) ──────┘
```

A5a and A5b are independent. A5c depends on both + R3. A5d is final polish.

### Stage A5.0: OOF Infrastructure (gate for A5a-A5c)

Status: **Done.** OOF persistence and `--save-oof` are implemented for the current LightGBM/TCN training paths; future ensemble work may extend the contract.

Goal: standardize saving/loading of out-of-fold predictions across all model types so A5c can blend them without retraining.

Scope:
- `src/rogii/oof.py` — `save_oof(df, path, strategy_name)` / `load_oof(path)`.
  - OOF contract: `pd.DataFrame` columns `["well_id", "row_idx", "fold", "y_true", "y_pred", "baseline"]`.
  - Values in delta-space (residual from baseline), baseline=0 when `residual_target=False`.
  - Format: Parquet in `outputs/oof/<run_name>_<strategy>_oof.parquet`.
- `src/rogii/train.py` — `TrainResult` has `oof_df: pd.DataFrame | None`. CV loop collects `oof_rows`; after CV, converts to `_build_oof_per_well()` → `oof_df`.
- `scripts/run_train.py` — new `--save-oof` flag. Saves `train_result.oof_df` to `outputs/oof/`.

Primary files: `src/rogii/oof.py`, `src/rogii/train.py`, `scripts/run_train.py`, `tests/test_oof.py`.

Verification:
- `python -m pytest tests/test_oof.py` — OOF save/load roundtrip, column contract, delta-space preservation.
- `python scripts/run_train.py --data-dir data --seed 42 --n-splits 5 --residual-target --include-geometry --include-gr --save-oof` — file `outputs/oof/r1_lgbm_optimized_lgbm_oof.parquet` created.

### Stage A5a: TCN Sequence Model (highest priority)

**Status: Implemented (raw v0). Pre-hyperparameter plan active.**

Goal: implement Temporal Convolutional Network on raw coordinates with automatic diff/lag/roll features. Target: CV < 12 (vs R3 14.05).

**Why TCN:** causal convolutions (no look-ahead leakage), proven by leaders (stpeteishii TCN ~LB 7.0), GPU-friendly, handles variable-length sequences per well. TCN captures sequential signal (GR curve shape, trajectory undulations) that LightGBM on tabular features cannot.

**Current status:** TCN v0 is implemented and trains, but has critical gaps:
- Train uses post-PS-only sequences; predict uses full-well sliding windows — mismatch causes degraded predict.
- Per-well normalization removes absolute X/Y/Z/MD position (the primary geological signal).
- No R1-proven geometry/GR features as channels — TCN must re-discover them from raw coordinates.
- `tune_tcn.py` and `train_tcn()` may diverge in evaluation contract.
- Loss/LR/architecture not yet tuned — representation errors may mask learning capacity.

**Pre-Hyperparameter Plan: bring TCN to a honest, strong baseline before any HP search.**

Hyperparameters are deferred. The priority is fixing train/CV/predict contract, adding absolute coordinates as input, injecting R1-proven signals as sequence channels, unifying the evaluation path, and verifying the loss/objective. Only after a fixed control config improves over R3 does HP tuning resume.

#### Fixed Control Config (for representation verification, not tuning)

```
channels = [32, 64, 128]
window = 64
kernel_size = 5
lr = 3e-4
epochs = 10
patience = 4
batch_size = 2048 or 4096
stride = 4
dropout = 0.1
weight_decay = 1e-4
```

#### Phase 0: Diagnostics — **DONE. Results: `a5_tcn_control`, CV 15.03 (4 folds), see `docs/EXPERIMENT_LOG.md`.**

Ran fixed control config (`channels=[32,64,128] window=64 kernel=5 lr=3e-4`) + R3 LGBM OOF comparison. Infrastructure created: `src/rogii/diagnostics.py`, `scripts/diagnose_tcn.py`, `tests/test_diagnostics.py` (14 tests). OOF contract extended with `fold` column.

**Key findings (priority-changing):**

| Finding | Value | Implication |
|---|---|---|
| RMSE by `frac_after_ps` | **7.2 → 20.0** (monotonic rise) | Early post-PS rows are **easiest**, late are hardest. Phase 4 (context) is NOT the primary error source. |
| Prediction dispersion | **std_ratio 0.42** (severe flattening) | Per-well median 0.81 — flattening is **between-well**, not within-well. Per-well normalization kills absolute position signal. → **Phase 1 is #1 priority.** |
| TCN vs LGBM error correlation | **0.76** (moderate) | Blend gain −0.21 (small). Errors overlap — TCN and LGBM fail on same wells. Need TCN to get different signals first. |
| Fold spread | 14.62–15.98, fold 5 aborted | Phase 3 (unified eval) needed to fix abort + tune/train divergence. |

**Re-ranked priority order (data-driven): Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6.**

#### Phase 4: Fix Train/Predict Context

Currently train TCN learns only on post-PS sequences, but predict builds windows over full wells. This is a critical mismatch.

| Step | Change | Criterion |
|---|---|---|
| 1 | `WellSequence` stores full-well features, target only for post-PS rows | Pre-PS used as context, not target |
| 2 | Dataset builds window ending at target row | First post-PS target also trained |
| 3 | Short context: use left-padding or repeat first row | No skip of first `window-1` post-PS rows |
| 4 | `predict_tcn()` uses the same windowing contract | Train/CV/predict identical |
| 5 | Add tests for inclusion of first post-PS rows | CV covers all post-PS submission-like rows |

Files: `src/rogii/sequence_data.py`, `src/rogii/train.py`, `src/rogii/predict.py`, `tests/test_tcn_pipeline.py`, new/expanded `tests/test_sequence_data.py`.

#### Phase 1: Return Absolute Coordinates — **IMPLEMENTED. Pending training gate.**

Per-well normalization removes absolute position — the primary geological signal for this problem.

| Step | Change | Criterion |
|---|---|---|
| 1 | Keep per-well normalized features | Model sees trajectory shape |
| 2 | Add global-standardized absolute features | Model sees geological position |
| 3 | Global scaler fit only on train folds in CV | No leakage through validation wells |
| 4 | Final scaler fit on all train, saved in `tcn_metadata["x_scaler"]` | Kaggle predict reproducible |
| 5 | `predict_tcn()` applies `input_scaler` param | No train/predict drift |

**Implementation:** `WellSequence.X_abs` (T,4). `_fit_global_x_scaler()` fits `StandardScaler` on raw X/Y/Z/MD. `_scaled_sequences()` concatenates `[per_well_norm(65), global_scaled(4)]` → input_size=69. `build_model_payload` gets `tcn_input_scaler`, `tcn_input_size`. `run_train.py`/`run_predict.py` wired.

**Gate:** Training with `configs/a5_tcn.yaml` must produce `std_ratio > 0.7` and screening folds (0,3) < 14.62/14.82.

Files: `src/rogii/train.py`, `src/rogii/predict.py`, `src/rogii/model_io.py`, `src/rogii/sequence_data.py`, `scripts/run_train.py`, `scripts/run_predict.py`, `tests/test_tcn_pipeline.py`, `tests/test_model_io.py`.

#### Phase 2: Add Proven R1 Features as Sequence Channels

Do not force TCN to re-invent features that LightGBM already proved.

Add causal/test-available sequence channels:

| Feature group | Examples | Reason |
|---|---|---|
| PS geometry | `md_since_ps`, `frac_after_ps`, `dx/dy/dz_since_ps`, `dxy_since_ps` | Strong R1 features |
| Local trajectory | `dxdmd`, `dydmd`, `dzdmd` | Borehole direction |
| GR stable features | `gr_energy`, trailing `gr_roll_mean_101`, trailing `gr_roll_std_101` | Best GR-derived features |
| Anchor/baseline | `last_tvt_input`, optional `baseline_value` | Explicit residual target link |
| Position masks | `is_pre_ps`, `is_post_ps`, `rows_since_ps` | Model understands sequence phase |

**Leakage rules for Phase 2:**

| Risk | Rule |
|---|---|
| Post-PS `TVT` leakage | Never use `TVT` as feature |
| `TVT_input` | Only known pre-PS anchor/baseline/PS boundary |
| Global scalers | Fit only on train folds |
| GR rolling | Default trailing/causal, no future rows |
| Pre-PS context | Only input columns available in test |

Files: `src/rogii/sequence_features.py`, `src/rogii/features.py` as reference, `tests/test_sequence_features.py`.

#### Phase 3: Unified TCN Evaluation Path

`tune_tcn.py` and `train_tcn()` may currently diverge.

| Step | Change | Criterion |
|---|---|---|
| 1 | Extract common TCN CV/eval logic from tune/train | One contract: split/scaler/window/RMSE |
| 2 | `tune_tcn.py` only sets candidates, calls evaluator | Tuning matches final train |
| 3 | `run_train.py` uses same evaluator for CV | Kaggle-transfer honest |
| 4 | OOF saved for TCN | Enables postprocess/blend analysis |

Current status: `scripts/tune_tcn.py` has been aligned to fold-selectable 5-fold `GroupKFold` with dense validation RMSE monitoring, but a single shared evaluator for `tune_tcn.py` and `run_train.py` is still pending.

Files: `src/rogii/train.py`, possibly new `src/rogii/tcn_training.py`, `scripts/tune_tcn.py`, `scripts/run_train.py`.

#### Phase 5: Verify Objective/Loss

This is about training formulation, not hyperparameter search.

| Step | What to try | Criterion |
|---|---|---|
| 1 | Keep MSE as baseline | Compare with current |
| 2 | Add `SmoothL1Loss` as option | If fold 5 / outliers stabilize |
| 3 | Always evaluate RMSE in TVT/delta scale | Do not select by scaled MSE |
| 4 | Check prediction dispersion | No flattening |

Files: `src/rogii/train.py`, `scripts/tune_tcn.py`.

#### Phase 6: Postprocessing and Blend

TCN does not need to beat LGBM solo. More important: errors should be different.

| Step | What to do | Criterion |
|---|---|---|
| 1 | Save TCN OOF | `(well_id, row_idx, y_true, y_pred, baseline)` |
| 2 | Run Savgol/clipping on TCN OOF | Compare raw vs postproc |
| 3 | Compare LGBM vs TCN error correlation | Low correlation = blend worthwhile |
| 4 | Simple weighted blend LGBM+TCN OOF | Blend improves R3/R4 CV |
| 5 | Only after OOF improvement consider Kaggle | Do not submit weak solo TCN |

Files: `src/rogii/oof.py`, `src/rogii/postprocess.py`, future `src/rogii/ensemble.py`.

#### Implementation Order (data-driven after Phase 0)

1. **Phase 1:** Add dual input normalization: global absolute + per-well normalized. Fixes flattening (std_ratio 0.42 → target > 0.7).
2. **Phase 2:** Add R1 geometry/GR sequence channels. Fixes monotonic RMSE rise with `frac_after_ps`.
3. **Phase 3:** Unify `tune_tcn.py` and `train_tcn()` to one evaluator. Fixes fold 5 abort + eval consistency.
4. **Phase 4:** Fix full-well context and target-index dataset. Correctness fix, not primary error source.
5. **Phase 5:** Verify objective/loss. MSE baseline + SmoothL1Loss option.
6. **Phase 6:** Postprocessing + blend. Only after TCN gets different signals than LGBM.

Screening folds: `0, 3, 4` are used for quick validation; full 5-fold CV only after screening improves.

#### Gates (updated after Phase 0)

| Gate | Condition | Baseline |
|---|---|---|
| Entry | Fixed control config CV `15.03` (4 folds), std_ratio `0.42` | Current state |
| After Phase 1 | std_ratio > 0.7, screening folds `0,3` better than `14.62, 14.82` | `a5_tcn_control` |
| After Phase 2 | Screening folds close to `14.2–14.5` (LGBM R1 level) | — |
| Full CV | Run only if screening improved | — |
| Promotion | Full CV `<= 14.0` OR blend improves active baseline | R3 CV `14.05` |

#### What NOT to Do Yet

- Do not continue random LR/channel search on current TCN.
- Do not change primary validation from `GroupKFold`.
- Do not use centered/future GR features until explicitly decided that future logs are allowed.
- Do not submit TCN solo while full CV is worse than R3/A4.

#### Original A5a Architecture Sketch (partially superseded by phases above)

Architecture:
- Input: raw `(X, Y, Z, GR, MD)` → diff(1,2) + lag(1,2,3,5) + rolling(3,5,10) = ~50 features per timestep.
- Model: `TCNBlock` × N layers with dilation 2^i, residual connections, `Chomp1d` for causal guarantee. Channels: `[32, 64, 128]`, kernel=5, dropout=0.1 (per fixed control config).
- Target: delta from `last_tvt_input` (same residual approach as R3).
- Loss: MSE; `SmoothL1Loss` as option (Phase 5).
- Training: AdamW, early stopping patience=4.
- Validation: GroupKFold 5 by well_id.
- GPU: CUDA + AMP, batch=2048 or 4096.

Scope (existing, to be refined per phases):
- `src/rogii/sequence_features.py` — `build_sequence_features(df)`.
- `src/rogii/sequence_data.py` — `WellSequenceDataset`, windowing per well.
- `src/rogii/tcn_model.py` — `TCNModel`, causal conv1d blocks.
- `src/rogii/train.py` — `train_tcn()`.
- `src/rogii/predict.py` — `run_predict_tcn()`.
- `src/rogii/model_io.py` — TCN serialization.
- `scripts/run_train.py` — `--model-type tcn`.
- `scripts/run_predict.py` — auto-detect TCN payload.
- `configs/a5_tcn.yaml` — TCN config.
- `requirements.txt` — `torch`.

Verification:
- `python -m pytest tests/test_sequence_features.py` — causal, shape.
- `python -m pytest tests/test_tcn_model.py` — causal guarantee, forward pass.
- `python -m pytest tests/test_tcn_pipeline.py` — smoke train/predict/I/O.
- `python scripts/run_train.py --model-type tcn --config configs/a5_tcn.yaml --data-dir data --save-oof` — full 5-fold CV.

Promotion gate:
- Full CV RMSE `<= 14.0` OR blend improves active baseline.
- TCN is causal (verified by test).
- Runtime < 30 min full train on local GPU.

### Stage A5b: Physics-Based Standalone Predictors

Goal: implement Beam Search and Particle Filter as independent TVT predictors (NOT as tabular features), to be blended with ML model in A5c.

**Why:** B1 confirmed beam search adds signal (beam_std was #2 feature by importance) but LightGBM cannot extract it — it cannibalized X/Y/Z importance. As a standalone predictor blended with model output, beam/PF avoid this zero-sum redistribution. Leader ensemble (plagiagia v2.8): `0.5×HGB + 0.25×Beam + 0.25×PF`.

Scope:
- `src/rogii/beam_predictor.py` — `predict_beam(gr_hidden, tw_tvt, tw_gr, start_tvt, n_configs=7)` → `np.ndarray` of TVT predictions.
  - Reuses existing `_beam_jit` from `src/rogii/beam_search.py`.
  - Multiple configs (move_cost, emit_scale) → averaged consensus.
- `src/rogii/particle_filter.py` — `predict_pf(hw_df, tw_tvt, tw_gr, n_particles=250)` → `np.ndarray` of TVT predictions.
  - N=250 particles, velocity decay 0.993, GR likelihood sigma=30.
  - Causal only: prediction at row i uses only rows 0..i.
- OOF evaluation: both predictors evaluated on same GroupKFold 5 folds as ML model.
- `tests/test_physics_predictors.py` — causal verification, standalone RMSE on synthetic data.

Primary files: `src/rogii/beam_predictor.py`, `src/rogii/particle_filter.py`, `tests/test_physics_predictors.py`.

Promotion gate:
- Standalone predictor RMSE < 30 (vs model RMSE ~14).
- Beam/PF predictions causal (verified by test).
- Runtime < 5 min full train (CPU, Numba JIT).

### Stage A5c: Multi-Strategy Ensemble

Goal: blend R3 (LightGBM) + A5a (TCN) + A5b (Beam/PF) OOF predictions via Ridge stacking or weighted averaging.

Scope:
- `src/rogii/ensemble.py` — `stack_ridge(oof_paths, alpha=1.0)` / `blend_weighted(oof_paths, weights)`.
  - Load OOF DataFrames from A5.0 → align by (well_id, row_idx) → fit Ridge / grid search weights.
  - Output: ensemble weights / Ridge coefficients saved as JSON.
- `scripts/run_ensemble.py` — CLI: `--oof-dir outputs/oof/ --method ridge --output weights.json`.
- `scripts/run_predict.py` — `--ensemble-weights weights.json` to apply ensemble at prediction time.

Primary files: `src/rogii/ensemble.py`, `scripts/run_ensemble.py`, `tests/test_ensemble.py`.

Promotion gate:
- Ensemble CV < best single-model CV by > 0.3 RMSE.
- All OOF artifacts validated for no target leakage.

### Stage A5d: Hyperparameter Optimization (optional, post-A5c)

Goal: fine-tune winning architecture from A5a-A5c via Optuna.

Scope:
- `src/rogii/hpo.py` — Optuna studies for TCN (channels, lr, dropout, weight_decay) and LightGBM (num_leaves, subsample, reg_alpha/lambda).
- `configs/a5_hpo.yaml` — HPO config with Optuna trials, pruning, 3-fold CV for speed.

Dependencies: `optuna` (new, lightweight, CPU-friendly).

---

## Stage A6 (future): CatBoost + XGBoost for Ensemble Diversity

Status: **Deferred until A5c produces first ensemble.** If TCN + LightGBM ensemble yields CV < 11 but still > 10, add CatBoost (Ordered Boosting for grouped data) and XGBoost as additional ensemble members for diversity.

---

## Stage B1: Beam Search Stratigraphic Alignment

Status: **Rejected.** CV 14.43 (5-fold) vs R1 14.19 — worse by +0.24. `beam_std` is #2 feature (6.2% importance) but cannibalizes X/Y/Z spatial importance: X 17%→5.6%, Y 14.9%→5.5%, Z 12.5%→6.0%. Beam-only (no geometry/GR) scores CV 16.02 — worse than naive baseline (15.91). Same pattern as typewell/DTW/geology — typewell-referenced features redistribute without net gain. Code kept in `src/rogii/beam_search.py`, feature flag `include_beam` available for future compound experiments.

Goal: implement Numba JIT beam search along typewell GR to produce TVT alignment signals that close the gap between our tabular baseline (LB 12.2, CV 14.1) and top public solutions (LB 5.9–7.5).

### Why

All top public solutions (Roman Tamrazov sub-9, Ravaghi hill-climbing, Pilkwang EDA) use physics-based stratigraphic alignment (beam search, particle filters, NCC) rather than pure tabular features. Our tabular approach is confirmed saturated at LB 12.2 / CV 14.1. Beam search is the strongest single alignment signal observed in public code.

### Scope

- `src/rogii/beam_search.py` — Numba JIT beam search kernel (`_beam_jit`) + feature builder (`build_beam_features`).
- 7 beam configs with diverse move_cost / emit_scale / smooth_radius.
- ~19 features: per-config TVT estimates, consensus (avg cons+sm5), std, consensus differences at 11 offsets (−40..+40).
- `include_beam` flag in model payload v2 (`FEATURE_FLAG_KEYS`).
- Optional GPU for LightGBM training (auto-detect, fallback CPU). Beam search runs on CPU via Numba JIT.

### Dependencies

- `numba` (new, approved) — CPU JIT compilation for beam search.
- `scipy` (already approved).

### Primary files

- `src/rogii/beam_search.py` — new module
- `src/rogii/features.py` — `BEAM_FEATURES`, `include_beam` integration
- `src/rogii/model_io.py` — `include_beam` flag in `FEATURE_FLAG_KEYS`
- `scripts/run_train.py`, `scripts/run_predict.py` — `--include-beam` CLI flag
- `configs/b1_lgbm.yaml` — beam config
- `tests/test_beam_search.py` — 9 tests including causal verification
- `requirements.txt` — added `numba`

### Verification

- `python -m pytest tests/test_beam_search.py` — 9 tests including JIT compilation, causal construction, shape checks
- `python scripts/run_train.py --config configs/b1_lgbm.yaml --data-dir data`
- `python scripts/run_predict.py --data-dir data --model models/b1_lgbm.pkl --output outputs/submission.csv`
- `python scripts/validate_submission.py --submission outputs/submission.csv`
- Kaggle: `kaggle kernels push -p notebooks/kernels/b1-beam/` (with numba offline)

### Promotion gate

- CV improves by > 1.0 RMSE vs R1 (14.19) → target < 13.0.
- Beam search is causal (verified by `test_beam_causal_construction`).
- Numba JIT compiles without errors on Kaggle.
- Runtime < 10 min full train locally.

## Stage PrP3: Post-Processing Pipeline (Savgol Smoothing + TVT Clipping)

Goal: Improve CV/LB through per-well post-processing of predicted TVT sequences without changing model architecture or features.

Status: **Promoted as R2, superseded by R3.** Savgol w=31 p=2. LB 12.239 (−0.008 vs R1 12.247). R3 later added multi-seed LightGBM averaging and reached LB 12.177.

Results (OOF 5-fold CV, 3.78M rows, 773 wells):

| Config | OOF RMSE | vs Raw |
|--------|----------|--------|
| **Savgol w=31 p=2** | **14.2123** | **−0.0064** |
| Savgol w=25 p=2 | 14.2128 | −0.0059 |
| Savgol w=17 p=3 | 14.2135 | −0.0052 |
| Savgol w=11 p=2 | 14.2141 | −0.0046 |
| Savgol w=5 p=2 | 14.2154 | −0.0033 |
| Raw (no postproc) | 14.2187 | reference |
| Clip p0.1-p99.9 + Savgol w=31 | 14.2208 | +0.0021 |

TVT clip bounds (p0.1-p99.9): [9851.80, 12860.23]. Only 0.2% of data outside.
Per-well visualization: 3/3 wells improved, max raw jump 1.6-4.1 ft (noise, not geology).

Decision: **Savgol w=31 p=2 PROMOTED. TVT clipping REJECTED.** Savgol is now the recommended default post-processing step. Default params updated: window=17→31, polyorder=3→2.

Scope:

- `src/rogii/smoothing.py` — added `clip_predictions()`, `compute_tvt_clip_bounds()`, `apply_postprocessing()` (clip → smooth chain).
- `src/rogii/train.py` — CV loop collects per-well OOF predictions; `evaluate_postprocessing()` tests all Savgol window/polyorder/clip bound combos and ranks by OOF RMSE.
- `scripts/run_train.py` — added `--eval-postproc` flag.
- `scripts/run_predict.py` — added `--savgol-window`, `--savgol-polyorder`, `--tvt-clip` flags. Auto-detects clip bounds from model payload.
- `scripts/inspect_tvt_range.py` — new diagnostic script for TVT percentile analysis.
- `scripts/visualize_postproc.py` — new per-well visualization script with continuity checks.
- `src/rogii/model_io.py` — `clip_lower` / `clip_upper` stored in model payload v2.
- `tests/test_baseline.py` — 7 new clipping + apply_postprocessing tests (17 total).

Tunable parameters:

| Param | Default | Grid search |
|---|---|---|
| Savgol window | 17 | [5, 11, 17, 25, 31] |
| Savgol polyorder | 3 | [2, 3] |
| Clip bounds | p0.1–p99.9 from train TVT | none, p0.1-p99.9, p0.5-p99.5, p1-p99 |
| Order | clip → smooth | fixed |

Primary files:

- `src/rogii/smoothing.py`
- `src/rogii/train.py`
- `src/rogii/model_io.py`
- `scripts/run_train.py`
- `scripts/run_predict.py`
- `scripts/inspect_tvt_range.py`
- `scripts/visualize_postproc.py`
- `tests/test_baseline.py`

Verification:

- `python -m pytest tests/test_baseline.py` — 17 tests pass
- `python scripts/inspect_tvt_range.py --data-dir data` — prints TVT percentile table with suggested clip bounds
- `python scripts/run_train.py --config configs/baseline_lgbm.yaml --data-dir data --eval-postproc` — prints OOF post-processing evaluation table (top 15 configs sorted by RMSE)
- `python scripts/visualize_postproc.py --data-dir data --model models/r1_lgbm.pkl` — generates per-well comparison plots
- `python scripts/run_predict.py --data-dir data --model models/r1_lgbm.pkl --savgol-smooth --tvt-clip --output outputs/submission.csv` — generates clipped+smoothed submission

Promotion gate:

- OOF CV RMSE with best postproc config ≤ raw OOF CV RMSE (no degradation).
- Per-well visualization shows smoothing does not erase real formation boundaries.
- Continuity check: max jump ≤ 30 ft for ≥ 99% of adjacent points after smoothing.
- If CV improves: update active baseline config and LB-submit.
- If CV flat: keep code as optional pipeline, do not change active baseline.

## Stage PoP2: Prediction-Time 3-Strategy Blend (Model + Z-Physics + DTW GR Matching)

Goal: Blend raw model predictions with two physics/alignment-based TVT estimates at prediction time, without model retraining.

Status: **Rejected. OOF CV 53.94 (+39.72 vs Model 14.22). Both Z-physics (111.29) and DTW (145.06) are weak standalone predictors — blending them degrades the accurate model predictions.**

### Why

All top public solutions use multi-strategy blending (model + beam + particle filter in plagiagia v2.8; model + Z-physics + DTW in Scott Weeden v13). Our Z-Drift features as model input were flat (PrP2, CV 14.20). This stage applied Z-physics and DTW-GR matching as **independent predictors** and blended them with the model (median) — a fundamentally different information pathway from feature-engineering approaches.

### OOF Results (5-fold GroupKFold, 773 wells, 3.78M rows)

| Strategy | OOF RMSE | vs Model |
|---|---|---|
| **Model (R1, delta target)** | **14.2187** | reference |
| Z-physics only | 111.29 | +97.07 |
| DTW matching only | 145.06 | +130.84 |
| PoP2 Median (model+z+dtw) | 53.94 | +39.72 |
| Best weighted (0.8/0.1/0.1) | 22.79 | +8.57 |

### Root Cause

Z-physics (`TVT_z = Z + offset`) and DTW (sliding-window GR SAD against typewell) are too weak as standalone TVT predictors. While they correlate with the model (0.98-0.99), their individual RMSE is 7-10× worse than the model. Any blend weight > 0 on them drags the final prediction away from the accurate model output.

This follows the identical pattern as all previous physics/alignment experiments:
- PrP2 Z-Drift as features: CV 14.20, flat (+0.01)
- A3a DTW alignment: CV 14.63 (+0.50)
- B1 Beam Search: CV 14.43 (+0.24)
- **PoP2 Blend: CV 53.94 (+39.72)**

The model has already extracted all useful signal from spatial coordinates and GR. Physics-based corrections add no new information — they are less accurate approximations of the same signal.

### Decision

**Reject PoP2.** Code kept behind `--postprocess-blend` flag in `scripts/run_predict.py` for future ensemble experiments with stronger auxiliary strategies. Tabular ceiling at CV ~14.2 reconfirmed.

### Scope (preserved)

- `src/rogii/z_physics.py` — `apply_z_physics()`
- `src/rogii/gr_matcher.py` — `apply_dtw_matching()`
- `src/rogii/postprocess.py` — `apply_postprocess_blend()`
- `scripts/run_predict.py` — `--postprocess-blend`, `--blend-weights` flags
- `scripts/eval_pop2_oof.py` — OOF evaluation script
- `tests/test_postprocess.py` — 16 tests pass

## Stop Criteria

| Signal | Action |
|---|---|
| Spatial KNN CV drops to implausibly low RMSE such as `2-3` | Stop, disable spatial features, run leakage audit |
| CV improves but public LB sharply worsens | Do not promote; document CV/LB gap |
| Feature block degrades CV by more than `0.3` RMSE without useful diagnostics | Disable or keep as rejected experiment |
| Target transform improves transformed-space score but worsens TVT RMSE | Revert to plain residual target |
| Predicted TVT variance collapses | Revert target/model change and inspect per-well predictions |
| Runtime exceeds Kaggle limits | Simplify or defer the block |
| New dependency is unstable in Kaggle | Revert dependency and keep branch local until resolved |

## Rejected or Deferred Ideas

| Idea | Status | Reason |
|---|---|---|
| Simple Typewell V1 anchor residuals | Rejected | CV degraded and features were mostly `GR - const` redundancy |
| Trajectory features (A1) | Rejected | r >= 0.99 with geometry features, CV flat, LB worse |
| Spatial KNN (A2b) | Not promoted | Flat CV (−0.02), no leakage, code kept |
| DTW typewell alignment (A3a) | Rejected | CV +0.50, GR cross-correlation only 0.43 |
| Signed-log target (A3b.1) | Rejected | CV +0.51, residuals not heavy-tailed |
| Derivative target (A3b.2) | Rejected | CV +0.19, integration error accumulation |
| Geology v1 well-level (A4) | Rejected | CV +0.44, signal unstable across folds |
| Geology v2 per-row (A4) | Not promoted | CV −0.04 flat, code kept |
| Public saved artifacts / TabICL artifact stack | Rejected for clean mainline | Opaque external artifacts and heavy dependency path |
| Exact train/test coordinate overlap blend | Rejected for clean mainline | High leakage risk; diagnostics only |
| Particle filters | Deferred | Complex and stochastic |
| **Beam Search (B1)** | **Rejected** | CV 14.43 (worse than R1 14.19). beam_std — #2 feature but cannibalizes X/Y/Z. Beam-only CV 16.02. Typewell-GR alignment adds no net signal. |
| **Slope baseline (B2b)** | **Rejected** | Best CV 14.16 (slope_recent) flat vs R1. slope_md CV 284, wls CV 130. TVT-vs-MD trend does not extrapolate after PS — wells change direction. |
| **Formation Plane KNN (B3)** | **Rejected** | CV 14.99 (+0.80 vs R1). fp_knn_mean_dist #5 feature but cannibalizes X/Y/Z (35% importance from spatial coordinates). Well-level formation imputation via KNN adds no net signal. |
| **Z-Drift Physics (PrP2)** | **Not promoted** | CV 14.20 (flat vs R1 14.19, +0.01). 3 TVT-Z coupling features. Only offset_at_anchor is new signal; implied_tvt = Z+const, implied_tvt_resid = dz_since_ps (r=1.0). Fold inconsistency: fold 2 -0.80, fold 3 +0.62. Code kept behind include_z_drift flag. |
| **1D CNN / TCN sequence model** | **Active as A5a** | TCN v0, OOF, diagnostics and Phase 1 dual normalization are implemented; Phase 1 training gate pending. |
| **Beam Search + Particle Filter as predictors** | **Planned as A5b** | Leader analysis: beam/PF work as independent predictors blended with ML, NOT as tabular features. B1 proved tabular features degrade; blend approach is architecturally different. |
| **Multi-model ensemble (LGBM+TCN+Beam+PF)** | **Planned as A5c** | Requires reliable OOF artifacts and diverse strategies (A5a, A5b). Leader pattern: 3+ independent strategies blended. |
| **Multi-seed LGBM + CatBoost + stacking** | **Superseded by A5** | Multi-seed promoted as R3. CatBoost deferred to A6 — lower priority than TCN for architecture diversity. |
| **Optuna HPO** | **Planned as A5d** | Lightweight dependency. Execute after A5c if ensemble CV < 10 to squeeze final 0.1–0.5. |
| **PoP2: 3-Strategy Blend** | **Rejected** | CV 53.94 (+39.72 vs Model 14.22). Z-physics (111) and DTW (145) are weak predictors — blending degrades model. Code kept behind `--postprocess-blend` flag. |
