# ROGII Kaggle Competitions

Воспроизводимый baseline-проект для Kaggle-соревнования `ROGII - Wellbore Geology Prediction`.

Страница соревнования: <https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview>

GitHub-репозиторий: <https://github.com/Lainterus1/ROGII_Kaggle_Competitions>

## Цель

Построить сильный и воспроизводимый baseline, который можно запускать локально на небольших выборках и на Kaggle для полного прогона по данным и ручной генерации submission-файла.

## Рабочий процесс

1. Разрабатывать локально.
2. Хранить код, конфиги и документацию в GitHub-репозитории `ROGII_Kaggle_Competitions`.
3. Запускать полные прогоны на Kaggle через тонкие runner-скрипты/ноутбуки.
4. Отправлять проверенный `submission.csv` вручную после подтверждения пользователя.
5. Вести текущие задачи, статусы и блокеры централизованно в Linear (`ROG-*`) через MCP; `docs/TASKS.md` остается историческим архивом.

Команда для публичного клонирования:

```bash
git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git
```

## Текущий статус

Текущий лучший baseline: **R3** — 3-seed LightGBM ensemble `[42, 7, 123]` на R1 18-feature set + Savgol `window=31, polyorder=2`.

Public LB: **12.177** (`53440641`). Local CV: `14.052 ± 0.868` (`GroupKFold`, 5 folds).

Готово:

- Data inventory CLI.
- Submission validator.
- Naive last-known-`TVT_input` baseline.
- Stage 4 LightGBM + `last_tvt_input` reference baseline: public LB RMSE `24.114`.
- R1 optimized LightGBM baseline: 18 features, residual target, CV RMSE `~14.19`, LB RMSE `12.247`.
- R2 post-processing baseline: R1 + Savgol `w=31 p=2`, OOF `14.2123`, LB `12.239`.
- R3 multi-seed baseline: 3 LightGBM seeds + Savgol, CV `14.052`, LB `12.177`; current active baseline.
- A1-B3 / PrP2 / PoP2 feature and physics experiments: implemented, evaluated, and mostly rejected or not promoted; tabular ceiling confirmed around CV `14.1` / LB `12.2`.
- A5 TCN path: raw TCN v0, OOF persistence, diagnostics, and Phase 2 dual normalization implemented; Phase 2 full training gate is pending.
- `docs/HOW_IT_WORKS.md` — feature-by-feature explanation of the active R3 feature set and rejected feature blocks.
- Test suite covers submission, leakage, feature engineering, post-processing, OOF, diagnostics and TCN contracts.

Следующее:

- Run A5 TCN Phase 2 full/screening training gate: target `std_ratio > 0.7` and screening folds better than the Phase 0 control.
- If Phase 2 passes, implement A5 Phase 3 R1 sequence channels and Phase 4 unified TCN evaluation path.
- Keep R3 LightGBM + Savgol as the Kaggle fallback until a new candidate beats it on CV/OOF and passes submission validation.

## Планируемая установка

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Команды

Реализованные команды:

```bash
# Fast local contract/unit loop. Skips expensive training smoke tests and retained experimental feature checks.
python -m pytest tests -m "not slow and not integration and not experimental"

# Full regression before handing off or changing model/validation/submission behavior.
python -m pytest tests
python scripts/make_data_inventory.py --data-dir data
python scripts/run_naive_baseline.py --data-dir data --output outputs/submission.csv
python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv

# R1 baseline (18 features, single seed)
python scripts/run_train.py --data-dir data --n-splits 5 --seed 42 --include-geometry --include-gr --residual-target --output-model models/r1_lgbm.pkl

# R3 active baseline (3 seeds + R1 features)
python scripts/run_train.py --config configs/a4_multiseed.yaml --data-dir data --output-model models/a4_multiseed.pkl

# Predict (model payload carries feature flags and column order)
python scripts/run_predict.py --data-dir data --model models/a4_multiseed.pkl --savgol-smooth --output outputs/submission.csv
python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv

# A5 TCN control / Phase 2 candidate
python scripts/run_train.py --config configs/a5_tcn.yaml --data-dir data --save-oof --output-model models/a5_tcn.pkl

# Optional feature flags (experiments, not in active baseline):
# --include-gr-dwt           (A2a DWT; LB rejected)
# --include-spatial          (OOF spatial KNN; flat CV)
# --include-dtw              (DTW typewell alignment; rejected)
# --include-geology          (formation geology; rejected/not promoted)
# --include-beam             (beam features; rejected as tabular features)
# --include-formation-plane  (formation-plane KNN; rejected)
# --include-z-drift          (TVT-Z drift features; not promoted)
```

Модели, сохраненные новым train payload, сами хранят feature flags и список колонок. Для таких моделей `run_predict.py` не должен получать повторные `--include-*` флаги, если только это не legacy-модель без payload metadata.

Kaggle workflow:

- Stable R1 fallback path uses `daniilgonchar/00-rogii-inference-r1` with internet OFF, `rogii-repo-v2`, `rogii-models-v2` and `notebooks/kernel-metadata.json`.
- Current R3 candidate path uses `notebooks/kernels/a4-multiseed/`, `daniilgonchar/00-rogii-inference-a4-v3`, `daniilgonchar/rogii-repo-a4` and `daniilgonchar/rogii-models-a4-multiseed`.
- Candidate builds must use candidate-specific repo/model/dependency datasets. If source code changed, create a new repo dataset via `kagglehub.dataset_upload()`; do not update shared fallback datasets.
- Push the intended kernel folder with `kaggle kernels push -p <kernel-folder>`.
- Validate `/kaggle/working/submission.csv` from kernel output before submission.
- Kaggle submission requires explicit user approval. After approval, submit the validated kernel version: `kaggle competitions submit -c rogii-wellbore-geology-prediction -k <kernel> -v <version> -f submission.csv -m "..."`.

## Документация

Сначала читать:

- `docs/PROJECT_CONTEXT.md`
- `docs/CONTEXT_MAP.md`
- `docs/ARCHITECTURE.md`
- `docs/DECISIONS.md`
- `docs/HOW_IT_WORKS.md` — как работает модель (для новичков)

## Данные и артефакты

Не коммитить Kaggle-данные, секреты, обученные модели, submissions или MLflow artifact stores. Runtime-файлы должны находиться в игнорируемых директориях: `data/`, `outputs/`, `models/`, `submissions/`, `mlruns/`.

Локальные данные ожидаются в `data/`. Kaggle-данные ожидаются в `/kaggle/input`; outputs на Kaggle нужно записывать в `/kaggle/working`.
