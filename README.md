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

Команда для публичного клонирования:

```bash
git clone https://github.com/Lainterus1/ROGII_Kaggle_Competitions.git
```

## Текущий статус

Текущий лучший clean baseline: **A2a (DWT)**.

Готово:

- Data inventory CLI.
- Submission validator.
- Naive last-known-`TVT_input` baseline.
- Stage 4 LightGBM + `last_tvt_input` reference baseline: public LB RMSE `24.114`.
- R1 optimized LightGBM baseline: 18 features, residual target, CV RMSE `~14.19`, LB RMSE `12.247`.
- **A2a DWT baseline: 20 features (+2 causal GR DWT), residual target, CV RMSE `14.13`, LB pending.**
- `docs/HOW_IT_WORKS.md` — feature-by-feature explanation including DWT.
- Staged roadmap A0-A4. A1-A4 feature experiments completed, tabular ceiling confirmed at CV ~14.13.
- 106 tests, all green.

Следующее:

- Submit A2a to Kaggle — verify LB gap.
- Deferred: 1D CNN sequence model, multi-model ensemble (Stage A4+).

## Планируемая установка

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Команды

Реализованные команды:

```bash
python -m pytest tests
python scripts/make_data_inventory.py --data-dir data
python scripts/run_naive_baseline.py --data-dir data --output outputs/submission.csv
python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv

# R1 baseline (18 features)
python scripts/run_train.py --data-dir data --n-splits 5 --seed 42 --include-geometry --include-gr --residual-target --output-model models/r1_lgbm.pkl

# A2a baseline (20 features, DWT promoted)
python scripts/run_train.py --data-dir data --n-splits 5 --seed 42 --include-geometry --include-gr --include-gr-dwt --residual-target --output-model models/a2a_lgbm.pkl

# Predict (model payload carries feature flags and column order)
python scripts/run_predict.py --data-dir data --model models/a2a_lgbm.pkl --output outputs/submission.csv
python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv

# Optional feature flags (experiments, not in active baseline):
# --include-spatial   (OOF spatial KNN)
# --include-dtw       (DTW typewell alignment)
# --include-geology   (formation geology)
```

Модели, сохраненные новым train payload, сами хранят feature flags и список колонок. Для таких моделей `run_predict.py` не должен получать повторные `--include-*` флаги, если только это не legacy-модель без payload metadata.

Kaggle workflow:

- Код разрабатывается локально и пушится в GitHub или обновляется в Kaggle Dataset, в зависимости от текущего notebook workflow.
- Ноутбук соревнования остается thin runner: install/prepare repo, run train, run predict, run validator.
- `submission.csv` генерируется и валидируется в `/kaggle/working`.
- Фактический submit на Kaggle выполняет только пользователь вручную.
- После подтверждения пуша в Kaggle агент должен написать точные инструкции, какие строки/команды изменить в ноутбуке соревнования.

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
