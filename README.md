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

Текущий лучший clean baseline: R1 optimized.

Готово:

- Data inventory CLI.
- Submission validator.
- Naive last-known-`TVT_input` baseline.
- Stage 4 LightGBM + `last_tvt_input` reference baseline: public LB RMSE `24.114`.
- R1 optimized LightGBM baseline: 18 features, residual target `TVT - last_tvt_input`, CV RMSE `~14.19`, public LB RMSE `12.247`.
- `docs/HOW_IT_WORKS.md` with feature-by-feature explanation.
- New staged roadmap A0-A4 for trajectory features, DWT, strict OOF spatial KNN, DTW, target engineering and structural blending.

Следующее:

- Stage A0: sync CLI/config/README/model-payload contracts before implementing new feature blocks.
- Stage A1: implement spatial kinematics and trajectory geometry features.
- Kaggle submissions remain manual only. After a push intended for Kaggle, the agent should provide exact competition notebook edit instructions.

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
python scripts/run_train.py --config configs/baseline_lgbm.yaml --data-dir data
python scripts/run_train.py --data-dir data --n-splits 5 --seed 42 --include-geometry --include-gr --residual-target --output-model models/r1_lgbm.pkl
python scripts/run_predict.py --data-dir data --model models/r1_lgbm.pkl --output outputs/submission.csv
python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv
```

Модели, сохраненные новым train payload, сами хранят feature flags и список колонок. Для таких моделей `run_predict.py` не должен получать повторные `--include-*` флаги, если только это не legacy-модель без payload metadata.

Планируемые команды будут добавляться по мере реализации этапов A1-A4. Не использовать несуществующие флаги из плана до соответствующей реализации в `scripts/`.

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
