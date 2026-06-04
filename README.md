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

Bootstrap проекта в процессе.

Готово:

- Документ с контекстом проекта.
- Skeleton source-of-truth документации.
- Начальное решение по balanced-архитектуре.
- Начальный skeleton проекта.
- Data inventory CLI.
- Submission validator.
- Naive last-known-`TVT_input` baseline.

Еще не готово:

- First ML baseline.
- MLflow model run logging.

## Планируемая установка

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Команды

Реализованные команды:

```bash
pytest tests/
python scripts/make_data_inventory.py --data-dir data
python scripts/run_naive_baseline.py --data-dir data --output outputs/submission.csv
python scripts/validate_submission.py --data-dir data --submission outputs/submission.csv
```

Планируемые команды для следующих шагов:

```bash
python scripts/run_smoke.py --config configs/baseline_lgbm.yaml
python scripts/run_train.py --config configs/baseline_lgbm.yaml --env local
```

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
