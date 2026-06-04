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

Еще не готово:

- Инспекция Kaggle-данных.
- Подтверждение официальной метрики.
- Подтверждение submission-контракта.
- Naive/model baselines.

## Планируемая установка

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Планируемые команды

Эти команды являются placeholders до реализации пайплайна на следующих шагах:

```bash
pytest tests/
python scripts/make_data_inventory.py --config configs/paths.local.yaml.example
python scripts/run_smoke.py --config configs/baseline_lgbm.yaml
python scripts/run_naive_baseline.py --config configs/baseline_naive.yaml
python scripts/run_train.py --config configs/baseline_lgbm.yaml --env local
python scripts/validate_submission.py --submission outputs/submission.csv
```

## Документация

Сначала читать:

- `ROGII_PROJECT_INTAKE_DOSSIER.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/CONTEXT_MAP.md`
- `docs/ARCHITECTURE.md`
- `docs/DECISIONS.md`

## Данные и артефакты

Не коммитить Kaggle-данные, секреты, обученные модели, submissions или MLflow artifact stores. Runtime-файлы должны находиться в игнорируемых директориях: `data/`, `outputs/`, `models/`, `submissions/`, `mlruns/`.

Локальные данные ожидаются в `data/`. Kaggle-данные ожидаются в `/kaggle/input`; outputs на Kaggle нужно записывать в `/kaggle/working`.
