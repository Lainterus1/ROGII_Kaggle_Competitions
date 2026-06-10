---
name: linear-issue-manager
description: Use when creating, updating, or formatting Linear issues for the ROGII project. Covers experiment/feature/bug/docs/infra templates, label conventions, status transitions, GitHub linking rules, and documentation impact.
---

# Linear Issue Manager

## When to use

Use this skill whenever you need to:

- Create a new Linear issue for the ROGII project.
- Update issue status (transition between Backlog → Todo → In Progress → In Review → Done / Canceled).
- Format an issue description for an experiment, feature, bug, docs change, or infra task.
- Link a Linear issue to a GitHub branch, PR, or project documentation such as ADRs or `docs/EXPERIMENT_LOG.md`.
- Close an experiment with a promotion/rejection decision.

Do NOT use this skill:

- For general project navigation — use `docs/CONTEXT_MAP.md` instead.
- For Kaggle submission workflows — use `kaggle-candidate-build` or `kaggle-runner` instead.
- For code reviews — use `code-review` instead.

## Source-of-truth

| Need | File |
|---|---|
| Team and project IDs | `.agents/skills/linear-issue-manager/ids.md` (auto-generated) |
| Project context | `docs/PROJECT_CONTEXT.md` |
| Current backlog and status | Linear MCP (`ROG-*` issues) |
| Historical task archive | `docs/TASKS.md` |
| Architecture decisions | `docs/DECISIONS.md` |
| Roadmap stages | `docs/ROADMAP.md` |
| Active baseline | Read from `AGENTS.md`, `docs/ROADMAP.md` and `docs/BASELINE_PLAN.md`; do not hardcode stale values |

## Language

**All Linear issues MUST be written in Russian.** The team is Russian-speaking. Titles, descriptions, hypotheses, approach steps, and result blocks — all in Russian. Only technical identifiers (file paths, CLI flags, variable names, CV/LB numbers) stay in English.

## Important: ask, don't improvise

**If any required field in a template cannot be filled from the conversation context alone, ask the user before creating the issue.** Never invent:

- Hypothesis details, expected CV values, or approach steps for experiments.
- Feature scope, acceptance criteria, or impacted files when the user hasn't specified them.
- Root cause or fix for bugs — ask for reproduction steps or error messages.
- Priority unless it is obvious from the project context (e.g., a5-tcn Phase 3 = high).

Minimal viable issues are fine — a clear title with one-line description is better than a hallucinated template. Ask: "What's the hypothesis?" / "Which files are affected?" / "What's the expected CV threshold?"

## Project constants (hardcoded)

These values are stable and should be reused without asking:

| Constant | Value | Where used |
|---|---|---|
| Team ID | `2905a8ed-1d01-41f5-acbd-f6b77dc771b6` | `linear_linear_createIssue` |
| Team key | `ROG` | Branch names, PR titles |
| Team name | `MLrs` | Context only |
| Project ID | `ce016362-1b1c-49f5-975c-704323dcb377` | `linear_linear_createIssue` |
| Project name | `ROGII - Wellbore Geology Prediction` | Context only |
| Default assignee source | `linear_linear_getViewer().id` | `assigneeId` for issues created or worked by the agent |
| Active baseline source | `AGENTS.md`, `docs/ROADMAP.md`, `docs/BASELINE_PLAN.md` | Experiment gate comparisons |

## Issue types and templates

### 1. Эксперимент (Experiment)

Используется для ML-гипотез: новые фичи, архитектурные изменения, подбор гиперпараметров, постобработка, блендинг. Каждый эксперимент обязан содержать гипотезу, gate-критерии и финальный результат с решением.

**Labels:** `experiment` + компонентный лейбл (`a5-tcn`, `lightgbm`, `kaggle`, `validation`, `postproc`) + `priority:high|medium|low`

**Template:**

```markdown
## Гипотеза
<Одно предложение: что улучшит CV/LB и почему. Спросить пользователя, если неясно.>

## Подход
- Модель/фичи: <тип модели, feature flags, конфиг>
- Конфиг: <путь к yaml, например configs/baseline_lgbm.yaml>
- Валидация: <n-splits, стратегия, например 5-fold GroupKFold>

## Gate
- [ ] CV лучше текущего promoted baseline из `docs/ROADMAP.md` / `docs/BASELINE_PLAN.md` на всех фолдах
- [ ] Leakage review пройден (`leakage-review` skill)
- [ ] OOF сохранен (`--save-oof`)
- [ ] `python -m pytest tests` — пройдены

## Результат (заполнить после эксперимента)
- CV: XX.XX (±X.XX) vs текущий promoted baseline
- LB: XX.XX (если submitted)
- OOF path: `outputs/oof/...`
- Решение: Promote | Reject | Keep code (not promoted)
- ADR: ADR-XXX (если создан)
```

**После завершения:**

| Исход | Статус задачи | Что обновить в доках |
|---|---|---|
| Promote | `Done` | Linear issue result, `DECISIONS.md` (ADR), `ROADMAP.md` если стадия промоутирована, `EXPERIMENT_LOG.md` |
| Reject / not promoted | `Canceled` | Linear issue result, `DECISIONS.md` (ADR) если решение значимое, `EXPERIMENT_LOG.md` |

При закрытии отвергнутого эксперимента добавить блок результата в описание и перевести в `Canceled`.

### 2. Фича (Feature)

Используется для новой функциональности: скрипты, модули, CLI-флаги, шаги пайплайна, которые НЕ являются экспериментами.

**Labels:** `Feature` (workspace label) + компонентный лейбл + `priority:high|medium|low`

**Template:**

```markdown
## Что
<Что делаем. Одно предложение.>

## Зачем
<Что это разблокирует — заблокированный эксперимент, недостающий шаг пайплайна, требование Kaggle.>

## Критерии
- [ ] `<file.py>` реализован
- [ ] Тесты: `tests/test_<...>.py` добавлены/обновлены
- [ ] CLI флаг (если нужен): `--<flag>`
- [ ] `python -m pytest tests` — пройдены

## Файлы (ожидаемые)
- `src/rogii/<...>.py`
- `scripts/<...>.py`
- `tests/<...>.py`
```

### 3. Баг (Bug)

Используется для сломанного поведения, регрессий, падающих тестов, runtime-ошибок.

**Labels:** `Bug` (workspace label) + `priority:high|medium|low`

**Template:**

```markdown
## Что сломалось
<Одно предложение: что упало.>

## Воспроизведение
```
python scripts/<...>.py --<args>
```
<Сообщение об ошибке / трейсбек если есть.>

## Ожидаемое поведение
<Что должно происходить вместо этого.>

## Исправление
- [ ] `<file>:<line>` исправлено
- [ ] Regression test добавлен
```

### 4. Документация (Docs)

Используется для изменений только в документации: новые ADR, обновления source-of-truth docs, README.

**Labels:** `docs` + `priority:low` (обычно)

**Template:**

```markdown
## Документ
`docs/<file>.md`

## Изменение
<Что изменилось и почему.>

## Связано
- ADR: ADR-XXX (если применимо)
- Задача: ROG-<id> (если связанный эксперимент/фича)
```

### 5. Инфраструктура (Infra)

Используется для инструментов, конфигов, Kaggle runner, CI, зависимостей.

**Labels:** `infra` + `priority:high|medium|low`

**Template:**

```markdown
## Инструмент/Конфиг
<Что меняется: Kaggle runner, зависимости, пути, и т.д.>

## Влияние
<Что затронуто. Ломает ли существующие workflow?>

## Критерии
- [ ] Проверено локально / на Kaggle
- [ ] Текущий Kaggle fallback не сломан
- [ ] `python -m pytest tests` — пройдены
```

## Переходы статусов

Всегда используй `stateId` из `linear_linear_getWorkflowStates` при смене статуса. Статусы команды MLrs:

| Статус | ID | Когда использовать |
|---|---|---|
| Backlog | `3f266f64-a7dc-48e5-9942-1ee2943c6024` | Идея, еще не приоритизирована |
| Todo | `970c9469-ddb2-4400-87fa-4cb8e438cd50` | Взято в текущую работу |
| In Progress | `33ddba28-5dd3-4f5a-9153-9164797d4ed7` | Активно делается |
| In Review | `cf175929-6a3c-4a37-9063-336747a1e0b0` | PR открыт, ждет ревью |
| Done | `0830cde7-af33-49b5-a3a9-85588af71aee` | Завершено / промоутированный эксперимент |
| Canceled | `85a2a4ab-910d-48f4-a00b-49a2c085596e` | Отвергнутый эксперимент, wontfix, дубликат |

**Правила:**

- Только ОДНА задача `In Progress` на человека одновременно.
- Если агент создает или берет задачу в работу, назначай исполнителем текущего authenticated Linear user из `linear_linear_getViewer`, если пользователь явно не указал другого исполнителя.
- Отвергнутые эксперименты ОБЯЗАНЫ идти в `Canceled`, никогда не в `Done`.
- Не переводить в `In Review` без реально открытого PR.

## Правила линковки с GitHub

При создании веток и PR для задачи Linear:

| Элемент | Шаблон | Пример |
|---|---|---|
| Ветка для фичи | `feature/ROG-<id>-<краткое-описание>` | `feature/ROG-12-add-savgol-cli` |
| Ветка для фикса | `fix/ROG-<id>-<краткое-описание>` | `fix/ROG-15-oof-nan-fix` |
| Заголовок PR | `[ROG-<id>] <описание>` | `[ROG-12] Add --savgol-window flag to run_predict.py` |
| Commit message | `ROG-<id>: <что сделано>` | `ROG-12: wire Savgol smoothing into predict pipeline` |

Linear автоматически линкует ветки и PR, обнаруживая `ROG-<id>` в названии. Ручное связывание не требуется.

## Интеграция с документацией

После создания или закрытия задачи обнови саму Linear issue и проверь, нужно ли обновить source-of-truth docs. Не дублируй статус задачи в `docs/TASKS.md`.

| Изменение | Что обновить |
|---|---|
| Эксперимент промоутирован | Linear issue result, `docs/DECISIONS.md` (новый ADR, ссылка на `ROG-<id>`), `docs/ROADMAP.md` если стадия промоутирована, `docs/EXPERIMENT_LOG.md` |
| Эксперимент отвергнут | Linear issue result, `docs/DECISIONS.md` если решение значимое, `docs/EXPERIMENT_LOG.md` если был meaningful run |
| Эксперимент: код оставлен, не промоутирован | Linear issue result; docs only if reusable code, risk or experiment record changed |
| Фича готова | Linear issue result; docs only if contract, behavior or user-facing workflow changed |
| Баг исправлен | Обычно ничего; `docs/KNOWN_ISSUES.md` если баг там отслеживался |
| Изменение архитектуры / зависимостей | `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` |
| Изменение валидации / метрики | `docs/VALIDATION_STRATEGY.md` или `docs/METRICS.md` |
| Обнаружен новый блокер | `docs/KNOWN_ISSUES.md` |

Используй скилл `documentation-maintenance` когда нужно координированное обновление нескольких документов.

## Процедура создания задачи

1. **Спроси пользователя** о любой недостающей обязательной информации (см. «Спрашивай, а не импровизируй» выше). НЕ создавай задачу с выдуманным содержимым.
2. **Выбери подходящий шаблон** (Эксперимент / Фича / Баг / Документация / Инфраструктура).
3. **Заполни шаблон** информацией из диалога.
4. **Выбери лейблы** — минимум: тип + приоритет. Добавь компонентные лейблы, если scope ясен.
5. **Получи исполнителя** через `linear_linear_getViewer`; используй `viewer.id` как `assigneeId`. Если viewer недоступен, спроси пользователя, кого назначить.
6. **Создай задачу** через `linear_linear_createIssue` с `teamId` = `2905a8ed-1d01-41f5-acbd-f6b77dc771b6`, `projectId` = `ce016362-1b1c-49f5-975c-704323dcb377` и `assigneeId` = `viewer.id`.
7. **Сообщи результат** — URL задачи (`https://linear.app/lainterus/issue/ROG-<id>/...`) и предлагаемое имя ветки.

## Процедура обновления / закрытия задачи

1. **Получи текущее состояние** через `linear_linear_getIssueById`.
2. **Проверь исполнителя**: если задача выполняется агентом и assignee пустой или отличается без явного указания пользователя, получи `viewer.id` через `linear_linear_getViewer` и назначь его через `linear_linear_assignIssue`.
3. **Обнови описание** (например, добавь результат эксперимента) через `linear_linear_updateIssue` при необходимости.
4. **Смени статус** через `linear_linear_updateIssue` с корректным `stateId`.
5. **Проверь влияние на документацию** (см. таблицу интеграции выше).
6. **Не обновляй `docs/TASKS.md`** для текущей задачи. Этот файл является историческим архивом.

## Fallback: Linear GraphQL через PowerShell

Используй этот fallback, если MCP `linear_linear_createIssue` или `linear_linear_updateIssue` падает на системной валидации optional UUID-полей, например `projectMilestoneId must be a UUID`, `projectMilestoneId cannot be used without projectId`, `parentId must be a valid UUID`, хотя эти поля не нужны задаче.

Также используй этот способ, если кириллица в Linear превращается в `????`: PowerShell должен отправлять JSON body как UTF-8 bytes.

Правила безопасности:

- Не выводи значение `$env:LINEAR_API_TOKEN` в терминал, логи, issue или docs.
- Можно проверять только факт наличия токена: `if ($env:LINEAR_API_TOKEN) { 'LINEAR_API_TOKEN=set' }`.
- Перед fallback получи `viewer.id` через `linear_linear_getViewer` и используй его как `assigneeId`, если пользователь не указал другого исполнителя.
- Перед fallback получи актуальные `stateId` через `linear_linear_getWorkflowStates` и `labelIds` через `linear_linear_getLabels`; не копируй их из памяти, если есть сомнение.
- Не передавай пустые optional UUID-поля. В GraphQL `input` должны быть только реально нужные поля.

Проверенный шаблон создания задачи:

```powershell
$query = 'mutation IssueCreate($input: IssueCreateInput!) { issueCreate(input: $input) { success issue { id identifier url title assignee { displayName } state { name } } } }'
$description = @'
## Документ
`AGENTS.md`, `.agents/skills/linear-issue-manager/SKILL.md`

## Изменение
<Что меняется и зачем.>

## Критерии
- [ ] <Проверяемый критерий>
'@
$variables = @{
  input = @{
    title = '<Короткий русский заголовок>'
    description = $description
    teamId = '2905a8ed-1d01-41f5-acbd-f6b77dc771b6'
    projectId = 'ce016362-1b1c-49f5-975c-704323dcb377'
    stateId = '<In Progress или другой актуальный stateId>'
    assigneeId = '<viewer.id из linear_linear_getViewer>'
    priority = 4
    labelIds = @('<docs/feature/bug/etc labelId>', '<priority labelId>')
  }
}
$body = @{ query = $query; variables = $variables } | ConvertTo-Json -Depth 10
$headers = @{ Authorization = $env:LINEAR_API_TOKEN; 'Content-Type' = 'application/json; charset=utf-8' }
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$result = Invoke-RestMethod -Method Post -Uri 'https://api.linear.app/graphql' -Headers $headers -Body $bytes
if ($result.errors) { $result.errors | ConvertTo-Json -Depth 6; exit 1 }
$issue = $result.data.issueCreate.issue
"created $($issue.identifier) $($issue.state.name) assignee=$($issue.assignee.displayName) $($issue.url)"
```

Проверенный шаблон обновления описания, исполнителя и статуса:

```powershell
$query = 'mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) { issueUpdate(id: $id, input: $input) { success issue { identifier url state { name } assignee { displayName } } } }'
$description = @'
## Результат
- [x] <Что сделано>

## Проверки
- `<command>` - пройдено.
'@
$variables = @{
  id = '<Linear issue UUID, не ROG-номер>'
  input = @{
    description = $description
    stateId = '<Done/Canceled/In Progress stateId>'
    assigneeId = '<viewer.id из linear_linear_getViewer>'
  }
}
$body = @{ query = $query; variables = $variables } | ConvertTo-Json -Depth 10
$headers = @{ Authorization = $env:LINEAR_API_TOKEN; 'Content-Type' = 'application/json; charset=utf-8' }
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$result = Invoke-RestMethod -Method Post -Uri 'https://api.linear.app/graphql' -Headers $headers -Body $bytes
if ($result.errors) { $result.errors | ConvertTo-Json -Depth 6; exit 1 }
$issue = $result.data.issueUpdate.issue
"updated $($issue.identifier) $($issue.state.name) assignee=$($issue.assignee.displayName) $($issue.url)"
```

После fallback обязательно проверь задачу через MCP `linear_linear_getIssueById` по `ROG-<id>`, чтобы подтвердить title, description, assignee, labels, project и status.

## Чеклист валидации

Перед завершением создания/обновления задачи:

- [ ] Заголовок конкретный и действенный (не «Починить всё» или «Улучшить модель»).
- [ ] Описание соответствует шаблону для типа задачи.
- [ ] Обязательные лейблы проставлены (тип + приоритет + компонент если применимо).
- [ ] Исполнитель установлен на authenticated Linear user из `linear_linear_getViewer`, если пользователь явно не указал другого исполнителя.
- [ ] Смена статуса корректна согласно таблице статусов.
- [ ] Для экспериментов: гипотеза и gate-критерии явно указаны.
- [ ] Для отвергнутых экспериментов: статус `Canceled`, НЕ `Done`.
- [ ] Соблюдены правила именования веток/PR для GitHub.
- [ ] Проверено влияние на документацию (даже если обновление не нужно — укажи почему).
- [ ] Ни одно поле не выдумано — если информации не хватало, пользователь был спрошен.

## Запрещенные действия

- Не создавай задачу с выдуманным/сгенерированным содержимым. Спроси пользователя, если обязательные поля не заполнены.
- Не закрывай отвергнутый эксперимент как `Done`. Всегда `Canceled`.
- Не используй ID из памяти — всегда получай актуальный `stateId` или `labelId` через Linear API.
- Не меняй константы проекта (team ID, project ID) без явного указания пользователя.
- Не создавай задачи без шаблона. Каждая задача обязана следовать одному из пяти типов выше.
- Не обновляй `docs/TASKS.md` для текущей работы. Старые выполненные задачи не переносить в Linear.
- Не сабмитай в Kaggle из workflow Linear-задачи — submission требует явного одобрения через `kaggle-candidate-build`.
