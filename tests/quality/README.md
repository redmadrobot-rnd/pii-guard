# Quality gate

Entity-level проверка качества детекции на размеченном датасете: micro-F1 по всем
типам, пороги по каждой категории и сравнение с базовой линией.

## Запуск

```bash
pip install -e ".[ner,dev]"
python -m spacy download ru_core_news_sm
pytest -q -s -m quality
```

По умолчанию движок вызывается **в процессе** — ни сервер, ни порт не нужны.
Если датасета или весов нет, тесты пропускаются, а не падают.

## Транспорт

`PII_GATE_TRANSPORT` выбирает, что именно проверяется:

| Значение | Что делает | Когда |
|---|---|---|
| `local` (по умолчанию) | вызывает `Engine` напрямую | проверка рабочего дерева |
| `http` | `POST /anonymize` на запущенный `pii_guard_server` | проверка собранного образа или развёрнутого экземпляра |

```bash
# проверить образ, а не рабочее дерево
docker build -t pii-guard . && docker run --rm -d -p 127.0.0.1:8080:8080 --name pg pii-guard
PII_GATE_TRANSPORT=http GUARD_API_URL=http://localhost:8080 pytest -q -s -m quality
docker rm -f pg
```

## Датасет

`test_data/test.xlsx`, формат IOB2, столбцы `text`, `tokens`, `ner_tags`,
`source`. **В git его нет намеренно**: он содержит реальные персональные данные,
а репозиторий публичный. Положите свой файл того же формата или укажите путь
через `PII_DATASET`.

## Структура

| Файл | Назначение |
|---|---|
| `test_quality_pii.py` | сами тесты: micro-F1, покрытие, регресс, пороги по категориям |
| `utils/config.py` | конфигурация из переменных окружения |
| `utils/transport.py` | выбор транспорта |
| `utils/local_client.py` | вызов движка в процессе |
| `utils/api_client.py` | HTTP-клиент |
| `utils/iob2_parser.py` | разбор gold-сущностей из IOB2 |
| `utils/evaluate.py` | оценка и логика гейта |
| `utils/reports.py` | JSON/CSV/PNG-артефакты |
| `baseline_result.json` | базовая линия для сравнения |
| `paper_score.py` | отдельный спановый скорер для таблиц статьи, к гейту отношения не имеет |
| `reports/` | результаты прогонов (в git не попадают) |

`paper_score.py` считает strict/soft на уровне спанов по JSON-дампам предсказаний
и запускается руками: `python tests/quality/paper_score.py <manifest.json>`.
Дампов в репозитории нет — в них лежат тексты оцениваемых наборов.

## Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `PII_GATE_TRANSPORT` | `local` | `local` или `http` |
| `PII_DATASET` | `tests/quality/test_data/test.xlsx` | путь к датасету |
| `PII_GATE_THRESHOLD` | `0.95` | порог общего micro-F1 |
| `PII_DEFAULT_CATEGORY_FLOOR` | `0.85` | единый порог по категории |
| `PII_CATEGORY_FLOOR_JSON` | пусто | точечные пороги, **JSON-объект**: `{"URL": 0.90}` |
| `PII_MAX_F1_DROP` | `0.02` | допустимая деградация против базовой линии |
| `PII_BASELINE_RESULT` | `tests/quality/baseline_result.json` | файл базовой линии |
| `PII_REQUIRE_SAME_DATASET` | `true` | требовать совпадение sha256 датасета при сравнении |
| `PII_REPORT_DIR` | `tests/quality/reports` | куда писать артефакты |
| `PII_ENABLE_PLOTS` | `true` | PNG-графики (нужен `matplotlib`) |
| `PII_LOG_PROGRESS` | `true` | построчный прогресс |

Только для `PII_GATE_TRANSPORT=http`:

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `GUARD_API_URL` | `http://localhost:8080` | адрес сервиса |
| `GUARD_API_KEY` | не задан | нужен, если на сервере задан `PII_GUARD_API_KEYS` |
| `GUARD_API_TIMEOUT` | `20` | таймаут запроса, секунды |
| `GUARD_API_RETRIES` | `1` | повторы при retriable-ошибках |
| `PII_TIMEOUT_POLICY` | `continue` | `continue` или `fail` |
| `PII_API_ERROR_RATE_THRESHOLD` | `0.05` | допустимая доля сетевых ошибок при `continue` |

Базовая линия лежит рядом с тестами, а не в `reports/`: последний каталог
игнорируется git, и базовая линия оттуда никогда не дошла бы до CI — гейт
регресса молча ничего не проверял бы на свежем клоне.
