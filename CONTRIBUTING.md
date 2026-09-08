# Contributing

## Локальная сборка

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[ner,server,dev]"
python -m spacy download ru_core_news_sm
```

Без `[ner]` тоже рабочее окружение — юнит-тесты весов не требуют.

## Тесты

```bash
pytest tests/unit               # секунды, без torch/spaCy/весов
pytest -m quality               # нужны веса и датасеты
ruff check . --fix
```

Юнит-сьют обязан оставаться независимым от тяжёлых зависимостей. Если тест
требует torch, spaCy или весов — ему место в `tests/quality`, помеченным
`@pytest.mark.quality`.

## Добавить тип сущности

Ветка правил построена на автообнаружении: положите файл в `src/pii_guard/entities/`
и повесьте декоратор. Больше ничего править не нужно — ни реестр, ни движок.

```python
# src/pii_guard/entities/my_entity.py
from pii_guard.framework.base import BaseEntityClassifier, register_entity

@register_entity
class MyEntityClassifier(BaseEntityClassifier):
    entity_type = "MY_ENTITY"
    ...
```

Для регулярных выражений — `@register_regex_entity`, фабрика получает
`(analyzer, language)`.

К новому типу нужны:

1. тест в `tests/unit/` с валидными и **невалидными** значениями — важнее всего,
   что классификатор отвергает похожий мусор;
2. запись в `DEFAULT_ENTITIES` (`src/pii_guard/config.py`), если тип должен
   маскироваться по умолчанию;
3. упоминание в перечне типов во вводном абзаце `README.md` и в
   `docs/architecture.md` (уровень 3), если тип виден снаружи.

## Границы, которые лучше не ломать

**Правила побеждают модель.** При перекрытии спанов приоритет у детерминированной
ветки (`resolve_ml_vs_rules_conflicts`). Контрольная сумма — доказательство,
вероятность модели — нет.

**Смещения относятся к `normalized_text`.** Любая правка препроцессоров должна
сохранять корректность переноса спанов, иначе поедут границы у всех потребителей.

**Ядро не знает про HTTP.** `src/pii_guard/` не импортирует FastAPI. Переменные
окружения читаются в `config.py`; единственное исключение — `PII_GUARD_DEVICE` в
`ner/recognizer.py`, чтобы выбор устройства работал и при прямом создании
распознавателя, без `Config`.

**Инференс сериализован.** spaCy и torch не рассчитаны на параллельный вызов из
потоков. Масштабирование — репликами процессов.

## Обновление весов

Смена `DEFAULT_NER_REVISION` в `src/pii_guard/models.py` — отдельный PR, к
которому прикладываются метрики quality-gate до и после. Без этого нельзя
отличить улучшение от регрессии.

## Лицензия вклада

Отправляя PR, вы соглашаетесь на Apache-2.0. Не добавляйте зависимости под
копилефт-лицензиями в `dependencies`: ядро должно оставаться пригодным для
встраивания в закрытые продукты. Копилефт допустим только как явный
optional-extra по образцу `latin-gender` — с записью в NOTICE.
