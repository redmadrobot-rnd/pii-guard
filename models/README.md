# Локальные веса

Эта папка нужна **только для закрытого контура**. В обычном случае она остаётся
пустой: `from_pretrained` скачивает модель сам, а в Docker-образ веса запекаются
на сборке.

Содержимое папки не коммитится (`.gitignore`) — 700 МБ, попавшие в git один раз,
уже не убрать без перезаписи истории.

## Как положить модель локально

На машине с доступом в интернет:

```bash
huggingface-cli download redmadrobot-rnd/rubert-base-pii-ner \
  --local-dir ./models/rubert-base-pii-ner
```

Перенесите папку на изолированный хост и укажите путь:

```bash
export PII_GUARD_NER_MODEL=/абсолютный/путь/models/rubert-base-pii-ner
```

## Что должно оказаться внутри

```
config.json            id2label с 43 BIO-лейблами (21 тип × B-/I- + O)
model.safetensors      ~700 МБ
tokenizer.json
tokenizer_config.json  model_max_length = 512
```

## Своя модель

Подойдёт любая token-classification модель с BIO-разметкой, чьи лейблы совпадают
с `NER_ENTITY_MAPPING` в `src/pii_guard/engine.py`. Лейблы, которых там нет,
игнорируются; типы из маппинга, которых нет в модели, просто не будут находиться.

```bash
export PII_GUARD_NER_MODEL=my-org/my-pii-model
export PII_GUARD_NER_REVISION=<commit sha>
```

Ревизию стоит пинить всегда: без неё перезаливка модели меняет поведение при
неизменном коде.
