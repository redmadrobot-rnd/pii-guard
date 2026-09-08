#!/usr/bin/env bash
# Полный цикл против запущенного сервиса: обезличили -> отдали "модели" ->
# восстановили. Ничего кроме curl и python3 не нужно.
#
#   docker compose up --build -d      # из корня репозитория
#   bash examples/quickstart/demo.sh
set -euo pipefail

BASE="${PII_GUARD_URL:-http://127.0.0.1:8080}"
AUTH=()
[[ -n "${PII_GUARD_API_KEY:-}" ]] && AUTH=(-H "Authorization: Bearer ${PII_GUARD_API_KEY}")
# `${AUTH[@]+...}` вместо простого `${AUTH[@]}`: в bash 3.2 (штатный на macOS)
# под `set -u` разворот пустого массива — ошибка «unbound variable».

TEXT='Меня зовут Иван Петров, ИНН 7707083893, тел. +7 916 123-45-67. Живу в Москве.'

echo "── здоровье ──"
curl -sS ${AUTH[@]+"${AUTH[@]}"} "$BASE/health" | python3 -m json.tool

echo
echo "── исходный текст ──"
echo "$TEXT"

echo
echo "── обезличивание (mode=pseudonymize) ──"
RESPONSE=$(curl -sS ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"texts":[sys.argv[1]],"mode":"pseudonymize"}))' "$TEXT")" \
  "$BASE/anonymize")

SANITISED=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["items"][0]["text"])' <<<"$RESPONSE")
MAPPING=$(python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["mapping"], ensure_ascii=False))' <<<"$RESPONSE")

echo "$SANITISED"
echo
echo "  ↑ именно это уходит в LLM. Найденные сущности:"
python3 -c '
import json, sys
d = json.load(sys.stdin)["items"][0]
text = d["normalized_text"]
for e in sorted(d["entities"], key=lambda e: e["start"]):
    print("    {:<16} {}".format(e["entity_type"], text[e["start"]:e["end"]]))
' <<<"$RESPONSE"

echo
echo "── таблица псевдонимов (осталась у клиента, сервис её не хранит) ──"
python3 -m json.tool --no-ensure-ascii <<<"$MAPPING"

# Модель отвечает плейсхолдерами. Берём именно тег PERSON: склонение видно
# только на нём, а порядок ключей таблицы обратен порядку в тексте.
PERSON_TAG=$(python3 -c '
import json, sys
mapping = json.load(sys.stdin)
print(next((k for k in mapping if "PERSON" in k), next(iter(mapping))))
' <<<"$MAPPING")
ANSWER="Я свяжусь с ${PERSON_TAG} и уточню детали."

echo
echo "── ответ 'модели' ──"
echo "$ANSWER"

echo
echo "── восстановление ──"
curl -sS ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"texts":[sys.argv[1]],"mapping":json.loads(sys.argv[2])}))' "$ANSWER" "$MAPPING")" \
  "$BASE/deanonymize" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["texts"][0])'

echo
echo 'Обратите внимание на падеж: "с Иваном Петровым", а не "с Иван Петров" —'
echo 'для имён в таблице лежит именительный, склонение делается по контексту.'
