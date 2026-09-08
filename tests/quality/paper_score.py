#!/usr/bin/env python3
"""Единый скорер для таблиц статьи. Один протокол на все системы и наборы.

Считает две метрики уровня спанов, обе с проверкой типа:

* **strict** — совпали начало, конец и тип. Никакой нормализации: ни обрезки
  пробелов, ни пунктуации, ни регистра. Смещения в кодовых точках Unicode.
* **soft** — совпал тип и спаны пересеклись хотя бы на один символ. Схема
  ``type`` из SemEval-2013.

Сопоставление один к одному, жадно по позиции: спаны обеих сторон сортируются по
началу, каждый эталонный спан забирает первое ещё не занятое подходящее
предсказание. Без этого одно широкое предсказание засчиталось бы нескольким
эталонным спанам сразу.

Агрегаты: micro (все спаны в общую кучу) и macro (F1 по типу, затем невзвешенное
среднее). Оба — только по типам, реализованным у обеих сравниваемых систем
(«общая область»): иначе метрика измеряет наличие функции, а не качество
детекции. Полная область считается отдельно и подписывается как полная.

Два режима гранулярности: ``raw`` (только маппинг меток) и ``pipeline`` (плюс
склейка соседних предсказаний одного семейства через адресные связки). Правило
склейки одно на все системы. Эталонные спаны не склеиваются никогда.

Вход — JSON-дампы предсказаний: список записей
``{"id", "text", "gold": [{"start","end","type"}], "pred": [{"start","end", <метка>}]}``,
где метка — ``entity_type`` (pii-guard), ``family`` или ``rule`` (cloud.ru).
Дампы держатся вне репозитория: в них тексты оцениваемых наборов.

Использование::

    python tests/quality/paper_score.py <manifest.json>

Манифест перечисляет прогоны: датасет, система, путь к дампу. Формат — см.
``load_manifest``.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ── Таксономия и маппинг меток ──────────────────────────────────
#
# Публикуется вместе со скриптом: без него числа не воспроизводимы. Слева метки
# систем и эталонов, справа — семейство, в котором идёт сравнение.

PERSON_PARTS = ("FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "PATRONYMIC")
ADDRESS_PARTS = ("COUNTRY", "REGION", "DISTRICT", "CITY", "STREET", "HOUSE", "POSTAL_CODE")

# pii-guard → семейство
OURS_TO_FAMILY = {
    "PERSON": "PERSON",
    "LOCATION": "ADDRESS",
    "POSTAL_CODE": "ADDRESS",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "URL": "URL",
    "IP_ADDRESS": "IP_ADDRESS",
    "IP_PORT": "IP_ADDRESS",
    "PASSPORT": "PASSPORT",
    "INN": "INN",
    "SNILS": "SNILS",
    "OMS": "OMS",
    "CREDIT_CARD": "CREDIT_CARD",
    "DRIVER_LICENSE": "DRIVER_LICENSE",
    "MILITARY_ID": "MILITARY_ID",
    "BIRTH_CERTIFICATE": "BIRTH_CERTIFICATE",
    "BANK_ACCOUNT": "BANK_ACCOUNT",
    "BIK": "BIK",
    "TELEGRAM": "TELEGRAM",
    "DATE_TIME": "DATE_TIME",
}

# cloud.ru: префикс rule_id → семейство. Из configs/guardrails_regex_rules.yaml.
CLOUDRU_RULE_PREFIX_TO_FAMILY = {
    "pii.fio-ru": "PERSON",
    "pii.docs.address": "ADDRESS",
    "pii.email": "EMAIL",
    "pii.phone-ru": "PHONE",
    "ip-addrs.": "IP_ADDRESS",
    "pii.docs.passport": "PASSPORT",
    "pii.docs.inn-": "INN",
    "pii.docs.snils": "SNILS",
    "pii.fin.credit-card": "CREDIT_CARD",
    "pii.docs.ogrn": "OGRN",
    "pii.docs.kpp": "KPP",
    "pii.fin.cvc": "CVC",
    "pii.fin.iban": "IBAN",
}

# Наша NER-модель без ветки правил: её 21 метка → семейство. Совпадает с
# `NER_ENTITY_MAPPING` в `pii_guard/engine.py`, доведённым до семейств: там метка
# модели превращается в тип pii-guard, здесь тип — в семейство. Дублируется, а не
# импортируется, чтобы скорер остался автономным.
NER_LABEL_TO_FAMILY = {
    "FIRST_NAME": "PERSON", "LAST_NAME": "PERSON", "MIDDLE_NAME": "PERSON",
    "COUNTRY": "ADDRESS", "REGION": "ADDRESS", "DISTRICT": "ADDRESS",
    "CITY": "ADDRESS", "STREET": "ADDRESS", "HOUSE": "ADDRESS",
    "EMAIL": "EMAIL", "PHONE": "PHONE", "URL": "URL",
    "IP_ADDRESS": "IP_ADDRESS", "PASSPORT": "PASSPORT", "INN": "INN",
    "SNILS": "SNILS", "OMS": "OMS", "CREDIT_CARD": "CREDIT_CARD",
    "DRIVER_LICENSE": "DRIVER_LICENSE", "MILITARY_ID": "MILITARY_ID",
    "BIRTH_CERTIFICATE": "BIRTH_CERTIFICATE",
}

# GLiNER Guard: штатная онтология из 32 меток → семейство. Одна метка ложится
# ровно в одно семейство либо отбрасывается; отброшенные перечислены в
# GLINER_DROPPED, чтобы список был виден, а не потерялся молча.
GLINER_TO_FAMILY = {
    "person": "PERSON", "first_name": "PERSON", "last_name": "PERSON",
    "country": "ADDRESS", "region": "ADDRESS", "city": "ADDRESS",
    "district": "ADDRESS", "street": "ADDRESS", "building": "ADDRESS",
    "unit": "ADDRESS", "postal_code": "ADDRESS", "landmark": "ADDRESS",
    "address": "ADDRESS",
    "email": "EMAIL",
    "phone": "PHONE",
    "social_account": "TELEGRAM", "messenger": "TELEGRAM",
    "passport": "PASSPORT",
    "date_of_birth": "DATE_TIME", "event_date": "DATE_TIME",
    "card_number": "CREDIT_CARD",
    "bank_account": "BANK_ACCOUNT",
}
# Отброшены с причиной:
#   alias, title            — не имя как таковое: alias срабатывает на никах
#                             (`@ivan_petrov`, скор 0.99), title — на обращениях;
#                             в эталонах спаны PERSON это настоящие ФИО
#   company, government,    — организации; семейства «организация» нет ни в одной
#   education, media,         из сравниваемых таксономий
#   product
#   national_id,            — не сводятся к одному типу: на проверочных текстах
#   document_id               national_id срабатывал и на номере паспорта, и на
#                             военном билете; выбор одного семейства был бы
#                             произволом, а нескольких — запрещён правилом
#   crypto_wallet           — соответствия в наборах нет
GLINER_DROPPED = frozenset({
    "alias", "title", "company", "government", "education", "media", "product",
    "national_id", "document_id", "crypto_wallet",
})

# Эталонные метки наборов → семейство. Разные наборы называют одно и то же
# по-разному, поэтому по набору отдельно.
GOLD_TO_FAMILY = {
    "hivetrace": {
        "NAME": "PERSON", "FIO": "PERSON",
        "ADDRESS": "ADDRESS",
        "EMAIL": "EMAIL", "PHONE": "PHONE", "PHONE_NUMBER": "PHONE",
        "SNILS": "SNILS", "INN": "INN",
        "PASSPORT": "PASSPORT", "PASSPORT_NUMBER": "PASSPORT",
        "BANK_CARD_NUMBER": "CREDIT_CARD", "CARD": "CREDIT_CARD",
        "OGRN": "OGRN", "OGRNIP": "OGRN", "KPP": "KPP",
        "CVC": "CVC", "TOKEN": "TOKEN", "IBAN": "IBAN",
        "DOB": "DATE_TIME", "IP": "IP_ADDRESS",
    },
    "alexen": {"PER": "PERSON", "NAME": "PERSON", "PHONE": "PHONE", "EMAIL": "EMAIL"},
    "alrosait": {"NAME": "PERSON", "ADDRESS": "ADDRESS"},
    "rmr": (
        dict.fromkeys(PERSON_PARTS, "PERSON")
        | dict.fromkeys(ADDRESS_PARTS, "ADDRESS")
        | {
            "EMAIL": "EMAIL", "PHONE": "PHONE", "URL": "URL",
            "IP_ADDRESS": "IP_ADDRESS", "PASSPORT": "PASSPORT", "INN": "INN",
            "SNILS": "SNILS", "OMS": "OMS", "CREDIT_CARD": "CREDIT_CARD",
            "DRIVER_LICENSE": "DRIVER_LICENSE", "MILITARY_ID": "MILITARY_ID",
            "BIRTH_CERTIFICATE": "BIRTH_CERTIFICATE", "DATE_TIME": "DATE_TIME",
            "BANK_ACCOUNT": "BANK_ACCOUNT", "BIK": "BIK", "TELEGRAM": "TELEGRAM",
        }
    ),
}

# Что каждая система умеет вообще. Пересечение задаёт общую область.
OURS_FAMILIES = frozenset(OURS_TO_FAMILY.values())
CLOUDRU_FAMILIES = frozenset(CLOUDRU_RULE_PREFIX_TO_FAMILY.values())
# Ветка правил не умеет NER-only типы (`config.NER_ONLY_ENTITIES`): PERSON,
# LOCATION, PHONE_NUMBER, URL. ADDRESS остаётся: индекс — правило, и он часть
# адресного спана разметки.
GLINER_FAMILIES = frozenset(GLINER_TO_FAMILY.values())
# У модели нет головы под DATE_TIME, BANK_ACCOUNT, BIK и TELEGRAM: эти типы в
# pii-guard делают только правила.
NER_FAMILIES = frozenset(NER_LABEL_TO_FAMILY.values())
SYSTEM_FAMILIES = {
    "ours": OURS_FAMILIES,
    "ours-rules": OURS_FAMILIES - {"PERSON", "PHONE", "URL"},
    "ours-ner": NER_FAMILIES,
    "cloudru": CLOUDRU_FAMILIES,
    # Фактическое покрытие после маппинга; онтология у обеих моделей одна.
    "gliner-uni": GLINER_FAMILIES,
    "gliner-omni": GLINER_FAMILIES,
}

# ── Склейка для режима pipeline ─────────────────────────────────
#
# Соседние предсказания одного семейства сливаются, если между ними только
# пробелы, пунктуация или адресная связка («ул.», «д.», «кв.»). Разметка наборов
# держит адрес одним спаном, а наши правила — покомпонентно; без склейки strict
# наказывает за гранулярность, а не за детекцию. Правило одно на все системы.
CONNECTIVE = re.compile(
    r"^[\s,.;«»\"'()-]*"
    r"(?:(?:ул|улица|улице|пр|просп|проспект|пер|переулок|наб|бул|б-р|ш|шоссе|"
    r"пл|площадь|проезд|д|дом|кв|квартира|корп|корпус|стр|строение|обл|область|"
    r"г|город|р-н|район|подъезд|под|этаж|офис|оф|индекс)\.?\s*)*"
    r"[\s,.;«»\"'()-]*$",
    re.IGNORECASE | re.UNICODE,
)
MERGE_FAMILIES = frozenset({"ADDRESS", "PERSON"})
MAX_MERGE_GAP = 12


def merge_spans(spans: list[dict], text: str) -> list[dict]:
    """Склейка соседних спанов одного семейства. Только для предсказаний."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    out = [dict(ordered[0])]
    for span in ordered[1:]:
        prev = out[-1]
        gap = text[prev["end"]: span["start"]]
        joinable = (
            span["type"] == prev["type"]
            and span["type"] in MERGE_FAMILIES
            and span["start"] >= prev["end"]
            and len(gap) <= MAX_MERGE_GAP
            and CONNECTIVE.match(gap) is not None
        )
        if joinable:
            prev["end"] = max(prev["end"], span["end"])
        else:
            out.append(dict(span))
    return out


# ── Нормализация записей дампа ──────────────────────────────────

def resolve_by_score(preds: list[dict]) -> list[dict]:
    """Одна метка на позицию: побеждает предсказание с большим скором.

    GLiNER опрашивается сразу всей онтологией, поэтому один и тот же спан
    приходит под несколькими метками (``4510 123456`` — сразу ``passport``,
    ``card_number``, ``national_id``, ``social_account``). Без разрешения каждая
    лишняя метка стала бы отдельным ложным срабатыванием, чего в реальном
    применении не бывает: потребитель выбирает одну. Правило то же по смыслу, что
    у нас (при перекрытии остаётся один спан), но критерий здесь — скор модели,
    единственная величина, которую она сама сообщает.
    """
    kept: list[dict] = []
    for pred in sorted(preds, key=lambda s: (-s.get("score", 0.0), s["start"], s["end"])):
        if any(pred["start"] < k["end"] and k["start"] < pred["end"] for k in kept):
            continue
        kept.append(pred)
    return sorted(kept, key=lambda s: (s["start"], s["end"]))


def _pred_family(pred: dict, system: str) -> str | None:
    if system == "ours-ner":
        # Дамп голой модели хранит её собственные метки, а не типы pii-guard:
        # `FIRST_NAME`, а не `PERSON`. Разрешать перекрытия не нужно —
        # `aggregation_strategy="simple"` уже отдаёт непересекающиеся спаны.
        return NER_LABEL_TO_FAMILY.get((pred.get("entity_type") or "").upper())
    if system.startswith("gliner"):
        label = (pred.get("entity_type") or "").lower()
        return GLINER_TO_FAMILY.get(label)
    if system == "cloudru":
        rule = pred.get("family") or pred.get("rule") or ""
        if pred.get("family"):
            return pred["family"]
        for prefix, fam in CLOUDRU_RULE_PREFIX_TO_FAMILY.items():
            if rule.startswith(prefix):
                return fam
        return None
    # Часть дампов хранит метку типом pii-guard (`entity_type`), часть — уже
    # семейством (`family`). Без второй ветки записи вида `{"family": "ADDRESS"}`
    # молча отбрасывались: `ADDRESS` не ключ таблицы типов, а её значение.
    label = (pred.get("entity_type") or pred.get("family") or pred.get("rule") or "").upper()
    if label in OURS_TO_FAMILY:
        return OURS_TO_FAMILY[label]
    return label if label in OURS_FAMILIES else None


def _gold_family(gold: dict, dataset: str) -> str | None:
    label = (gold.get("type") or gold.get("family") or "").upper()
    return GOLD_TO_FAMILY[dataset].get(label)


@dataclass
class Record:
    id: str
    text: str
    gold: list[dict]
    pred: list[dict]
    excluded_gold: list[dict] = field(default_factory=list)


def load_run(path: Path, dataset: str, system: str, scope: frozenset[str] | None) -> list[Record]:
    """Читает дамп, приводит метки к семействам, применяет область типов.

    *scope* ``None`` — полная область (только то, что вообще отображается в
    семейства). Иначе за областью остаются и эталонные спаны, и предсказания:
    иначе метрика измеряла бы наличие функции у одной из сторон.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    records: list[Record] = []
    for i, item in enumerate(raw):
        text = item.get("normalized_text") or item.get("text") or ""
        gold_all, gold_out, excluded = [], [], []
        for g in item.get("gold", []):
            fam = _gold_family(g, dataset)
            if fam is None:
                continue
            span = {"start": g["start"], "end": g["end"], "type": fam}
            gold_all.append(span)
            if scope is None or fam in scope:
                gold_out.append(span)
            else:
                excluded.append(span)
        mapped = []
        for p in item.get("pred", []):
            fam = _pred_family(p, system)
            if fam is None:
                continue
            mapped.append({"start": p["start"], "end": p["end"], "type": fam,
                           "score": p.get("score", 1.0)})
        if system.startswith("gliner"):
            # Разрешение до сужения области, а не после: система выдаёт свой ответ
            # один раз, независимо от того, какие типы мы потом решили измерять.
            mapped = resolve_by_score(mapped)
        preds = [{"start": p["start"], "end": p["end"], "type": p["type"]}
                 for p in mapped if scope is None or p["type"] in scope]
        records.append(
            Record(id=str(item.get("id", i)), text=text, gold=gold_out,
                   pred=preds, excluded_gold=excluded)
        )
    return records


def drop_preds_on_excluded(rec: Record) -> list[dict]:
    """Убирает предсказания, попавшие в эталон исключённого типа.

    Побочный эффект сужения области: находка на спане, который мы из знаменателя
    убрали, иначе превратилась бы в ложное срабатывание. Применяется одинаково ко
    всем системам.
    """
    if not rec.excluded_gold:
        return rec.pred
    kept = []
    for p in rec.pred:
        if any(p["start"] < g["end"] and g["start"] < p["end"] for g in rec.excluded_gold):
            continue
        kept.append(p)
    return kept


# ── Сопоставление ───────────────────────────────────────────────

def match(gold: list[dict], pred: list[dict], strict: bool) -> tuple[int, list[dict], list[dict]]:
    """Жадное сопоставление один к одному. Возвращает (tp, unmatched_gold, unmatched_pred)."""
    used = [False] * len(pred)
    tp = 0
    missed = []
    for g in sorted(gold, key=lambda s: (s["start"], s["end"])):
        hit = None
        for j, p in enumerate(sorted(pred, key=lambda s: (s["start"], s["end"]))):
            if used[j] or p["type"] != g["type"]:
                continue
            if strict:
                ok = p["start"] == g["start"] and p["end"] == g["end"]
            else:
                ok = p["start"] < g["end"] and g["start"] < p["end"]
            if ok:
                hit = j
                break
        if hit is None:
            missed.append(g)
        else:
            used[hit] = True
            tp += 1
    spurious = [p for j, p in enumerate(sorted(pred, key=lambda s: (s["start"], s["end"]))) if not used[j]]
    return tp, missed, spurious


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class Score:
    micro: tuple[float, float, float]
    macro_f1: float
    per_type: dict[str, tuple[int, int, int]]  # type -> (tp, fp, fn)
    tp: int
    fp: int
    fn: int
    negatives_total: int
    negatives_with_fp: int


def score(records: list[Record], *, strict: bool, pipeline: bool) -> Score:
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    tp = fp = fn = 0
    neg_total = neg_fp = 0
    for rec in records:
        preds = drop_preds_on_excluded(rec)
        if pipeline:
            preds = merge_spans(preds, rec.text)
        hits, missed, spurious = match(rec.gold, preds, strict)
        tp += hits
        fn += len(missed)
        fp += len(spurious)
        for g in rec.gold:
            per_type[g["type"]][0] += 0  # гарантируем строку даже при нулях
        for g in missed:
            per_type[g["type"]][2] += 1
        for p in spurious:
            per_type[p["type"]][1] += 1
        matched_by_type: dict[str, int] = defaultdict(int)
        for g in rec.gold:
            matched_by_type[g["type"]] += 1
        for g in missed:
            matched_by_type[g["type"]] -= 1
        for t, n in matched_by_type.items():
            per_type[t][0] += n
        if not rec.gold and not rec.excluded_gold:
            neg_total += 1
            if spurious:
                neg_fp += 1
    macro = [prf(*per_type[t])[2] for t in per_type if per_type[t][0] + per_type[t][2] > 0]
    return Score(
        micro=prf(tp, fp, fn),
        macro_f1=sum(macro) / len(macro) if macro else 0.0,
        per_type={t: tuple(v) for t, v in sorted(per_type.items())},
        tp=tp, fp=fp, fn=fn,
        negatives_total=neg_total, negatives_with_fp=neg_fp,
    )


# ── Манифест и вывод ────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    """Манифест прогонов::

        {
          "runs": [
            {"dataset": "hivetrace", "system": "ours", "dump": "/path/x.json",
             "split": "entity", "limit": 2000, "label": "hivetrace entity"}
          ]
        }

    ``split`` и ``limit`` необязательны: первое фильтрует по полю ``split`` в
    дампе, второе берёт первые N записей (для ``alrosait train[:2000]``).
    """
    return json.loads(path.read_text(encoding="utf-8"))


def split_ids(raw: list[dict], split: str, split_from: Path | None) -> set[str]:
    """Идентификаторы нужного сплита.

    Дамп конкурента поля ``split`` не несёт, поэтому список берётся из дампа, где
    оно есть (*split_from*), по совпадающим ``id``. Сравнивать сплиты hivetrace
    смешанно нельзя: пер-типовые таблицы только на entity, пер-доменные только на
    domain.
    """
    source = json.loads(split_from.read_text(encoding="utf-8")) if split_from else raw
    return {str(item.get("id", i)) for i, item in enumerate(source)
            if item.get("split") == split}


def filter_records(records, raw, split, limit, split_from=None):
    if split:
        keep = split_ids(raw, split, split_from)
        records = [r for r in records if r.id in keep]
    if limit:
        records = records[:limit]
    return records


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    manifest = load_manifest(Path(argv[1]))
    results: dict[str, dict] = {}

    for run in manifest["runs"]:
        if "dataset" not in run:  # запись-комментарий вида {"_note": ...}
            continue
        dataset, system = run["dataset"], run["system"]
        dump = Path(run["dump"])
        label = run.get("label") or f"{dataset} / {system}"
        peers = run.get("peers") or ["ours", "cloudru"]
        shared = frozenset.intersection(*(SYSTEM_FAMILIES[p] for p in peers))
        raw = json.loads(dump.read_text(encoding="utf-8"))

        for scope_name, scope in (("shared", shared), ("full", None)):
            records = load_run(dump, dataset, system, scope)
            records = filter_records(
                records, raw, run.get("split"), run.get("limit"),
                Path(run["split_from"]) if run.get("split_from") else None,
            )
            for mode in ("raw", "pipeline"):
                for metric in ("strict", "soft"):
                    s = score(records, strict=metric == "strict", pipeline=mode == "pipeline")
                    results[f"{label} :: {system} :: {scope_name} :: {mode} :: {metric}"] = {
                        "precision": s.micro[0], "recall": s.micro[1], "f1": s.micro[2],
                        "macro_f1": s.macro_f1, "tp": s.tp, "fp": s.fp, "fn": s.fn,
                        "per_type": {k: list(v) for k, v in s.per_type.items()},
                        "negatives_total": s.negatives_total,
                        "negatives_with_fp": s.negatives_with_fp,
                        "texts": len(records),
                    }

    out = Path(manifest.get("out", "paper_scores.json"))
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    header = (f"{'прогон':40} {'обл':7} {'реж':9} {'метр':7} "
              f"{'P':>6} {'R':>6} {'F1':>6} {'macro':>6} {'текстов':>8}")
    print(header)
    print("-" * len(header))
    for key, v in results.items():
        label, system, scope, mode, metric = key.split(" :: ")
        print(f"{label:40} {scope:7} {mode:9} {metric:7} "
              f"{v['precision'] * 100:6.1f} {v['recall'] * 100:6.1f} "
              f"{v['f1'] * 100:6.1f} {v['macro_f1'] * 100:6.1f} {v['texts']:8d}")
    print(f"\nзаписано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
