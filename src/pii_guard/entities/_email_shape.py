"""Email shape-gate — единый источник истины «что выглядит как почта».

``_``-префиксный модуль: авто-дискавери НЕ регистрирует его как сущность
(см. ``framework.base.auto_discover_entities``). Держит детерминированные
правила формы, которыми фильтруется выход NER по типу ``EMAIL_ADDRESS``:
модель иногда метит как почту соц-ники (``@user_id``, ``instagram @gmail``)
или случайные слова. Правила сохраняют обфусцированные НАСТОЯЩИЕ почты
(reversed, гомоглиф, base64-decoded) — ради которых NER-email и держим —
и отбрасывают ник-образные ложные срабатывания.

``looks_like_email_shape`` — чистый предикат ``str -> bool``: ни presidio, ни
загрузки модели, поэтому юнит-тестируется быстро. ``filter_ner_email_spans``
detector-agnostic — duck-typed над ``RecognizerResult`` (читает
``.entity_type/.start/.end``), без импорта самого класса. Правила пробело-терпимы:
датасет токенизирован (``john doe @ example . com``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TypeVar

from pii_guard.framework.normalize import normalize_safe

EMAIL_ENTITY = "EMAIL_ADDRESS"

# Ядро почты: непустой local + @ + хотя бы 1 символ домена. Charset-агностично
# (local = любой непробельный, не @), чтобы не терять кириллические/гомоглиф-почты.
_CORE_RE = re.compile(r"[^\s@]+\s*@\s*[^\s@]")
# Плотная привязка: непробельный символ НЕПОСРЕДСТВЕННО перед @ ("andrey@gmail").
_TIGHT_RE = re.compile(r"[^\s@]@")
# Домен с .tld (пробело-терпимо: "sales . ru").
_TLD_RE = re.compile(r"@[^@]*\.\s*[A-Za-z]{2,}")
# Email-контекст рядом (действие, НЕ имена провайдеров вроде "gmail").
# `почт` — стем: `\bпочт\b` не совпадал ни с одной живой формой, то есть самый
# частый русский маркер почты гейт не проходил никогда. Латинские слова целиком —
# `\bmail\b` не должен ловить «gmail».
_CTX_RE = re.compile(r"(?iu)\bпочт\w{0,6}|\b(?:пиши|напиши)\s+на\b|\b(?:email|e-mail|mail)\b")

# Сколько символов слева от спана считаем «контекстом».
_CTX_WINDOW = 40


def looks_like_email_shape(surface: str, context: str = "") -> bool:
    """True, если *surface* похож на почту по форме.

    Требует ядро ``local@domain`` И хотя бы одно из: плотная привязка
    ``local@`` (без пробела перед ``@``), домен с ``.tld``, либо email-контекст
    рядом. Иначе (форма «слово @ник» без TLD и без контекста, напр.
    ``instagram @gmail``) → False.
    """
    surface = normalize_safe(surface)
    if not _CORE_RE.search(surface):
        return False
    return bool(
        _TIGHT_RE.search(surface)
        or _TLD_RE.search(surface)
        or _CTX_RE.search(normalize_safe(context))
    )


T = TypeVar("T")


def filter_ner_email_spans(results: Iterable[T], text: str) -> list[T]:
    """Отбрасывает NER-спаны ``EMAIL_ADDRESS``, не похожие на почту по форме.

    Остальные типы сущностей проходят без изменений. Detector-agnostic:
    duck-typed над объектами со ``.entity_type/.start/.end`` (RecognizerResult).
    Вызывается из ``Engine._analyze_raw``, поэтому видит правильную
    поверхность во всех ветках (основной текст, decoded base64, кириллица/
    латиница при транслите).
    """
    kept: list[T] = []
    for r in results:
        if getattr(r, "entity_type", None) == EMAIL_ENTITY:
            surface = text[r.start : r.end]
            context = text[max(0, r.start - _CTX_WINDOW) : r.start]
            if not looks_like_email_shape(surface, context):
                continue
        kept.append(r)
    return kept
