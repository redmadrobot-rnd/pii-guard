"""Оффлайн юнит-тесты транслит-препроцессора (без NER и Presidio).

Модуль загружается напрямую из файла, чтобы тест не зависел от тяжёлых
зависимостей пакета ``pii_guard``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import pii_guard

_MODULE_PATH = (
    Path(pii_guard.__file__).resolve().parent
    / "preprocess"
    / "translit.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("translit_preprocess", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["translit_preprocess"] = module
    spec.loader.exec_module(module)
    return module


tr = _load_module()


# ── transliterate_to_cyrillic ────────────────────────────────────────


@pytest.mark.parametrize(
    "src, expected",
    [
        ("menya", "меня"),
        ("evgeniya", "евгения"),
        ("moj nomer", "мой номер"),
        ("zhivu", "живу"),
        ("Sankt-Peterburg", "Санкт-Петербург"),
        ("Yekaterinburg", "Екатеринбург"),
        ("Shcherbakova", "Щербакова"),
        ("Dmitriy", "Дмитрий"),
        ("Yuliya", "Юлия"),
        ("Tsvetkova", "Цветкова"),
    ],
)
def test_transliteration_digraphs(src: str, expected: str) -> None:
    assert tr.transliterate_to_cyrillic(src).cyrillic == expected


def test_digits_and_punctuation_preserved() -> None:
    src = "moj nomer 8499876543, eto pravda"
    out = tr.transliterate_to_cyrillic(src).cyrillic
    assert "8499876543" in out  # цифры не тронуты
    assert "," in out


# ── looks_like_translit ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "menya zovut evgeniya, moj nomer 8499876543",
        "privet, ya ivan petrov, prozhivayu v sankt-peterburge",
        "Dmitriy Yurevich Kuznetsov, gorod Yekaterinburg",
    ],
)
def test_detects_translit(text: str) -> None:
    assert tr.looks_like_translit(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "my name is John and I am from New York, you can call me later",
        "the weather is nice and the meeting is scheduled for tomorrow",
        "hello world",  # слишком мало сигналов
    ],
)
def test_ignores_plain_english(text: str) -> None:
    assert tr.looks_like_translit(text) is False


# ── SpanMap.to_source_span (ремаппинг координат) ─────────────────────


def test_span_remap_to_source() -> None:
    src = "menya zovut Ivan Petrov moj nomer 89161234567"
    span_map = tr.transliterate_to_cyrillic(src)
    cyr = span_map.cyrillic
    start = cyr.index("Иван Петров")
    end = start + len("Иван Петров")
    mapped = span_map.to_source_span(start, end)
    assert mapped is not None
    s, e = mapped
    assert src[s:e] == "Ivan Petrov"


def test_span_remap_phone_digits() -> None:
    src = "moj nomer 89161234567"
    span_map = tr.transliterate_to_cyrillic(src)
    cyr = span_map.cyrillic
    start = cyr.index("89161234567")
    end = start + len("89161234567")
    # цифры переносятся дословно и корректно ремаппятся обратно в источник
    mapped = span_map.to_source_span(start, end)
    assert mapped is not None
    assert src[mapped[0]:mapped[1]] == "89161234567"


def test_pure_digits_kept_verbatim() -> None:
    span_map = tr.transliterate_to_cyrillic("12345")
    assert span_map.cyrillic == "12345"
    # цифры переносятся дословно (1:1) и корректно ремаппятся обратно
    assert span_map.to_source_span(0, 5) == (0, 5)


def test_digits_inside_span_not_dropped() -> None:
    # ключевой регресс-кейс: спан, охватывающий слово + цифры, должен включать цифры
    src = "pasport seriya 4509 nomer 123456"
    sm = tr.transliterate_to_cyrillic(src)
    frag = "серия 4509 номер 123456"
    i = sm.cyrillic.find(frag)
    mapped = sm.to_source_span(i, i + len(frag))
    assert mapped is not None
    assert src[mapped[0]:mapped[1]] == "seriya 4509 nomer 123456"


# ── строгая детекция: email / url / base64 / сверхдлинные не транслит ─


@pytest.mark.parametrize(
    "text",
    [
        # email-токены (есть '@') — раньше ложно заходили по диграфам zh/ts/kh
        "Старый email elena.vinogradov@hotmail.com новый aleksey.zhukov@web.de",
        "отправить mikhail.tarasov64@list.ru сегодня",
        # url
        "Контакты: сайт https://gov.ru почта e.zaitsev@internet.ru",
        # base64-подобный токен (есть '=' / '+')
        "вот строка 0KHQndCY0JvQoSAxMTIyMzM0NDU5NQ== в логе",
        # сверхдлинный латинский run (> 25 символов)
        "идентификатор abcdefghijklmnopqrstuvwxyzABCDEF тут",
    ],
)
def test_structural_tokens_not_translit(text: str) -> None:
    assert tr.looks_like_translit(text) is False


def test_email_kept_verbatim_in_transliteration() -> None:
    # email сохраняется ДОСЛОВНО (не транслитерируется и не затирается),
    # обычные слова вокруг — транслитерируются. Так в NER уходит валидный email.
    out = tr.transliterate_to_cyrillic("na pochtu ivan@mail.ru pishi").cyrillic
    assert "ivan@mail.ru" in out          # email цел
    assert "почту" in out and "пиши" in out  # контекст транслитерирован
    assert "жуков" not in out  # (email-домены не искажены)


def test_genuine_translit_still_detected_with_email_present() -> None:
    # настоящий транслит остаётся транслитом, даже если рядом есть email
    text = "menya zovut ivan moj nomer pishite na test@mail.ru"
    assert tr.looks_like_translit(text) is True
