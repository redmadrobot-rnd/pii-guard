"""Оффлайн юнит-тесты поиска/декодирования Base64-подстрок (без NER)."""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

import pytest

import pii_guard

_MODULE_PATH = (
    Path(pii_guard.__file__).resolve().parent
    / "preprocess"
    / "base64_scan.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("base64_scan_preprocess", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["base64_scan_preprocess"] = module
    spec.loader.exec_module(module)
    return module


b64mod = _load()
find = b64mod.find_base64_spans


def _enc(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


@pytest.mark.parametrize(
    "payload",
    [
        "Иван Петров",
        "ivan.petrov@mail.ru",
        "+79161234567",
        "паспорт серия 4509 номер 123456",
        "good morning everyone how are you",  # длинный валидный, но без PII
    ],
)
def test_finds_and_decodes(payload: str) -> None:
    token = _enc(payload)
    text = f"данные: {token} конец"
    spans = find(text)
    assert len(spans) == 1
    assert spans[0].decoded == payload
    assert text[spans[0].start:spans[0].end] == token


def test_span_offsets_in_source() -> None:
    token = _enc("Зеленоград Москва")
    text = f"город {token}"
    sp = find(text)[0]
    assert text[sp.start:sp.end] == token


@pytest.mark.parametrize(
    "text",
    [
        "обычное длинное слово информационнотехнологический отдел",
        "короткий QW5uYQ== токен слишком мал",  # 8 символов < min_len
        "no base64 here at all, just words",
        "числа 1234567890 и текст",
    ],
)
def test_rejects_non_base64(text: str) -> None:
    assert find(text) == []


def test_rejects_binary_garbage() -> None:
    # валидный base64, но декодируется в бинарный мусор (не UTF-8 / непечатаемое)
    token = base64.b64encode(bytes(range(0, 32)) * 2).decode("ascii")
    assert find(f"данные {token} тут") == []


def test_min_len_threshold() -> None:
    # ровно на границе: короткая строка -> base64 < 16 символов -> отсев
    assert find(f"x {_enc('hi')} y") == []
