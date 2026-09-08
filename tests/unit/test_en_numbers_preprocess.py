"""Оффлайн юнит-тесты нормализации английских числительных (без NER)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import pii_guard

_MODULE_PATH = (
    Path(pii_guard.__file__).resolve().parent
    / "preprocess"
    / "en_numbers.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("en_numbers_preprocess", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["en_numbers_preprocess"] = module
    spec.loader.exec_module(module)
    return module


en = _load()
norm = en.normalize_english_numbers


@pytest.mark.parametrize(
    "src, expected",
    [
        # поразрядная склейка телефонов
        ("call me at three one oh five five five one two three four",
         "call me at 3105551234"),
        ("my number is eight oh oh five five five one two one two",
         "my number is 8005551212"),
        ("six one seven five five five zero zero zero zero",
         "6175550000"),
        # карта (16 цифр)
        ("card four one one one one one one one one one one one one one one one",
         "card 4111111111111111"),
        # кардинальные значения
        ("I am twenty five years old", "I am 25 years old"),
        ("she is nineteen", "she is 19"),
        ("code one hundred twenty three", "code 123"),
        ("balance two thousand", "balance 2000"),
    ],
)
def test_normalization(src: str, expected: str) -> None:
    assert norm(src) == expected


def test_single_ambiguous_words_untouched() -> None:
    # одиночные простые слова не должны превращаться в цифры
    assert norm("the meeting is at five and i have one dog") == \
        "the meeting is at five and i have one dog"


def test_existing_digits_and_text_preserved() -> None:
    src = "order 42 is ready, call nine one seven one two three four"
    out = norm(src)
    assert "42" in out
    assert out.endswith("9171234")


def test_no_number_words_noop() -> None:
    src = "please contact our support team tomorrow"
    assert norm(src) == src
