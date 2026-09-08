"""Unit tests for text normalization utilities."""

from __future__ import annotations

import pytest

from pii_guard.framework.normalize import (
    _parse_ru_number_stream,
    normalize_safe,
    normalize_text,
)


def test_normalize_safe_replaces_spaces_and_dashes_without_length_change() -> None:
    source = "A\u00a0B\u202fC\u2014D\tE"
    normalized = normalize_safe(source)

    assert normalized == "A B C-D E"
    assert len(normalized) == len(source)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Код: один два три четыре.", "Код: 1234."),
        ("Количество: сто двадцать три.", "Количество: 123."),
    ],
)
def test_normalize_text_happy_path(source: str, expected: str) -> None:
    assert normalize_text(source) == expected


def test_normalize_text_keeps_short_digit_word_sequences() -> None:
    source = "код один два три"
    assert normalize_text(source) == source


def test_normalize_text_removes_soft_hyphen() -> None:
    source = "тест\u00adтекст"
    assert "\u00ad" not in normalize_text(source)


def test_normalize_text_removes_zero_width_chars() -> None:
    source = "abc\u200bdef\u200cgh\ufeffi"
    result = normalize_text(source)
    for ch in "\u200b\u200c\u200d\ufeff":
        assert ch not in result


def test_normalize_text_collapses_multiple_spaces() -> None:
    source = "один  два   три"
    assert "  " not in normalize_text(source)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["двадцать", "пять"], [25]),
        (["сто", "двадцать", "три"], [123]),
        (["одна", "тысяча"], [1000]),
        (["девять", "сто"], [9, 100]),
    ],
)
def test_parse_ru_number_stream_happy_path(
    tokens: list[str],
    expected: list[int],
) -> None:
    assert _parse_ru_number_stream(tokens) == expected


@pytest.mark.parametrize(
    "tokens",
    [
        ["abc"],
        ["двадцать", "abc"],
        ["тысяча", "двадцать", "abc"],
    ],
)
def test_parse_ru_number_stream_returns_none_for_invalid_input(
    tokens: list[str],
) -> None:
    assert _parse_ru_number_stream(tokens) is None
