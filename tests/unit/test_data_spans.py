"""Unit tests for narrowing a candidate span down to the data it carries."""

from __future__ import annotations

import pytest

from pii_guard.framework.recognizer import NumericPIIRecognizer
from pii_guard.framework.spans import data_spans


def _values(raw: str) -> list[str]:
    return [raw[start:end] for start, end in data_spans(raw)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # A service word between two chunks splits them: these are two values.
        ("серия 7518, номер 492137", ["7518", "492137"]),
        ("серия 77 14, номер 345678", ["77 14", "345678"]),
        ("серия МЗ, номер 0045678", ["МЗ", "0045678"]),
        ("II - КЗ номер 765432", ["II - КЗ", "765432"]),
        # Whitespace and dashes do not: one number written with separators.
        ("4510 123456", ["4510 123456"]),
        ("45 10 123456", ["45 10 123456"]),
        ("697-843 02", ["697-843 02"]),
        ("II - АВ 123456", ["II - АВ 123456"]),
        # A leading sign is dropped whether or not it is spaced off.
        ("№ 4719-596586", ["4719-596586"]),
        ("№6203-57 97-2798 55 20", ["6203-57 97-2798 55 20"]),
        # Nothing to strip -- the span is returned unchanged.
        ("123-456-789 00", ["123-456-789 00"]),
        ("АВ(12)34567", ["АВ(12)34567"]),
    ],
)
def test_data_spans_keeps_values_and_drops_service_words(raw: str, expected: list[str]) -> None:
    assert _values(raw) == expected


def test_data_spans_falls_back_to_the_whole_string() -> None:
    """A caller must always get something to report, even for a degenerate match."""
    assert data_spans("серия номер") == [(0, len("серия номер"))]
    assert data_spans("") == [(0, 0)]


def test_recognizer_reports_the_document_number_without_the_keywords() -> None:
    text = "Мои данные: паспорт серия 7518, номер 492137. Проверьте, пожалуйста."
    results = NumericPIIRecognizer().analyze(text, entities=["PASSPORT"])

    spans = sorted((r.start, r.end) for r in results)
    assert [text[start:end] for start, end in spans] == ["7518", "492137"]
    assert {r.entity_type for r in results} == {"PASSPORT"}


def test_recognizer_keeps_a_spaced_number_whole() -> None:
    """The split is driven by service words, not by whitespace inside a value."""
    text = "Паспорт 4510 123456 выдан ОВД."
    results = NumericPIIRecognizer().analyze(text, entities=["PASSPORT"])

    assert [(text[r.start:r.end]) for r in results] == ["4510 123456"]
