"""Unit tests for regex-based PII detection components."""

from __future__ import annotations

import re

from pii_guard.entities.birth_certificate import BIRTH_CERT_NUMBER_RE
from pii_guard.entities.date_time import register_date_time
from pii_guard.entities.inn import InnClassifier
from pii_guard.entities.ip_address import register_ip_address
from pii_guard.entities.ip_port import register_ip_port
from pii_guard.entities.military_id import MILITARY_ID_RE, MILITARY_SERIES_NUMBER_RE
from pii_guard.framework.patterns import (
    NUM_CANDIDATE_RE,
    NUM_CANDIDATE_WITH_SIGN_RE,
    SERIES_NUMBER_NUMERIC_RE,
)


class _FakeRegistry:
    def __init__(self) -> None:
        self.recognizers: list[object] = []

    def add_recognizer(self, recognizer: object) -> None:
        self.recognizers.append(recognizer)


class _FakeAnalyzer:
    def __init__(self) -> None:
        self.registry = _FakeRegistry()


def _compiled_patterns_from(register_fn) -> list[re.Pattern[str]]:
    analyzer = _FakeAnalyzer()
    register_fn(analyzer, language="ru")

    out: list[re.Pattern[str]] = []
    for recognizer in analyzer.registry.recognizers:
        for pattern in getattr(recognizer, "patterns", []):
            out.append(re.compile(pattern.regex, re.IGNORECASE | re.UNICODE))
    return out


def _matches_any(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) is not None for p in patterns)


def _all_matches(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    return [match.group(0) for pattern in patterns for match in pattern.finditer(text)]


def test_framework_numeric_candidate_patterns() -> None:
    assert NUM_CANDIDATE_RE.search("паспорт 4510 678234")
    assert not NUM_CANDIDATE_RE.search("код 12345")

    assert NUM_CANDIDATE_WITH_SIGN_RE.search("№ 4510 678234")
    assert not NUM_CANDIDATE_WITH_SIGN_RE.search("N 4510 678234")


def test_framework_series_number_patterns() -> None:
    text = "серия 2200 7133 номер 9596 2923"
    assert SERIES_NUMBER_NUMERIC_RE.search(text)



def test_ip_port_regex_registration() -> None:
    patterns = _compiled_patterns_from(register_ip_port)

    assert _matches_any(patterns, "Разрешить 10.0.5.21:8443")
    assert _matches_any(patterns, "listen on [2001:db8::1]:443")
    assert not _matches_any(patterns, "Адрес 10.0.5.21 без порта")


def test_ip_address_regex_registration() -> None:
    patterns = _compiled_patterns_from(register_ip_address)

    assert _matches_any(patterns, "IPv4 10.0.5.21")
    assert _matches_any(patterns, "IPv6 2001:0db8:85a3:0000:0000:8a2e:0370:7334")
    assert _matches_any(patterns, "IPv6 compressed 2001:db8::1")


def test_date_time_regex_registration() -> None:
    patterns = _compiled_patterns_from(register_date_time)

    positive_samples = [
        "Встреча 15 мая 2026 в офисе.",
        "Дата 15.04.2026",
        "Дата 15/04/2026",
        "Дата 2026/04/15",
        "Время 14:00",
        "Займёт 12 часов",
        "Срок 05/2026",
        "ISO 2026-04-15T10:30:00Z",
        "Дата 15 янв. 2026",
        "Дата 15-го января 2026",
        "Период январь 2025",
        "Дата 2026, 15 мая",
        "English January 15th, 2024",
    ]
    for sample in positive_samples:
        assert _matches_any(patterns, sample), f"No DATE_TIME regex matched: {sample}"

    assert not _matches_any(patterns, "номер заказа 1234567890")
    assert not _matches_any(patterns, "СП 2.1.3678-20")
    assert not _matches_any(patterns, "п. 3.2 договора")
    assert not _matches_any(patterns, "версия 1.2.3")
    assert not _matches_any(patterns, "IP 10.0.5.21")


def test_inn_candidate_patterns() -> None:
    patterns = InnClassifier.candidate_patterns

    assert _matches_any(patterns, "ИНН организации 7715964180")
    assert _matches_any(patterns, "ИНН физлица 500100732259")
    assert not _matches_any(patterns, "ИНН 771596418")
    assert not _matches_any(patterns, "ИНН 77159641801")

    matches = _all_matches(patterns, "ИНН 7727855555 123007")
    assert matches == ["7727855555"]


def test_military_id_candidate_patterns() -> None:
    patterns = [MILITARY_ID_RE, MILITARY_SERIES_NUMBER_RE]

    assert _matches_any(patterns, "Военный билет АО-1657262")
    assert _matches_any(patterns, "Военный билет PT7654321")
    assert _matches_any(patterns, "серия PT номер 9876543")
    assert not _matches_any(patterns, "PT765432")
    assert not _matches_any(patterns, "серия PT номер 987654")
    assert not _matches_any(patterns, "ABC 1234567")


def test_birth_certificate_candidate_patterns() -> None:
    assert BIRTH_CERT_NUMBER_RE.search("II-МЮ 950132")
    assert BIRTH_CERT_NUMBER_RE.search("VII-SHSH 333238")
    assert not BIRTH_CERT_NUMBER_RE.search("II-МЮ 95013")
    assert not BIRTH_CERT_NUMBER_RE.search("МЮ 950132")
