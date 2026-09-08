"""Unit tests for conflict resolution on synthetic spans."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pytest

# Importing the entity modules is what registers their conflict rules; the
# triangle test below reads the live rule set rather than a stubbed one.
from pii_guard.entities import date_time as _date_time  # noqa: F401
from pii_guard.entities import ip_address as _ip_address  # noqa: F401
from pii_guard.entities import ip_port as _ip_port  # noqa: F401
from pii_guard.entities import ner_conflicts
from pii_guard.framework import base
from pii_guard.framework.base import BaseEntityClassifier, EntityClassifierRegistry
from pii_guard.framework.conflict_resolver import (
    _resolve_by_score,
    resolve_conflicts,
    resolve_ml_vs_rules_conflicts,
)
from pii_guard.framework.resolvers import resolve_passport_dl_conflict


@dataclass
class _Span:
    entity_type: str
    start: int
    end: int
    score: float = 0.5


class _PassportWinner(BaseEntityClassifier):
    entity_type = "PASSPORT"
    conflict_wins_over = {"PHONE_NUMBER"}

    def classify(
        self,
        raw: str,
        digits: str,
        ctx: str,
        wide_ctx: str | None,
        ctx_pos: int = 0,
        wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        return None


class _PhoneLoser(BaseEntityClassifier):
    entity_type = "PHONE_NUMBER"

    def classify(
        self,
        raw: str,
        digits: str,
        ctx: str,
        wide_ctx: str | None,
        ctx_pos: int = 0,
        wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        return None


@pytest.fixture
def _restore_conflict_globals():
    orig_pairs = list(base._PAIRWISE_CONFLICT_RULES)
    orig_handlers = list(base._CONFLICT_HANDLERS)
    yield
    base._PAIRWISE_CONFLICT_RULES[:] = orig_pairs
    base._CONFLICT_HANDLERS[:] = orig_handlers


def test_resolve_ml_vs_rules_conflicts_keeps_non_overlapping_ner() -> None:
    rules = [_Span("PASSPORT", 0, 10, 0.9)]
    ner = [
        _Span("PERSON", 5, 12, 0.8),
        _Span("EMAIL_ADDRESS", 15, 25, 0.7),
    ]

    resolved = resolve_ml_vs_rules_conflicts(rules, ner)

    assert resolved == [rules[0], ner[1]]


def test_resolve_by_score_prefers_higher_score_and_longer_span() -> None:
    higher_score = _Span("A", 2, 8, 0.9)
    lower_score = _Span("B", 0, 5, 0.8)
    tie_longer = _Span("C", 10, 16, 0.7)
    tie_shorter = _Span("D", 11, 14, 0.7)

    resolved = _resolve_by_score([lower_score, higher_score, tie_shorter, tie_longer])

    assert resolved == [higher_score, tie_longer]


def test_resolve_conflicts_applies_registry_conflict_rules(
    _restore_conflict_globals,
) -> None:
    base._PAIRWISE_CONFLICT_RULES[:] = []
    base._CONFLICT_HANDLERS[:] = []

    registry = EntityClassifierRegistry()
    registry.register(_PassportWinner()).register(_PhoneLoser())

    entities = [
        _Span("PASSPORT", 0, 10, 0.8),
        _Span("PHONE_NUMBER", 5, 12, 0.9),
    ]

    resolved = resolve_conflicts(entities, "паспорт 4510 678234", registry=registry)

    assert resolved == [entities[0]]


def test_resolve_conflicts_applies_same_type_dedup(_restore_conflict_globals) -> None:
    base._PAIRWISE_CONFLICT_RULES[:] = []
    base._CONFLICT_HANDLERS[:] = []

    entities = [
        _Span("URL", 0, 15, 0.6),
        _Span("URL", 5, 10, 0.9),
    ]

    resolved = resolve_conflicts(entities, "https://example.com", registry=EntityClassifierRegistry())

    assert resolved == [entities[0]]


def test_resolve_conflicts_applies_ner_pairwise_priority(
    _restore_conflict_globals,
) -> None:
    base._PAIRWISE_CONFLICT_RULES[:] = [("EMAIL_ADDRESS", "URL")]
    base._CONFLICT_HANDLERS[:] = []

    entities = [
        _Span("EMAIL_ADDRESS", 0, 16, 0.7),
        _Span("URL", 5, 16, 0.9),
    ]

    resolved = resolve_conflicts(entities, "user@example.com", registry=EntityClassifierRegistry())

    assert resolved == [entities[0]]


def test_resolve_conflicts_keeps_ip_port_over_nested_ip_address(
    _restore_conflict_globals,
) -> None:
    base._PAIRWISE_CONFLICT_RULES[:] = [("IP_PORT", "IP_ADDRESS")]
    base._CONFLICT_HANDLERS[:] = []

    entities = [
        _Span("IP_PORT", 7, 22, 0.6),
        _Span("IP_ADDRESS", 7, 16, 0.85),
    ]

    resolved = resolve_conflicts(entities, "сервер 10.0.5.21:8443", registry=EntityClassifierRegistry())

    assert resolved == [entities[0]]


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_ip_port_survives_the_ip_address_date_time_triangle(order) -> None:
    """The three types that can cover one ip:port must not form a conflict cycle.

    `сервер 10.0.5.21:45 работает` produces all three candidates at once: the
    ip:port, the address inside it and `21:45` as a time. The pairwise pass has no
    transitivity, so a cycle among them makes the outcome depend on candidate
    order -- and two orders used to drop all three, putting the address in the
    clear. Registered rules are used as-is on purpose: the point is to fail if a
    future rule closes the cycle again.
    """
    text = "сервер 10.0.5.21:45 работает"
    candidates = {
        "IP_PORT": _Span("IP_PORT", 7, 19, 0.6),
        "IP_ADDRESS": _Span("IP_ADDRESS", 7, 16, 0.85),
        "DATE_TIME": _Span("DATE_TIME", 14, 19, 0.75),
    }
    names = list(candidates)

    resolved = resolve_conflicts(
        [candidates[names[i]] for i in order], text, registry=EntityClassifierRegistry()
    )

    assert [e.entity_type for e in resolved] == ["IP_PORT"]


def test_resolve_conflicts_applies_ner_url_email_adjacency_handler(
    _restore_conflict_globals,
) -> None:
    base._PAIRWISE_CONFLICT_RULES[:] = []
    base._CONFLICT_HANDLERS[:] = [ner_conflicts._handle_url_email_adjacency]

    entities = [_Span("URL", 5, 16, 0.8)]

    resolved = resolve_conflicts(entities, "user@example.com", registry=EntityClassifierRegistry())

    assert resolved == []


@pytest.mark.parametrize(
    ("ctx", "wide_ctx", "expected"),
    [
        ("паспорт гражданина 4510 678234", None, "passport"),
        ("водительские права 4510 678234", None, "driver_license"),
        ("4510 678234", "данные паспорта 4510 678234", "passport"),
        ("4510 678234", None, None),
    ],
)
def test_resolve_passport_dl_conflict(
    ctx: str,
    wide_ctx: str | None,
    expected: str | None,
) -> None:
    raw = "4510 678234"
    ctx_pos = ctx.index(raw)
    wide_ctx_pos = wide_ctx.index(raw) if wide_ctx is not None else 0

    assert (
        resolve_passport_dl_conflict(raw, ctx, wide_ctx, ctx_pos, wide_ctx_pos)
        == expected
    )
