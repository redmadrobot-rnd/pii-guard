"""Unit tests for framework core helpers and registries."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from pii_guard.framework import base, patterns
from pii_guard.framework.base import BaseEntityClassifier, EntityClassifierRegistry
from pii_guard.framework.utils import digits_only, drop_spans_without_digits


def test_digits_only_strips_non_digit_characters() -> None:
    assert digits_only("№ 45-10 678234 ext. 99") == "451067823499"
    assert digits_only("abc") == ""


@dataclass
class _Entity:
    entity_type: str
    start: int
    end: int


def test_drop_spans_without_digits_filters_only_target_entity_type() -> None:
    text = "НЛ 1234 email"
    entities = [
        _Entity("MILITARY_ID", 0, 2),
        _Entity("MILITARY_ID", 3, 7),
        _Entity("EMAIL_ADDRESS", 8, 13),
    ]

    assert drop_spans_without_digits(entities, text, set(), "MILITARY_ID") == {0}


def test_drop_spans_without_digits_skips_already_removed() -> None:
    text = "НЛ АБ"
    entities = [
        _Entity("MILITARY_ID", 0, 2),
        _Entity("MILITARY_ID", 3, 5),
    ]

    assert drop_spans_without_digits(
        entities, text, already_removed={1}, entity_type="MILITARY_ID"
    ) == {0}


class _ClassifierLowPriority(BaseEntityClassifier):
    entity_type = "LOW"
    priority = 20
    candidate_patterns = [re.compile(r"low"), re.compile(r"shared")]
    conflict_wins_over = {"OTHER"}

    def classify(
        self,
        raw: str,
        digits: str,
        ctx: str,
        wide_ctx: str | None,
        ctx_pos: int = 0,
        wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        return self.entity_type, 0.2


class _ClassifierHighPriority(BaseEntityClassifier):
    entity_type = "HIGH"
    priority = 10
    candidate_patterns = [re.compile(r"high"), _ClassifierLowPriority.candidate_patterns[1]]

    def classify(
        self,
        raw: str,
        digits: str,
        ctx: str,
        wide_ctx: str | None,
        ctx_pos: int = 0,
        wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        return self.entity_type, 0.9


def test_build_candidate_patterns_keeps_entity_patterns_first() -> None:
    registry = EntityClassifierRegistry()
    registry.register(_ClassifierLowPriority()).register(_ClassifierHighPriority())

    compiled = patterns.build_candidate_patterns(registry)

    assert [pattern.pattern for pattern in compiled[:3]] == ["high", "shared", "low"]
    assert compiled[-3:] == patterns.GENERIC_CANDIDATE_PATTERNS


@pytest.fixture
def _restore_base_globals():
    orig_classes = list(base._REGISTERED_CLASSES)
    orig_pairs = list(base._PAIRWISE_CONFLICT_RULES)
    yield
    base._REGISTERED_CLASSES[:] = orig_classes
    base._PAIRWISE_CONFLICT_RULES[:] = orig_pairs


class _RegisteredClassifier(BaseEntityClassifier):
    entity_type = "REGISTERED"

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


def test_register_entity_adds_classifier_once(_restore_base_globals) -> None:
    base.register_entity(_RegisteredClassifier)
    base.register_entity(_RegisteredClassifier)

    assert base._REGISTERED_CLASSES.count(_RegisteredClassifier) == 1


def test_register_pairwise_conflict_adds_rule_once(_restore_base_globals) -> None:
    base.register_pairwise_conflict("A", "B")
    base.register_pairwise_conflict("A", "B")

    assert base._PAIRWISE_CONFLICT_RULES.count(("A", "B")) == 1


def test_auto_discover_entities_imports_public_python_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    monkeypatch.setattr(
        base.os,
        "listdir",
        lambda _path: ["z_module.py", "_private.py", "notes.txt", "a_module.py"],
    )
    monkeypatch.setattr(
        base.importlib,
        "import_module",
        lambda module_name: imported.append(module_name),
    )

    base.auto_discover_entities(["/fake/path"], "pii_guard.entities")

    assert imported == [
        "pii_guard.entities.a_module",
        "pii_guard.entities.z_module",
    ]
