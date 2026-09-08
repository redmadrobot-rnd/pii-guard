"""
PII detection pipeline — single entry point.

Triggers auto-discovery of all entity classifiers (numeric and regex)
from ``pii_guard.entities.*`` and provides the public API consumed
by :class:`~pii_guard.engine.Engine`.

Architecture
------------
``pii_guard/``
    ``entities/``   — self-contained entity definitions (one file per entity)
    ``framework/``  — generic infrastructure (no entity knowledge)
    ``detect.py``   — **this file** (entry point)

Adding a new entity
-------------------
1. Create ``pii_guard/entities/my_entity.py``
2. Decorate the classifier with ``@register_entity`` (numeric/document)
   or ``@register_regex_entity`` (regex / PatternRecognizer).
3. Done — no other file needs changing.
"""

from __future__ import annotations

# ── Trigger auto-discovery of all entity modules ────────────────
import pii_guard.entities  # noqa: F401
from pii_guard.framework.base import (
    _REGISTERED_CLASSES,
    _REGISTERED_REGEX_FACTORIES,
    BaseEntityClassifier,
    EntityClassifierRegistry,
    register_conflict_handler,
    register_entity,
    register_pairwise_conflict,
    register_regex_entity,
)
from pii_guard.framework.conflict_resolver import resolve_conflicts, resolve_ml_vs_rules_conflicts
from pii_guard.framework.recognizer import NumericPIIRecognizer

# ── Public API ──────────────────────────────────────────────────


def default_registry() -> EntityClassifierRegistry:
    """Build a registry pre-loaded with all auto-discovered classifiers."""
    registry = EntityClassifierRegistry()
    for cls in _REGISTERED_CLASSES:
        registry.register(cls())
    return registry


def register_regex_recognizers(
    analyzer,
    language: str = "ru",
) -> None:
    """Register all auto-discovered regex entities on *analyzer*.

    Each ``@register_regex_entity``-decorated factory in ``entities/``
    is called with ``(analyzer, language)``.
    """
    for factory in _REGISTERED_REGEX_FACTORIES:
        factory(analyzer, language)


__all__ = [
    # core classes
    "BaseEntityClassifier",
    "EntityClassifierRegistry",
    "NumericPIIRecognizer",
    # decorators
    "register_entity",
    "register_regex_entity",
    "register_pairwise_conflict",
    "register_conflict_handler",
    # factories
    "default_registry",
    "register_regex_recognizers",
    # conflict resolution
    "resolve_conflicts",
    "resolve_ml_vs_rules_conflicts",
]
