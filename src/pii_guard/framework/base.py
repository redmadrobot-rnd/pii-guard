"""
Base class, registry, and auto-discovery for pluggable entity classifiers.

Framework
---------
1. Create a file under ``pii_guard/entities/`` (e.g. ``my_entity.py``)
2. Define a subclass of :class:`BaseEntityClassifier`
3. Decorate it with :func:`register_entity`
4. Done — the entity is auto-discovered and registered.

Example::

    from pii_guard.framework.base import BaseEntityClassifier, register_entity

    @register_entity
    class MyEntityClassifier(BaseEntityClassifier):
        entity_type = "MY_ENTITY"
        priority = 55

        candidate_patterns = [MY_PATTERN_RE]
        conflict_wins_over = {"OTHER_ENTITY"}

        def classify(self, raw, digits, ctx, wide_ctx, ctx_pos=0, wide_ctx_pos=0):
            if len(digits) == 8 and my_checksum(digits):
                return self.entity_type, 0.90
            return None

For regex-based Presidio ``PatternRecognizer`` entities::

    from pii_guard.framework.base import register_regex_entity

    @register_regex_entity
    def register_my_regex(analyzer, language="ru"):
        analyzer.registry.add_recognizer(...)

No changes to framework files are required — everything wires
automatically.
"""

from __future__ import annotations

import importlib
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

# ── Abstract classifier ────────────────────────────────────────


class BaseEntityClassifier(ABC):
    """Base class for all entity classifiers.

    Subclass attributes
    -------------------
    entity_type : str
        Presidio-compatible entity type string.
    priority : int
        Lower values are tried first.  Pattern-based classifiers
        use low priorities; digit-based classifiers use higher values.
    candidate_patterns : list[re.Pattern]
        Regex patterns that extract candidates specific to this entity.
        They are merged into the global candidate list automatically
        (before the generic numeric patterns).
    conflict_wins_over : set[str]
        Entity types that this entity beats when their spans overlap
        during conflict resolution.
    """

    entity_type: str = ""
    priority: int = 100
    candidate_patterns: ClassVar[list[re.Pattern]] = []
    conflict_wins_over: ClassVar[set[str]] = set()

    @abstractmethod
    def classify(
        self,
        raw: str,
        digits: str,
        ctx: str,
        wide_ctx: str | None,
        ctx_pos: int = 0,
        wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        """Classify a candidate.

        Parameters
        ----------
        raw : str
            Raw matched text (may contain spaces, dashes, letters).
        digits : str
            Digit-only version of *raw*.
        ctx : str
            Sentence-local context around the candidate.
        wide_ctx : str | None
            Wider context window (≈260 chars each side).
        ctx_pos : int
            Start position of the candidate within *ctx*.
        wide_ctx_pos : int
            Start position of the candidate within *wide_ctx*.

        Returns
        -------
        tuple[str, float] | None
            ``(entity_type, score)`` on match, or ``None`` to let other
            classifiers try.
        """
        ...


# ── Global class registry (numeric classifiers) ────────────────

_REGISTERED_CLASSES: list[type[BaseEntityClassifier]] = []


def register_entity(
    cls: type[BaseEntityClassifier],
) -> type[BaseEntityClassifier]:
    """Class decorator — registers an entity classifier for auto-discovery.

    Usage::

        @register_entity
        class MyClassifier(BaseEntityClassifier):
            entity_type = "MY_ENTITY"
            priority = 30

            def classify(self, raw, digits, ctx, wide_ctx, ctx_pos=0, wide_ctx_pos=0):
                ...
    """
    if cls not in _REGISTERED_CLASSES:
        _REGISTERED_CLASSES.append(cls)
    return cls


# ── Global factory registry (regex / PatternRecognizer entities) ─

_REGISTERED_REGEX_FACTORIES: list[Callable] = []


def register_regex_entity(func: Callable) -> Callable:
    """Function decorator — registers a regex-entity factory.

    The decorated function must accept ``(analyzer, language)`` and
    register one or more ``PatternRecognizer`` instances on the
    ``AnalyzerEngine``.

    Usage::

        @register_regex_entity
        def register_my_regex(analyzer, language="ru"):
            analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="MY_REGEX_ENTITY", ...,
                )
            )
    """
    if func not in _REGISTERED_REGEX_FACTORIES:
        _REGISTERED_REGEX_FACTORIES.append(func)
    return func


# ── Conflict resolution declarations ───────────────────────────

_PAIRWISE_CONFLICT_RULES: list[tuple[str, str]] = []
_CONFLICT_HANDLERS: list[Callable] = []


def register_pairwise_conflict(winner: str, loser: str) -> None:
    """Declare that *winner* beats *loser* when their spans overlap.

    Call from entity files to register conflict resolution rules.
    The framework resolves these generically — no entity-specific
    knowledge is needed in ``conflict_resolver.py``.
    """
    pair = (winner, loser)
    if pair not in _PAIRWISE_CONFLICT_RULES:
        _PAIRWISE_CONFLICT_RULES.append(pair)


def register_conflict_handler(fn: Callable) -> Callable:
    """Decorator — registers a custom conflict resolution handler.

    The decorated function must accept ``(entities, text, already_removed)``
    and return a ``set[int]`` of additional entity indices to remove.

    Use for complex rules that go beyond simple overlap (e.g.,
    adjacency checks, text-context analysis).
    """
    if fn not in _CONFLICT_HANDLERS:
        _CONFLICT_HANDLERS.append(fn)
    return fn


# ── Registry ───────────────────────────────────────────────────


class EntityClassifierRegistry:
    """Ordered collection of entity classifiers.

    Classifiers are tried in priority order (lower first).
    The classifier returning the **highest score** wins.

    Typical lifecycle::

        from pii_guard.detect import default_registry
        registry = default_registry()
        registry.unregister("MY_ENTITY")
        registry.register(MyCustom())
    """

    def __init__(self) -> None:
        self._classifiers: list[BaseEntityClassifier] = []

    # ── mutation ────────────────────────────────────────────────

    def register(
        self, classifier: BaseEntityClassifier,
    ) -> EntityClassifierRegistry:
        """Add a classifier and re-sort by priority.  Returns *self*."""
        self._classifiers.append(classifier)
        self._classifiers.sort(key=lambda c: c.priority)
        return self

    def unregister(self, entity_type: str) -> EntityClassifierRegistry:
        """Remove **all** classifiers for *entity_type*.  Returns *self*."""
        self._classifiers = [
            c for c in self._classifiers if c.entity_type != entity_type
        ]
        return self

    # ── introspection ───────────────────────────────────────────

    @property
    def entity_types(self) -> list[str]:
        """Unique entity types currently registered (in priority order)."""
        return list(dict.fromkeys(c.entity_type for c in self._classifiers))

    def __len__(self) -> int:
        return len(self._classifiers)

    # ── pattern & conflict introspection ────────────────────────

    def all_candidate_patterns(self) -> list[re.Pattern]:
        """Candidate-extraction patterns contributed by all classifiers.

        Returned in classifier-priority order, de-duplicated by identity.
        These should be prepended to the generic numeric patterns.
        """
        patterns: list[re.Pattern] = []
        seen: set[int] = set()
        for c in self._classifiers:
            for p in c.candidate_patterns:
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    patterns.append(p)
        return patterns

    def conflict_rules(self) -> dict[str, set[str]]:
        """Mapping: ``entity_type → set of types it wins over on overlap``."""
        rules: dict[str, set[str]] = {}
        for c in self._classifiers:
            if c.conflict_wins_over:
                rules.setdefault(c.entity_type, set()).update(
                    c.conflict_wins_over,
                )
        return rules

    # ── classification ──────────────────────────────────────────

    def classify(
        self,
        raw: str,
        digits: str,
        ctx: str,
        wide_ctx: str | None,
        ctx_pos: int = 0,
        wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        """Try all registered classifiers, return the highest-scoring result.

        Results are based solely on digit length, checksums, strict regex
        patterns, and keyword context.  If no classifier matches, returns
        ``None``.
        """
        best: tuple[str, float] | None = None
        for classifier in self._classifiers:
            result = classifier.classify(
                raw, digits, ctx, wide_ctx, ctx_pos, wide_ctx_pos
            )
            if result is not None and (best is None or result[1] >= best[1]):
                best = result

        return best


# ── Auto-discovery ──────────────────────────────────────────────


def auto_discover_entities(
    package_path: list[str], package_name: str,
) -> None:
    """Import all ``.py`` modules in *package_path* to trigger decorators.

    Called once by ``pii_guard.entities.__init__`` on first import.
    Scans the ``entities/`` directory and imports every non-private
    Python module, which causes ``@register_entity`` and
    ``@register_regex_entity`` decorators to fire.
    """
    for path_entry in package_path:
        for filename in sorted(os.listdir(path_entry)):
            if filename.startswith("_") or not filename.endswith(".py"):
                continue
            module_name = f"{package_name}.{filename[:-3]}"
            importlib.import_module(module_name)
