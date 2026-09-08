"""
Presidio-compatible recognizer adapter (branch A — numeric / document PII).

Delegates classification to :class:`EntityClassifierRegistry` and
translates results into Presidio ``RecognizerResult`` objects.

Usage with default classifiers::

    from pii_guard.detect import NumericPIIRecognizer
    recognizer = NumericPIIRecognizer()
    results = recognizer.analyze(text, entities=[...])

Usage with a customised registry::

    from pii_guard.detect import default_registry, NumericPIIRecognizer
    registry = default_registry()
    registry.unregister("MY_ENTITY")
    recognizer = NumericPIIRecognizer(registry=registry)
"""

from __future__ import annotations

from presidio_analyzer import EntityRecognizer, RecognizerResult

from pii_guard.framework.base import EntityClassifierRegistry
from pii_guard.framework.context import get_context, get_wide_context
from pii_guard.framework.normalize import normalize_safe
from pii_guard.framework.patterns import build_candidate_patterns
from pii_guard.framework.spans import data_spans
from pii_guard.framework.utils import digits_only

# ================================================================
# Candidate iterator
# ================================================================


def iter_candidates(text: str, registry: EntityClassifierRegistry):
    """Yield regex matches in priority order, de-duplicating by span.

    Patterns are merged automatically:
    1. Entity-specific patterns (from registry, in priority order)
    2. Generic numeric patterns (catch-all)
    """
    patterns = build_candidate_patterns(registry)
    seen: set[tuple[int, int]] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span in seen:
                continue
            seen.add(span)
            yield match


# ================================================================
# Presidio recognizer
# ================================================================


class NumericPIIRecognizer(EntityRecognizer):
    """Presidio-compatible recognizer for numeric / document PII.

    Parameters
    ----------
    registry : EntityClassifierRegistry, optional
        Custom classifier registry.  When ``None``, the default
        registry with all built-in classifiers is used.
    supported_language : str
        Language code (default ``"ru"``).
    supported_entities : list[str], optional
        Explicit entity-type list.  When ``None``, derived from
        *registry*.
    """

    def __init__(
        self,
        registry: EntityClassifierRegistry | None = None,
        supported_language: str = "ru",
        supported_entities: list[str] | None = None,
    ) -> None:
        if registry is None:
            from pii_guard.detect import default_registry

            registry = default_registry()
        self._registry = registry
        super().__init__(
            supported_entities=supported_entities or registry.entity_types,
            name="NumericPIIRecognizer",
            supported_language=supported_language,
        )

    # -- Presidio EntityRecognizer interface --------------------------

    @property
    def registry(self) -> EntityClassifierRegistry:
        """The classifier registry used by this recognizer."""
        return self._registry

    def load(self) -> None:
        pass

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts=None,
    ) -> list[RecognizerResult]:
        # Position-safe normalisation — offsets match original text.
        safe = normalize_safe(text)

        results: list[RecognizerResult] = []
        seen_digits: list[tuple[int, int, str]] = []  # (start, end, digits)

        for match in iter_candidates(safe, self._registry):
            raw = match.group(0)
            d = digits_only(raw)
            ctx, ctx_start = get_context(safe, match.start(), match.end())
            wide_ctx, wide_ctx_start = get_wide_context(safe, match.start(), match.end())
            ctx_pos = match.start() - ctx_start
            wide_ctx_pos = match.start() - wide_ctx_start

            cls = self._registry.classify(raw, d, ctx, wide_ctx, ctx_pos, wide_ctx_pos)
            if cls is None:
                continue
            entity_type, score = cls

            if entity_type not in entities:
                continue

            # Deduplicate overlapping candidates with the same digits.
            skip = False
            for s, e, prev_d in seen_digits:
                overlaps = not (match.end() <= s or e <= match.start())
                if overlaps and prev_d == d:
                    skip = True
                    break
            if skip:
                continue
            seen_digits.append((match.start(), match.end(), d))

            # The candidate is matched by its phrase, but reported as the data it
            # carries: «серия 7518, номер 492137» leaves as two spans over the
            # digits. Classification has already happened on the full candidate,
            # so nothing about detection changes here -- only the offsets.
            for span_start, span_end in data_spans(raw):
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=match.start() + span_start,
                        end=match.start() + span_end,
                        score=score,
                    )
                )
        return results
