"""
Entity conflict resolution — fully generic.

All entity-specific rules are declared in entity files via:

- ``conflict_wins_over`` on ``BaseEntityClassifier`` subclasses
- ``register_pairwise_conflict(winner, loser)``
- ``@register_conflict_handler``

This module contains **NO** entity-specific knowledge.
"""

from __future__ import annotations

from pii_guard.framework.base import EntityClassifierRegistry


def resolve_ml_vs_rules_conflicts(
    rules_results: list,
    ner_results: list,
) -> list:
    """Merge rules and NER results with rules taking priority on overlap.

    Scenarios
    ---------
    1. **Conflict** (spans overlap): rules win — the overlapping NER result
       is dropped.
    2. **NER found, rules missed** (no overlap): NER result is kept.
    3. **Symbiosis** (no overlaps at all): everything from both branches
       is returned.
    4. **Both silent**: empty list returned.

    Parameters
    ----------
    rules_results : list
        Entities produced by the rules pipeline (``NumericPIIRecognizer``
        + ``PatternRecognizer`` instances).
    ner_results : list
        Entities produced by ``TransformerNERRecognizer``.

    Returns
    -------
    list
        Merged list of ``RecognizerResult`` objects ready for the
        existing conflict-resolution passes (Pass 1-5).
    """

    def _overlaps(a, b) -> bool:
        return not (a.end <= b.start or b.end <= a.start)

    # Rules always kept as-is.
    merged: list = list(rules_results)

    # Keep NER results that do NOT overlap with any rules result.
    for ner_ent in ner_results:
        conflicts = any(_overlaps(ner_ent, r) for r in rules_results)
        if not conflicts:
            merged.append(ner_ent)

    return merged


def resolve_conflicts(
    entities: list,
    text: str,
    *,
    registry: EntityClassifierRegistry | None = None,
) -> list:
    """Return a filtered copy of *entities* with conflicts resolved.

    Resolution order
    ----------------
    1. ``conflict_wins_over`` rules from the classifier registry.
    2. Pairwise rules declared via ``register_pairwise_conflict``.
    3. Custom handlers registered via ``@register_conflict_handler``.
    4. Same-type deduplication (keep the longer span).
    5. Score-based overlap resolution (keep the higher score).
    """
    from pii_guard.framework.base import (
        _CONFLICT_HANDLERS,
        _PAIRWISE_CONFLICT_RULES,
    )

    if registry is None:
        from pii_guard.detect import default_registry
        registry = default_registry()

    result = list(entities)
    to_remove: set[int] = set()

    # -- helpers ----------------------------------------------------------

    def overlaps(e1, e2) -> bool:
        return not (e1.end <= e2.start or e2.end <= e1.start)

    # -- Pass 1: conflict_wins_over from registry -------------------------

    conflict_winners = registry.conflict_rules()
    for i, e1 in enumerate(result):
        if i in to_remove:
            continue
        if e1.entity_type in conflict_winners:
            for j, e2 in enumerate(result):
                if (
                    j != i
                    and j not in to_remove
                    and e2.entity_type in conflict_winners[e1.entity_type]
                    and overlaps(e1, e2)
                ):
                    to_remove.add(j)

    # -- Pass 2: pairwise conflict rules ----------------------------------

    pairwise_map: dict[tuple[str, str], str] = {}
    for winner, loser in _PAIRWISE_CONFLICT_RULES:
        pairwise_map[(winner, loser)] = winner
        pairwise_map[(loser, winner)] = winner

    for i, e1 in enumerate(result):
        if i in to_remove:
            continue
        for j, e2 in enumerate(result):
            if j <= i or j in to_remove:
                continue
            pair = (e1.entity_type, e2.entity_type)
            if pair in pairwise_map and overlaps(e1, e2):
                winner_type = pairwise_map[pair]
                loser_idx = j if e1.entity_type == winner_type else i
                to_remove.add(loser_idx)
                if loser_idx == i:
                    break

    # -- Pass 3: custom conflict handlers ---------------------------------

    for handler in _CONFLICT_HANDLERS:
        extra = handler(result, text, to_remove)
        to_remove.update(extra)

    # -- Pass 4: same-type dedup (keep longer span) -----------------------

    for i, e1 in enumerate(result):
        if i in to_remove:
            continue
        for j, e2 in enumerate(result):
            if j <= i or j in to_remove:
                continue
            if e1.entity_type == e2.entity_type and overlaps(e1, e2):
                l1 = e1.end - e1.start
                l2 = e2.end - e2.start
                # `j > i` always holds -- the loop above skips `j <= i` -- so equal
                # lengths deterministically keep the earlier span and drop `j`.
                if l1 < l2:
                    to_remove.add(i)
                    break
                else:
                    to_remove.add(j)

    # -- Remove -----------------------------------------------------------

    for idx in sorted(to_remove, reverse=True):
        result.pop(idx)

    # -- Pass 5: score-based overlap resolution ---------------------------

    return _resolve_by_score(result)


def _resolve_by_score(entities: list) -> list:
    """Drop lower-scored entities when spans overlap."""
    if len(entities) <= 1:
        return entities

    ranked = sorted(
        enumerate(entities),
        key=lambda ie: (
            -(getattr(ie[1], "score", 0) or 0),
            -(ie[1].end - ie[1].start),
            ie[1].start,
        ),
    )

    keep_indices: set[int] = set()
    kept: list = []

    for orig_idx, ent in ranked:
        dominated = False
        for accepted in kept:
            if not (ent.end <= accepted.start or accepted.end <= ent.start):
                dominated = True
                break
        if not dominated:
            keep_indices.add(orig_idx)
            kept.append(ent)

    return [e for i, e in enumerate(entities) if i in keep_indices]
