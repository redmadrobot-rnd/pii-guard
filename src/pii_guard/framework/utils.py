"""
Generic text-processing utilities for PII detection.

Only pure, entity-agnostic helpers belong here.
"""

from __future__ import annotations

import re


def digits_only(value: str) -> str:
    """Strip everything except ASCII digits."""
    return re.sub(r"\D", "", value)


def luhn_valid(number: str) -> bool:
    """Luhn check for any digit-only string of length ≥ 10."""
    if not number.isdigit() or len(number) < 10:
        return False
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def drop_spans_without_digits(
    entities: list,
    text: str,
    already_removed: set,
    entity_type: str,
) -> set:
    """Drop entity spans that contain no digits.

    Used as a post-filter in ``@register_conflict_handler`` functions to
    remove NER noise (mis-classified words, syllables, punctuation runs).

    Parameters
    ----------
    entities:
        Full entity list passed to the conflict handler.
    text:
        Original document text.
    already_removed:
        Indices already scheduled for removal — these are skipped.
    entity_type:
        Only spans of this type are examined.

    Returns
    -------
    set[int]
        Indices of spans to remove (to be merged into ``already_removed``
        by the caller).
    """
    extra: set[int] = set()
    for i, ent in enumerate(entities):
        if i in already_removed or ent.entity_type != entity_type:
            continue
        span = text[ent.start:ent.end]
        if not any(ch.isdigit() for ch in span):
            extra.add(i)
    return extra
