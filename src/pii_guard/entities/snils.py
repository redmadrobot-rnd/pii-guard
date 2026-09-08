"""SNILS (СНИЛС) classifier.

Self-contained entity definition: validator, classification — all in one file.
"""

from __future__ import annotations

from pii_guard.framework.base import BaseEntityClassifier, register_entity

ENTITY_TYPE = "SNILS"


# ── Validator (self-contained) ──────────────────────────────────

def _valid_snils(snils: str) -> bool:
    """Return *True* if *snils* (11 digits) passes the control-number check."""
    if len(snils) != 11 or not snils.isdigit():
        return False
    base = snils[:9]
    control = int(snils[9:])

    s = sum(int(base[i]) * (9 - i) for i in range(9))
    if s < 100:
        expected = s
    elif s in (100, 101):
        expected = 0
    else:
        expected = s % 101
        if expected == 100:
            expected = 0
    return control == expected


@register_entity
class SnilsClassifier(BaseEntityClassifier):
    """11 digits + SNILS control-number checksum.

    Example::

        112-233-445 95
    """

    entity_type = ENTITY_TYPE
    priority = 30
    conflict_wins_over = {"PHONE_NUMBER"}

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        if len(digits) != 11:
            return None
        if _valid_snils(digits):
            return self.entity_type, 0.95
        return None
