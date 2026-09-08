"""Credit card (банковская карта) classifier.

Self-contained entity definition: validator, keyword context, classification — all in one file.
"""

from __future__ import annotations

import re

from pii_guard.entities._card_common import KW_CARD
from pii_guard.framework.base import BaseEntityClassifier, register_entity
from pii_guard.framework.context import has_keyword, nearest_before_distance
from pii_guard.framework.patterns import KW_OMS
from pii_guard.framework.utils import luhn_valid

ENTITY_TYPE = "CREDIT_CARD"

# ── Keyword patterns ────────────────────────────────────────────


KW_CARD_WEAK = re.compile(
    r"(?iu)"
    r"\bкарт\w{0,12}\b"
    r"|\bномер\w{0,8}\s+карт\w{0,12}\b"
    r"|\bлогин\w{0,8}\b"
    r"|\bпарол\w{0,8}\b"
    r"|\bкод\w{0,8}\b"
    r"|\bсписан\w{0,8}\b"
    r"|\bначислен\w{0,8}\b"
    r"|\bтариф\w{0,8}\b"
    r"|\bпин\w{0,8}\b"
    r"|\bpin\b"
    r"|\bsms[-\s]?код\w{0,8}\b"
    r"|\bоплат\w{0,12}\b"
    r"|\bплат[её]ж\w{0,12}\b"
    r"|\bперевод\w{0,12}\b"
    r"|\bпредавторизац\w{0,12}\b"
    r"|\bсписыва\w{0,12}\b"
    r"|\bарендн\w{0,12}\s+плат\w{0,12}\b"
)


@register_entity
class CreditCardClassifier(BaseEntityClassifier):
    """Bank card numbers — 13 / 15 / 16 / 18 / 19 digits + Luhn.

    For 16-digit candidates the classifier competes with OMS; keyword
    proximity decides.  For other lengths there is no competition.
    """

    entity_type = ENTITY_TYPE
    priority = 50
    conflict_wins_over = {"PHONE_NUMBER"}

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        length = len(digits)

        # ── Non-ambiguous lengths ───────────────────────────────
        if length in (13, 15, 18, 19):
            if luhn_valid(digits):
                return self.entity_type, 0.90
            return None

        # ── 16-digit: card vs OMS ───────────────────────────────
        if length != 16 or not luhn_valid(digits):
            return None

        oms_before = nearest_before_distance(KW_OMS, ctx, ctx_pos, max_dist=100)
        card_before = nearest_before_distance(KW_CARD, ctx, ctx_pos, max_dist=100)
        card_weak_before = nearest_before_distance(KW_CARD_WEAK, ctx, ctx_pos, max_dist=100)
        if wide_ctx:
            if oms_before is None:
                oms_before = nearest_before_distance(KW_OMS, wide_ctx, wide_ctx_pos, max_dist=220)
            if card_before is None:
                card_before = nearest_before_distance(KW_CARD, wide_ctx, wide_ctx_pos, max_dist=220)
            if card_weak_before is None:
                card_weak_before = nearest_before_distance(KW_CARD_WEAK, wide_ctx, wide_ctx_pos, max_dist=220)

        has_oms = oms_before is not None
        has_card = card_before is not None
        has_weak = card_weak_before is not None

        # Only card keywords before → card
        if has_card and not has_oms:
            return self.entity_type, 0.90

        # Both present → card only claims when closer
        if has_oms and has_card:
            if card_before < oms_before:  # type: ignore[operator]
                return self.entity_type, 0.85
            if card_before == oms_before:  # type: ignore[operator]
                return self.entity_type, 0.60  # ambiguous default
            return None  # OMS closer → let OMS claim

        # Only OMS before → let OMS
        if has_oms and not has_card:
            return None

        # Weak card keywords before
        if has_weak:
            return self.entity_type, 0.80

        # Whole-context fallback
        ctx_card = has_keyword(KW_CARD, ctx)
        ctx_weak = has_keyword(KW_CARD_WEAK, ctx)
        ctx_oms = has_keyword(KW_OMS, ctx)
        if wide_ctx and not ctx_card and not ctx_oms:
            ctx_card = has_keyword(KW_CARD, wide_ctx)
            ctx_weak = ctx_weak or has_keyword(KW_CARD_WEAK, wide_ctx)
            ctx_oms = has_keyword(KW_OMS, wide_ctx)
        if not ctx_card and ctx_weak and not ctx_oms:
            ctx_card = True
        if ctx_card and not ctx_oms:
            return self.entity_type, 0.85
        if ctx_oms:
            return None  # let OMS handle

        return None
