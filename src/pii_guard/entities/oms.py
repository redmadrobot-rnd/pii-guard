"""OMS (полис ОМС) classifier.

Self-contained entity definition: validator, keyword context, classification — all in one file.
"""

from __future__ import annotations

from pii_guard.entities._card_common import KW_CARD
from pii_guard.framework.base import BaseEntityClassifier, register_entity
from pii_guard.framework.context import has_keyword, nearest_before_distance
from pii_guard.framework.patterns import KW_OMS
from pii_guard.framework.utils import luhn_valid

ENTITY_TYPE = "OMS"


@register_entity
class OmsClassifier(BaseEntityClassifier):
    """OMS medical insurance policy — 16 digits + Luhn + OMS keywords.

    Competes with credit-card classifier for 16-digit candidates.
    Returns a result only when OMS keyword context is present.
    """

    entity_type = ENTITY_TYPE
    priority = 51  # just after credit_card
    conflict_wins_over = {"PHONE_NUMBER"}

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        if len(digits) != 16 or not luhn_valid(digits):
            return None

        oms_before = nearest_before_distance(KW_OMS, ctx, ctx_pos, max_dist=100)
        card_before = nearest_before_distance(KW_CARD, ctx, ctx_pos, max_dist=100)
        if wide_ctx:
            if oms_before is None:
                oms_before = nearest_before_distance(KW_OMS, wide_ctx, wide_ctx_pos, max_dist=220)
            if card_before is None:
                card_before = nearest_before_distance(KW_CARD, wide_ctx, wide_ctx_pos, max_dist=220)

        has_oms = oms_before is not None
        has_card = card_before is not None

        # Only OMS keywords before → OMS
        if has_oms and not has_card:
            return self.entity_type, 0.90

        # Both present → OMS only if closer
        if has_oms and has_card:
            if oms_before < card_before:  # type: ignore[operator]
                return self.entity_type, 0.85
            return None  # card is closer or equal

        # Whole-context fallback
        ctx_oms = has_keyword(KW_OMS, ctx)
        ctx_card = has_keyword(KW_CARD, ctx)
        if wide_ctx and not ctx_oms and not ctx_card:
            ctx_oms = has_keyword(KW_OMS, wide_ctx)
            ctx_card = has_keyword(KW_CARD, wide_ctx)
        if ctx_oms and not ctx_card:
            return self.entity_type, 0.85

        return None
