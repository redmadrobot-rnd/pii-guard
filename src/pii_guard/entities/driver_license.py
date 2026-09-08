"""Driver license (водительское удостоверение) classifier.

Self-contained entity definition: keywords, conflict resolution,
classification — all in one file.
"""

from __future__ import annotations

from pii_guard.framework.base import BaseEntityClassifier, register_entity
from pii_guard.framework.resolvers import (
    KW_DL,
    KW_PASSPORT,
    document_keyword_nearby,
    resolve_passport_dl_conflict,
)

ENTITY_TYPE = "DRIVER_LICENSE"


# ── Classifier ──────────────────────────────────────────────────


@register_entity
class DriverLicenseClassifier(BaseEntityClassifier):
    """Russian driver license — 10 digits + DL context keywords.

    Competes with passport (same digit length); keyword proximity
    resolves conflicts.
    """

    entity_type = ENTITY_TYPE
    priority = 70
    conflict_wins_over = {"PHONE_NUMBER"}

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        if len(digits) != 10:
            return None

        has_passport = document_keyword_nearby(
            KW_PASSPORT, raw, ctx, wide_ctx, ctx_pos, wide_ctx_pos
        )
        has_dl = document_keyword_nearby(KW_DL, raw, ctx, wide_ctx, ctx_pos, wide_ctx_pos)

        if has_dl and not has_passport:
            return self.entity_type, 0.80

        if has_passport and has_dl:
            resolved = resolve_passport_dl_conflict(raw, ctx, wide_ctx, ctx_pos, wide_ctx_pos)
            if resolved == "driver_license":
                return self.entity_type, 0.75

        return None
