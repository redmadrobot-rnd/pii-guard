"""INN (ИНН) classifier — handles both INN-10 and INN-12.

Self-contained entity definition: validator, keyword context, classification — all in one file.
"""

from __future__ import annotations

import re

from pii_guard.framework.base import BaseEntityClassifier, register_entity
from pii_guard.framework.context import has_keyword, nearest_before_distance
from pii_guard.framework.patterns import SERIES_NUMBER_NUMERIC_RE
from pii_guard.framework.resolvers import KW_DL, KW_PASSPORT

ENTITY_TYPE = "INN"

# ── Validators (self-contained) ─────────────────────────────────


def _valid_inn10(inn: str) -> bool:
    """Return *True* if *inn* (exactly 10 digits) passes the INN-10 checksum."""
    if len(inn) != 10 or not inn.isdigit():
        return False
    weights = [2, 4, 10, 3, 5, 9, 4, 6, 8]
    checksum = (
        sum(int(ch) * w for ch, w in zip(inn[:9], weights, strict=False)) % 11
    ) % 10
    return checksum == int(inn[9])


def _valid_inn12(inn: str) -> bool:
    """Return *True* if *inn* (exactly 12 digits) passes the INN-12 checksum."""
    if len(inn) != 12 or not inn.isdigit():
        return False
    w1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    w2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    c1 = (sum(int(ch) * w for ch, w in zip(inn[:10], w1, strict=False)) % 11) % 10
    c2 = (sum(int(ch) * w for ch, w in zip(inn[:11], w2, strict=False)) % 11) % 10
    return c1 == int(inn[10]) and c2 == int(inn[11])

# ── Keyword patterns ────────────────────────────────────────────

# Только целое слово. `инн\w{0,8}` без границ срабатывал внутри слова
# («длинный», «минный», «старинные»), а с границей слева — на «инновации».
# Ключ поднимает скор с 0.85 до 0.90, а `_resolve_by_score` разрешает
# пересечения именно по скору, так что это не косметика.
KW_INN = re.compile(
    r"(?iu)\bинн\b|налогов\w{0,12}\s+номер\w{0,8}"
)

KW_SERIES = re.compile(r"(?iu)\bсер(?:ия)?\b")
KW_UDOST = re.compile(r"(?iu)\bудостоверен\w{0,12}\b")


@register_entity
class InnClassifier(BaseEntityClassifier):
    """INN-12 (individual, 12 digits) and INN-10 (legal entity, 10 digits).

    INN-12 is unambiguous (unique length).  INN-10 competes with
    passport / driver-license — keyword context decides.
    """

    entity_type = ENTITY_TYPE
    priority = 40
    conflict_wins_over = {"PHONE_NUMBER"}
    candidate_patterns = [
        # Strict contiguous-digit patterns — tried before NUM_CANDIDATE_RE to
        # prevent the generic pattern from greedily merging an INN with an
        # adjacent digit group (e.g. postal code: "ИНН 7727855555 123007").
        re.compile(r"(?<!\d)\d{10}(?!\d)"),  # INN-10 (legal entity)
        re.compile(r"(?<!\d)\d{12}(?!\d)"),  # INN-12 (individual)
    ]

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        length = len(digits)

        # ── INN-12 (no competition) ─────────────────────────────
        if length == 12:
            if _valid_inn12(digits):
                return self.entity_type, 0.95
            return None

        # ── INN-10 (competes with passport / DL) ────────────────
        if length != 10 or not _valid_inn10(digits):
            return None

        # Gather context signals
        has_passport = has_keyword(KW_PASSPORT, ctx)
        has_udost = has_keyword(KW_UDOST, ctx)
        has_series = has_keyword(KW_SERIES, ctx)
        if wide_ctx:
            has_passport = has_passport or has_keyword(KW_PASSPORT, wide_ctx)
            has_udost = has_udost or has_keyword(KW_UDOST, wide_ctx)
            has_series = has_series or has_keyword(KW_SERIES, wide_ctx)

        stripped = raw.strip()
        # «серия» nearby, or the value itself shaped like a series+number pair.
        # This used to also test «номер»/«№» next to «серия», via
        # `has_series or ((has_series and (has_number or has_num_sign)) or RE)` --
        # which reduces to `has_series or RE`, so those four keyword lookups could
        # not change the outcome.
        has_series_marker = (
            has_series or SERIES_NUMBER_NUMERIC_RE.fullmatch(stripped) is not None
        )
        has_passport_or_udost = has_passport or has_udost

        # Directional signals
        inn_before = nearest_before_distance(KW_INN, ctx, ctx_pos, max_dist=80)
        passport_before = nearest_before_distance(KW_PASSPORT, ctx, ctx_pos, max_dist=80)
        dl_before = nearest_before_distance(KW_DL, ctx, ctx_pos, max_dist=80)
        if wide_ctx:
            if inn_before is None:
                inn_before = nearest_before_distance(KW_INN, wide_ctx, wide_ctx_pos, max_dist=160)
            if passport_before is None:
                passport_before = nearest_before_distance(KW_PASSPORT, wide_ctx, wide_ctx_pos, max_dist=160)
            if dl_before is None:
                dl_before = nearest_before_distance(KW_DL, wide_ctx, wide_ctx_pos, max_dist=160)

        # INN keyword is nearest before → INN
        if inn_before is not None:
            nearest = "inn"
            nearest_dist = inn_before
            if passport_before is not None and passport_before <= nearest_dist:
                nearest = "passport"
                nearest_dist = passport_before
            if dl_before is not None and dl_before <= nearest_dist:
                nearest = "driver_license"
            if nearest == "inn":
                return self.entity_type, 0.90

        # No series / passport / identity markers → INN
        if not has_series_marker and not has_passport_or_udost and dl_before is None:
            return self.entity_type, 0.85

        return None
