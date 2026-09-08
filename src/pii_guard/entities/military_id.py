"""Military ID (военный билет) classifier.

Self-contained entity definition: patterns, keywords, classification — all in one file.
"""

from __future__ import annotations

import re

from pii_guard.framework.base import BaseEntityClassifier, register_entity
from pii_guard.framework.context import nearest_before_distance

ENTITY_TYPE = "MILITARY_ID"

# ── Candidate extraction patterns ───────────────────────────────

_LAT_RU_UNIT = r"(?:SHCH|ZH|KH|TS|CH|SH|YU|YA|YE|YO|JO|IO|EH|[A-Z])"
_MIL_SERIES = rf"(?:[А-ЯЁ]{{2}}|(?:{_LAT_RU_UNIT}){{2}})"

MILITARY_ID_RE = re.compile(
    rf"(?iu)\b(?:№\s*)?(?:{_MIL_SERIES})\s*(?:[-\u2013\u2014]\s*)?(?:№|N)?\s*(?:\d[ \t\n/._\\-]?){{6}}\d\b"
)

MILITARY_SERIES_NUMBER_RE = re.compile(
    rf"(?iu)\b(?:сер(?:ия)?\s*)?(?:{_MIL_SERIES})"
    rf"[\s,;:\.\-]*"
    rf"(?:номер|№|N)\s*(?:\d[ \t\n/._\\-]?){{6}}\d\b"
)

MILITARY_STRICT_RE = re.compile(
    rf"(?iu)^(?:№\s*)?(?P<series>{_MIL_SERIES})\s*(?:-| )\s*(?:№\s*)?(?P<number>\d{{7}})$"
)

MILITARY_AMBIG_SERIES = {"РФ", "RF", "ИИ", "II"}

# ── Keyword patterns ────────────────────────────────────────────

KW_MILITARY = re.compile(
    r"(?iu)"
    r"военн\w{0,12}\s+билет\w{0,8}"
    r"|военн\w{0,12}\s+документ\w{0,8}"
    r"|билет\w{0,8}\s+военн\w{0,12}"
    r"|военник\w{0,8}"
    r"|воинск\w{0,8}\s+документ\w{0,8}"
    r"|документ\w{0,8}\s+воинск\w{0,8}\s+уч[её]т\w{0,8}"
    r"|призывн\w{0,8}\s+удостоверен\w{0,12}"
    r"|солдатск\w{0,8}\s+книжк\w{0,8}"
    r"|воинск\w{0,8}\s+книжк\w{0,8}"
    r"|удостоверен\w{0,12}\s+военнообязан\w{0,12}"
    r"|воинск\w{0,8}\s+билет\w{0,8}"
    r"|удостоверен\w{0,12}\s+военнослужащ\w{0,12}"
    r"|военн(?:о|-|\s)?учетн\w{0,8}\s+документ\w{0,8}"
)


# Budgets measured over the 278 MILITARY_ID spans of the quality-gate benchmark: keyword to the left
# at p50 = 16, p90 = 48, p99 = 112, furthest 119; only four spans have it solely
# to the right, the furthest at 35. The two tiers below keep the existing
# confidence split -- close keyword scores higher than distant one -- but both are
# now bounded, where before the outer tier accepted the keyword anywhere in a
# 260-character window.
MAX_DIST_NEAR = 60
MAX_DIST_WIDE = 140
MAX_DIST_AFTER = 40


def _keyword_distances(
    raw: str, ctx: str, wide_ctx: str | None, ctx_pos: int, wide_ctx_pos: int
) -> tuple[bool, bool]:
    """``(near, wide)`` -- whether a military keyword is close, or merely in range."""
    near = nearest_before_distance(KW_MILITARY, ctx, ctx_pos, max_dist=MAX_DIST_NEAR)
    if near is not None:
        return True, True
    wide = nearest_before_distance(KW_MILITARY, ctx, ctx_pos, max_dist=MAX_DIST_WIDE)
    if wide is None and wide_ctx is not None:
        wide = nearest_before_distance(
            KW_MILITARY, wide_ctx, wide_ctx_pos, max_dist=MAX_DIST_WIDE
        )
    if wide is None:
        # Trailing form: «АО-1657262, военный билет выдан …». Measured from the end
        # of the candidate, and the slice must fit a whole keyword.
        tail_start = ctx_pos + len(raw)
        if KW_MILITARY.search(ctx[tail_start: tail_start + MAX_DIST_AFTER]):
            wide = 0
    return False, wide is not None


# ── Classifier ──────────────────────────────────────────────────


@register_entity
class MilitaryIdClassifier(BaseEntityClassifier):
    """Two-letter Cyrillic / Latin series + 7 digits.

    Examples::

        НЛ №1234567
        АО-1657262
        серия PT номер 9876543
    """

    entity_type = ENTITY_TYPE
    priority = 20
    candidate_patterns = [MILITARY_SERIES_NUMBER_RE, MILITARY_ID_RE]
    conflict_wins_over = {"PHONE_NUMBER"}

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        stripped = raw.strip()
        if not (
            MILITARY_ID_RE.fullmatch(stripped)
            or MILITARY_SERIES_NUMBER_RE.fullmatch(stripped)
        ):
            return None

        strict_m = MILITARY_STRICT_RE.fullmatch(stripped)
        if strict_m:
            series_norm = re.sub(
                r"[^A-Za-zА-Яа-яЁё]", "", strict_m.group("series"),
            ).upper()
            if series_norm in MILITARY_AMBIG_SERIES:
                near, wide_near = _keyword_distances(raw, ctx, wide_ctx, ctx_pos, wide_ctx_pos)
                if near:
                    return self.entity_type, 0.90
                if wide_near:
                    return self.entity_type, 0.85
                return None
            return self.entity_type, 0.95

        near, wide_near = _keyword_distances(raw, ctx, wide_ctx, ctx_pos, wide_ctx_pos)
        if near:
            return self.entity_type, 0.85
        if wide_near:
            return self.entity_type, 0.80
        return None
