"""Shared conflict-resolution helpers for entity classifiers."""

from __future__ import annotations

import re

from pii_guard.framework.context import nearest_before_distance, nearest_distance

# ── Shared keyword patterns ─────────────────────────────────────

KW_PASSPORT = re.compile(
    r"(?iu)"
    r"паспорт\w{0,8}"
    r"|загранпаспорт\w{0,8}"
    r"|паспортн\w{0,8}\s+данн\w{0,8}"
    r"|документ\w{0,12}\s+гражданин\w{0,12}"
    r"|удостоверяющ\w{0,12}\s+личност\w{0,12}"
    r"|удостоверен\w{0,12}\s+личност\w{0,12}"
    r"|документ\w{0,12}\s*,?\s*(?:являющ\w{0,12}\s+)?удостоверяющ\w{0,12}\s+личност\w{0,12}"
    r"|идентификатор\w{0,12}\s+личност\w{0,12}"
)

KW_DL = re.compile(
    r"(?iu)"
    r"водит\w{0,12}"
    r"|вод\."
    r"|удостоверен\w{0,12}"
    r"|вождени\w{0,8}"
    r"|\bправа\b"
    r"|\bгибдд\w{0,8}\b"
    r"|(?:замен\w{0,12}|выдач\w{0,12}|получени\w{0,12}|лишен\w{0,12})\s+прав\w{0,8}"
    r"|\bс\s+прав\w{0,8}\b"
    r"|номер\w{0,8}\s+прав\w{0,8}"
    r"|прав\w{0,8}\s+на\s+управлен\w{0,12}"
    r"|транспортн\w{0,12}\s+средств\w{0,12}"
    r"|[Вв][/\.][Уу]"
    r"|\b[Вв][Уу]\b"
    r"|водительск\w{0,8}\s+прав\w{0,8}"
)


# ── Proximity gate ──────────────────────────────────────────────
# Measured over the quality-gate benchmark, not guessed: keyword to the left,
# passport p99 = 83, licence p99 = 96; only to the right, 191 and 108. Coverage
# saturates well inside these -- the 16.7% of passports with no keyword either
# side are found by the model, not by this rule. Both patterns share the budgets
# on purpose: `resolve_passport_dl_conflict` compares their distances directly,
# so asymmetry would make the verdict depend on which side asked.
MAX_DIST_DOC_CTX = 80
MAX_DIST_DOC_WIDE = 160
MAX_DIST_DOC_AFTER = 200


def document_keyword_nearby(
    pattern: re.Pattern[str],
    raw: str,
    ctx: str,
    wide_ctx: str | None,
    ctx_pos: int,
    wide_ctx_pos: int,
) -> bool:
    """Is *pattern* close enough to the candidate to count as evidence?

    Replaces a plain ``has_keyword`` over the context window. Presence anywhere
    in a 260-character window is not evidence about ten particular digits: it let
    a section heading or an unrelated following sentence decide the type. See the
    comment in ``entities/postal_code.py`` for the same defect with observed
    consequences.
    """
    if nearest_before_distance(pattern, ctx, ctx_pos, max_dist=MAX_DIST_DOC_CTX) is not None:
        return True
    if wide_ctx is not None and nearest_before_distance(
        pattern, wide_ctx, wide_ctx_pos, max_dist=MAX_DIST_DOC_WIDE
    ) is not None:
        return True
    # Trailing form: «50 12 884920, паспорт выдан …». Bounded, and the bound must
    # fit whole keywords -- a window cut mid-word matches nothing.
    tail_start = ctx_pos + len(raw)
    return bool(pattern.search(ctx[tail_start: tail_start + MAX_DIST_DOC_AFTER]))


# ── Resolver ────────────────────────────────────────────────────

def resolve_passport_dl_conflict(
    raw: str,
    ctx: str,
    wide_ctx: str | None,
    ctx_pos: int,
    wide_ctx_pos: int,
) -> str | None:
    """Return ``'passport'`` or ``'driver_license'`` if resolvable, else ``None``."""
    p_before = nearest_before_distance(KW_PASSPORT, ctx, ctx_pos, max_dist=MAX_DIST_DOC_CTX)
    d_before = nearest_before_distance(KW_DL, ctx, ctx_pos, max_dist=MAX_DIST_DOC_CTX)
    if wide_ctx:
        if p_before is None:
            p_before = nearest_before_distance(KW_PASSPORT, wide_ctx, wide_ctx_pos, max_dist=MAX_DIST_DOC_WIDE)
        if d_before is None:
            d_before = nearest_before_distance(KW_DL, wide_ctx, wide_ctx_pos, max_dist=MAX_DIST_DOC_WIDE)

    if p_before is not None and d_before is None:
        return "passport"
    if d_before is not None and p_before is None:
        return "driver_license"
    if p_before is not None and d_before is not None:
        if p_before < d_before:
            return "passport"
        if d_before < p_before:
            return "driver_license"

    center = ctx_pos + len(raw) // 2
    p_dist = nearest_distance(KW_PASSPORT, ctx, center)
    d_dist = nearest_distance(KW_DL, ctx, center)
    if p_dist is not None and d_dist is not None:
        if p_dist < d_dist:
            return "passport"
        if d_dist < p_dist:
            return "driver_license"

    if wide_ctx:
        wide_center = wide_ctx_pos + len(raw) // 2
        p_dist_w = nearest_distance(KW_PASSPORT, wide_ctx, wide_center)
        d_dist_w = nearest_distance(KW_DL, wide_ctx, wide_center)
        if p_dist_w is not None and d_dist_w is not None:
            if p_dist_w < d_dist_w:
                return "passport"
            if d_dist_w < p_dist_w:
                return "driver_license"
    return None
