"""BIK (БИК — Russian bank identifier code) classifier.

BIK as a standalone entity: a 9-digit candidate + "БИК" context. The
keyword regex and structural validity live in the shared
:mod:`._bik_common` (single source of truth; the same module is used by
:mod:`.bank_account` when verifying the account checksum).

Recognised only when the explicit word "БИК"/"BIC" is nearby — otherwise
bare 9 digits are too ambiguous (without it NER tags them as PHONE). See
the :mod:`._bik_common` docstring for BIK structure details.
"""

from __future__ import annotations

import re

from pii_guard.entities._bik_common import KW_BIK, valid_bik
from pii_guard.framework.base import BaseEntityClassifier, register_entity
from pii_guard.framework.context import nearest_before_distance

ENTITY_TYPE = "BIK"

# Measured over the 163 BIK spans of the quality-gate benchmark: keyword always to the left, p50 = 1,
# p95 = 12, furthest = 24. 40 covers the whole measured distribution.
MAX_DIST_BIK = 40


@register_entity
class BikClassifier(BaseEntityClassifier):
    """BIK — 9 digits + explicit "БИК" context.

    Scoring: "БИК" context present and structure valid (``04…`` +
    member-bank number ``050–999``) → ``0.95``; "БИК" context present but
    structure off (synthetic/typo) → ``0.85``; no context → ``None``.

    ``conflict_wins_over = {"PHONE_NUMBER"}``: NER often mistakes a
    9-digit BIK for a phone — on overlap, BIK wins.
    """

    entity_type = ENTITY_TYPE
    priority = 50
    candidate_patterns = [
        # 9 digits; separators between groups are allowed (like account/card):
        # space/tab/`/`/`_`/`\`/`-`. Lookbehind `(?<![\d-])` + trailing
        # lookahead prevent grabbing digits from an adjacent number.
        re.compile(r"(?<![\d-])\d(?:[ \t/_\\-]{0,3}\d){8}(?![ \t/_\\-]*\d)"),
    ]
    conflict_wins_over = {"PHONE_NUMBER"}

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        if len(digits) != 9:
            return None

        # Proximity only. The two window-wide fallbacks that used to follow made
        # the distance check above dead code and let any nine-digit number within
        # 260 characters of the word «БИК» be claimed -- and an invalid checksum
        # still scores 0.85, so a wrong claim is not even downgraded much.
        # Measured on the 163 BIK spans of the quality-gate benchmark: every one has the keyword to
        # its left, the furthest at 24 characters. A 40-character budget covers
        # the measured distribution completely, so nothing is traded away here.
        if nearest_before_distance(KW_BIK, ctx, ctx_pos, max_dist=MAX_DIST_BIK) is None:
            return None

        return self.entity_type, (0.95 if valid_bik(digits) else 0.85)
