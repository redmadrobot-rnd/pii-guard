"""POSTAL_CODE (Russian postal index) classifier.

Self-contained entity definition: keyword context, classification —
all in one file.

A Russian Post index is exactly 6 digits. Validation is two-level:
syntax (6 digits) + address context. The first 3 digits encode the
region / sorting node. The actually existing prefixes are 101–699, with
the 500–599 range entirely absent. So valid prefixes are 101–499 and
600–699; anything else is not a postal index.
"""

from __future__ import annotations

import re

from pii_guard.framework.base import BaseEntityClassifier, register_entity
from pii_guard.framework.context import has_keyword, nearest_before_distance

ENTITY_TYPE = "POSTAL_CODE"


# ── Keyword patterns ────────────────────────────────────────────

KW_POSTAL = re.compile(
    r"(?iu)\bиндекс\w{0,6}|\bинде?к\b|почтов\w{0,6}|\bпочт\w{0,5}|\bадрес\w{0,6}"
    r"|\bг\.|\bгород\w{0,6}|\bул\.|\bд\.|\bкрай\b|\bобл\w{0,6}|росси\w{0,3}"
    r"|проспект|шоссе"
)
# «биржевой индекс» qualifies the keyword itself rather than competing with it,
# so distance cannot separate the two: the qualifier always stands *before* the
# word it disqualifies and is therefore always further from the digits. Hence an
# unconditional veto, which is safe -- a stock index never shares a sentence
# with a postal address.
KW_POSTAL_VETO = re.compile(r"(?iu)биржев\w{0,6}")

# Competing readings of the same six digits: passport/series, order number,
# support ticket, driver licence. Unlike the veto above these legitimately share
# a sentence with an address («паспорт выдан ... прописан по адресу ...»), so
# they compete by distance instead of suppressing outright.
KW_POSTAL_NEG = re.compile(
    r"(?iu)паспорт\w{0,6}|\bсери\w{0,3}\b|заказ\w{0,6}|обращени\w{0,6}"
    r"|водительск\w{0,3}|удостоверени\w{0,2}|\bву\b"
)

# Keyword evidence must sit next to the digits, not anywhere in the window: a
# window-wide `has_keyword` reached across paragraphs, so a heading 190 characters
# back suppressed a real index and «по адресу» in the next sentence invented one
# out of a driver-licence number.
MAX_DIST_BEFORE = 60
# Tighter on the right, but not by much: Russian addresses put the index first
# («630009, г. Новосибирск»), and the keyword that confirms it can be a street
# type two components later -- «125167, Алатырь, Ленинградский проспект» needs 33
# characters, «115487, Регион 77, Чапаевск, Дмитровское шоссе» needs 40. Must fit
# the whole keyword, not its prefix: a window cut mid-word matches nothing.
MAX_DIST_AFTER = 45


@register_entity
class PostalCodeClassifier(BaseEntityClassifier):
    """Russian postal index — 6 digits + address context, minus document/order/exchange.

    The prefix must be real: ``101–499`` or ``600–699`` → ``0.92``.
    An impossible prefix (``500–599``, ``<101``, ``>699``) → ``None``:
    no such index exists, so it is not a postal code.
    """

    entity_type = ENTITY_TYPE
    priority = 60
    candidate_patterns = [
        # 6 digits; separators between groups are allowed (like account/card):
        # space/tab/`/`/`_`/`\`/`-` ('411-786', '108_095', '6 8 4 5 1 0').
        # The lookbehinds forbid starting mid-number: '10/10/2018' would yield
        # the sub-slice '10/2018' (a valid prefix). Space and tab need the second
        # one because they are both an in-number separator and the ordinary
        # boundary before one, so only "digit then space" may be forbidden --
        # without it 'адрес 9 1 2 3 0 0 7' matched from the second digit. Same
        # idiom as `NUM_CANDIDATE_RE`. The lookahead stops the match from
        # grabbing digits of an adjacent number.
        re.compile(
            r"(?<![\d/_\\-])(?<!\d[ \t])\d(?:[ \t/_\\-]{0,3}\d){5}(?![ \t/_\\-]*\d)"
        ),
    ]

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        if len(digits) != 6:
            return None

        if has_keyword(KW_POSTAL_VETO, ctx) or (
            wide_ctx is not None and has_keyword(KW_POSTAL_VETO, wide_ctx)
        ):
            return None

        # Nearest keyword wins. Both an address word and a document word can be
        # in range at once -- «водительское удостоверение 77 АВ 123456, индекс
        # 630009» needs the licence number rejected and the index accepted from
        # the same sentence, which a boolean check in either direction cannot do.
        pos_before = nearest_before_distance(
            KW_POSTAL, ctx, ctx_pos, max_dist=MAX_DIST_BEFORE
        )
        neg_before = nearest_before_distance(
            KW_POSTAL_NEG, ctx, ctx_pos, max_dist=MAX_DIST_BEFORE
        )
        if neg_before is not None and (pos_before is None or neg_before < pos_before):
            return None

        if pos_before is None:
            tail = ctx[ctx_pos + len(raw): ctx_pos + len(raw) + MAX_DIST_AFTER]
            if not has_keyword(KW_POSTAL, tail):
                return None

        # Real Russian indices: prefix 101–699, but the 500–599 range is
        # entirely absent. Anything outside 101–499 / 600–699 does not
        # exist as an index → do not tag it.
        pref = int(digits[:3])
        if not (101 <= pref <= 499 or 600 <= pref <= 699):
            return None
        return self.entity_type, 0.92
