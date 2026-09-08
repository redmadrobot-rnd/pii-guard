"""BANK_ACCOUNT (settlement / correspondent account) classifier.

Checksum validator, keyword context, classification. Looking up the BIK
value lives in the shared :mod:`._bik_common` (single source of truth;
the same module is used by :mod:`.bik`) so the BIK regex is not
duplicated.

A Russian account is exactly 20 digits. The check digit is computed
**together with the bank's BIK** (9 digits): the account alone cannot be
verified. Without a BIK, only the syntactic check (20 digits) + "счёт"
context remains.
"""

from __future__ import annotations

import re

from pii_guard.entities._bik_common import find_bik
from pii_guard.framework.base import BaseEntityClassifier, register_entity
from pii_guard.framework.context import nearest_before_distance

ENTITY_TYPE = "BANK_ACCOUNT"


# ── Checksum validator (self-contained) ─────────────────────────

WEIGHTS = [7, 1, 3] * 7 + [7, 1]  # 23 coefficients


def valid_account(account: str, bik: str) -> bool:
    """Settlement account (20 digits) + bank BIK (9 digits), weighted sum mod 10.

    For a correspondent account (``301…``) the prefix is ``"0" + BIK[:2]``,
    otherwise the last 3 digits of the BIK.
    """
    if not (account.isdigit() and len(account) == 20):
        return False
    if not (bik.isdigit() and len(bik) == 9):
        return False
    prefix = ("0" + bik[:2]) if account.startswith("301") else bik[-3:]
    return sum(int(d) * w for d, w in zip(prefix + account, WEIGHTS, strict=False)) % 10 == 0


# ── Keyword patterns ────────────────────────────────────────────

KW_ACCOUNT = re.compile(
    r"(?iu)расч[ёе]тн\w{0,6}\s+сч[ёе]т\w{0,6}|\bсч[ёе]т\w{0,6}|\bсч\.|\bр/?сч?\b"
    r"|\bк/?сч?\b|корсч\w{0,6}|реквизит\w{0,6}|перечисл\w{0,8}"
    r"|плат[ёе]жн\w{0,6}\s+поручени\w{0,6}"
)


@register_entity
class BankAccountClassifier(BaseEntityClassifier):
    """Settlement/correspondent account — 20 digits + "счёт" context.

    Two-branch strategy:

    * **BIK nearby** → verify the checksum: valid → ``0.97``; invalid but
      "счёт" context present → ``0.60`` (synthetic/typo); otherwise →
      ``None``.
    * **no BIK** → syntax (20 digits) + "счёт" context → ``0.90``.
    """

    entity_type = ENTITY_TYPE
    priority = 45
    candidate_patterns = [
        re.compile(r"(?<![\d-])\d(?:[ \t/_\\-]{0,3}\d){19}(?![ \t/_\\-]*\d)"),
    ]

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        if len(digits) != 20:
            return None

        # Proximity only. Window-wide `has_keyword` fallbacks used to make this
        # distance check dead code, leaving the same "keyword anywhere in 260
        # characters" defect bounded in the sibling classifiers. Measured over 273
        # benchmark spans: keyword always to the left, p95 = 24, furthest 41. The
        # wide budget is 120, not 240, which on a 260-character window is
        # "anywhere" again.
        acct = nearest_before_distance(KW_ACCOUNT, ctx, ctx_pos, max_dist=120)
        if acct is None and wide_ctx:
            acct = nearest_before_distance(KW_ACCOUNT, wide_ctx, wide_ctx_pos, max_dist=120)
        has_acct = acct is not None

        bik = find_bik(ctx, wide_ctx)
        if bik:  # BIK present → verify the checksum
            if valid_account(digits, bik):
                return self.entity_type, 0.97
            return (self.entity_type, 0.60) if has_acct else None

        if has_acct:  # no BIK — syntax + context
            return self.entity_type, 0.90
        return None
