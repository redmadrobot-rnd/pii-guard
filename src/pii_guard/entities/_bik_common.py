"""Shared BIK logic — single source of truth for the entity files.

A ``_``-prefixed module: auto-discovery does NOT register it as an
entity. It holds the BIK regexes and structural validity so they are not
duplicated (and not allowed to drift) between :mod:`bik` (BIK as a
standalone entity) and :mod:`bank_account` (BIK as the key for the
account checksum).

Structure of a Russian BIK (9 digits):

* digits 1–2 — country code, always ``04`` for Russia;
* digits 3–4 — OKATO region code (``01–99``; ``00`` for clearing / the
  Central Bank);
* digits 5–6 — internal unit number within the Bank of Russia settlement
  network (``00–99``);
* digits 7–9 — three-digit member-bank number, range ``050–999``.

Hard validity constraints: prefix ``04`` and member-bank number
``050–999``; the region and unit fields allow any ``00–99``.
"""

from __future__ import annotations

import re

# Extract the BIK VALUE: keyword + up to 6 non-digits + 9 digits, groups allowed.
# Used where the number itself is needed (account checksum). Must accept exactly
# what `KW_BIK` accepts: it took neither `BIC` nor `044 525 225` while the gate
# took both, so `bank_account` silently skipped its checksum branch for those.
BIK_VALUE_RE = re.compile(r"(?iu)\b(?:бик|bic)\D{0,6}((?:\d[ \t]?){8}\d)\b")

# Keyword "БИК"/"BIC" for the context gate (no digit capture). Used where
# a 9-digit candidate has already been found and only context is checked.
KW_BIK = re.compile(r"(?iu)\bбик\w{0,3}|\bbic\b")


def find_bik(*texts: str | None) -> str | None:
    """Extract the BIK value (9 digits) from the first matching text.

    Searches for the "БИК … 9 digits" pattern in the given strings in
    order (usually ``ctx`` and ``wide_ctx``) and returns the captured
    9 digits.
    """
    for text in texts:
        if text:
            match = BIK_VALUE_RE.search(text)
            if match:
                # Группировка допускается в паттерне, наружу отдаются только
                # цифры: значение уходит прямо в контрольную сумму счёта.
                return re.sub(r"\D", "", match.group(1))
    return None


def valid_bik(bik: str) -> bool:
    """Structural validity of a Russian BIK (9 digits).

    Checks two hard constraints: country code (digits 1–2 == ``04``) and
    member-bank number (digits 7–9 in the ``050–999`` range). The region
    (3–4) and unit (5–6) fields allow any ``00–99``.
    """
    if not (bik.isdigit() and len(bik) == 9):
        return False
    if bik[:2] != "04":
        return False
    return 50 <= int(bik[6:9]) <= 999
