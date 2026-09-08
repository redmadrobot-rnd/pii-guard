"""Shared bank-card keywords — single source of truth for the entity files.

A ``_``-prefixed module: auto-discovery does NOT register it as an entity. It
exists for the same reason as :mod:`._bik_common` — two entity files consult the
same pattern and must not be allowed to drift apart.

:mod:`.credit_card` and :mod:`.oms` both classify a 16-digit candidate by
comparing the *distance* to a card keyword against the distance to an OMS
keyword: whichever is nearer claims the number. That comparison is only sound if
both sides read the same list. They had already diverged — the card copy grew
four English alternatives (``card``, ``credit card``, ``debit card``,
``card number``) that the OMS copy never got, so «card number 4276...» looked
like a card from one side and like nothing at all from the other, and an English
context silently fell through to OMS.
"""

from __future__ import annotations

import re

# Unambiguous card evidence: a brand, a card-specific field name, or the word
# «карта» with a qualifier that rules out a loyalty or medical card.
KW_CARD = re.compile(
    r"(?iu)"
    r"\bvisa\b"
    r"|\bmastercard\b"
    r"|\bмир\b"
    r"|\bpan\b"
    r"|\bcvv\b"
    r"|\bбанковск\w{0,12}\s+карт\w{0,12}\b"
    r"|\bкредитн\w{0,12}\s+карт\w{0,12}\b"
    r"|\bдебетов\w{0,12}\s+карт\w{0,12}\b"
    r"|\bномер\w{0,8}\s+карт\w{0,12}\b"
    # `\bcard\b` матчит и «credit card», и «card number»: паттерн используется
    # только как булев признак, поэтому отдельные альтернативы были недостижимы.
    r"|\bcard\b"
)

__all__ = ["KW_CARD"]
