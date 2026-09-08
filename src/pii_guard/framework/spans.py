"""Narrowing a candidate span down to the data it carries.

Document candidates are matched by their phrase, because the keyword is what
makes the digits recognisable: ``серия 7518, номер 492137`` is a passport,
``7518`` alone is not. The span that comes out, though, should point at the data
and nothing else -- «серия» and «номер» are not personal data, the entity type is
already reported next to the span, and masking them costs the sentence more than
it protects:

    паспорт ************************      <- candidate span as matched
    паспорт серия ****, номер ******      <- data spans

A composite number also stops being one span here. Series and number are two
values, they are annotated as two in every dataset we score against, and a
pseudonym for each keeps the sentence readable.
"""

from __future__ import annotations

import re

# Words that introduce a document number without being part of it. Deliberately
# short: only what the candidate patterns in `patterns.py` and `entities/*.py`
# can actually capture, so the list cannot silently start eating data.
SERVICE_WORD = re.compile(r"(?iu)^(?:сер(?:и[ияей])?|номер\w*|ном|№|N|No)\.?$")

# Punctuation that may sit around a token without belonging to the value. The
# hyphen is missing on purpose -- it lives inside series («II - АВ», «697-843»).
_TRIM_LEFT = "№ \t,.;:()[]«»\"'"
_TRIM_RIGHT = " \t,.;:()[]«»\"'"

# What may separate two chunks of one value: whitespace and dashes only. A comma
# is not here -- `4510 123456` is one number, `7518, номер 492137` is two.
_JOINABLE = re.compile(r"^[\s\-–—/]*$")


def data_spans(raw: str) -> list[tuple[int, int]]:
    """Offsets of the data inside *raw*, relative to *raw*.

    Returns one span per value: service words are dropped, and chunks separated
    by a service word become separate spans. Falls back to the whole string when
    nothing looks like data, so a caller can always emit something.
    """
    chunks: list[tuple[int, int]] = []
    broken = False  # a service word was seen since the previous data chunk

    for token in re.finditer(r"\S+", raw):
        start, end = token.start(), token.end()
        while start < end and raw[start] in _TRIM_LEFT:
            start += 1
        while end > start and raw[end - 1] in _TRIM_RIGHT:
            end -= 1
        value = raw[start:end]

        if not value or SERVICE_WORD.match(value):
            broken = True
            continue

        joinable = (
            chunks
            and not broken
            and _JOINABLE.match(raw[chunks[-1][1]: start]) is not None
        )
        if joinable:
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
        broken = False

    return chunks or [(0, len(raw))]
