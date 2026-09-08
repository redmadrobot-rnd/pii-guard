"""Birth certificate (свидетельство о рождении) classifier.

Self-contained entity definition: patterns, classification — all in one file.
"""

from __future__ import annotations

import re

from pii_guard.framework.base import BaseEntityClassifier, register_entity

ENTITY_TYPE = "BIRTH_CERTIFICATE"

# ── Candidate extraction patterns ───────────────────────────────

_LAT_RU_UNIT = r"(?:SHCH|ZH|KH|TS|CH|SH|YU|YA|YE|YO|JO|IO|EH|[A-Z])"
_BIRTH_LAT_SERIES = rf"(?:{_LAT_RU_UNIT})(?:[\s\-]?(?:{_LAT_RU_UNIT})){{1,5}}"
_BIRTH_SERIES = (
    rf"[IVXLCDM]{{1,4}}\s*[-\u2013\u2014]?\s*"
    rf"(?:[А-ЯЁ]{{2}}|[A-ZА-ЯЁ]{{2,6}}|{_BIRTH_LAT_SERIES})"
)
_BIRTH_NUMBER = r"(?:\d[ \t\n/._\\-]?){5}\d"


BIRTH_CERT_STRICT_RE = re.compile(
    rf"(?iu)^(?:№\s*)?(?:{_BIRTH_SERIES})"
    rf"[\s,;:\.\-]*"
    rf"(?:(?:номер|№|N)\s*)?"
    rf"(?:{_BIRTH_NUMBER})$"
)

# Unanchored series+number only — used for candidate extraction (no keyword prefix).
#
# `(?<!\.)` keeps the match out of hostnames. `C` is a Roman numeral and `om` a
# two-letter series, so «https://example.com, 630009» parsed as a certificate and
# swallowed the URL and the number after it. A real series never follows a dot
# without a space; none of the 126 candidates in the benchmark does.
BIRTH_CERT_NUMBER_RE = re.compile(
    rf"(?iu)(?<!\.)\b(?:№\s*)?(?:{_BIRTH_SERIES})"
    rf"[\s,;:\.\-]*"
    rf"(?:(?:номер|№|N)\s*)?"
    rf"(?:{_BIRTH_NUMBER})\b"
)

KW_BIRTH = re.compile(
    r"(?iu)"
    r"свидетел\w{0,20}\s+о\s+рожд\w{0,12}"
    r"|св[-\s]*во\s+о\s+рожд\w{0,12}"
    r"|удостоверен\w{0,20}\s+о\s+рожд\w{0,12}"
    r"|метрик\w{0,8}"
    r"|загс\w{0,8}"
    r"|запис\w{0,12}\s+акт\w{0,8}\s+о\s+рожд\w{0,12}"
)


# ── Classifier ──────────────────────────────────────────────────


@register_entity
class BirthCertificateClassifier(BaseEntityClassifier):
    """Roman-numeral series + Cyrillic letters + 6-digit number.

    Examples::

        IV-ЖА 123456
        VII-SHSH 333238
        свидетельство о рождении серия II-МЮ №950132
    """

    entity_type = ENTITY_TYPE
    priority = 10  # pattern-based → checked first
    candidate_patterns = [BIRTH_CERT_NUMBER_RE]
    conflict_wins_over = {"PHONE_NUMBER"}

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        from pii_guard.framework.context import has_keyword
        stripped = raw.strip()
        if not BIRTH_CERT_NUMBER_RE.search(stripped):
            return None
        has_kw = has_keyword(KW_BIRTH, ctx) or (
            wide_ctx is not None and has_keyword(KW_BIRTH, wide_ctx)
        )
        if BIRTH_CERT_STRICT_RE.fullmatch(stripped):
            return self.entity_type, 0.95 if has_kw else 0.85
        return self.entity_type, 0.90 if has_kw else 0.80
