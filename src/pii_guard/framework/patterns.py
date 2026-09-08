"""
Candidate extraction regex patterns for document PII detection.

Generic (shared) patterns live here.  Entity-specific patterns live in
their respective ``entities/*.py`` files and are collected automatically
by the registry via ``BaseEntityClassifier.candidate_patterns``.

The final ordered pattern list is built at runtime by
:func:`build_candidate_patterns`.
"""

from __future__ import annotations

import re

# ── Shared keyword patterns ──────────────────────────────────────

KW_OMS = re.compile(
    r"(?iu)"
    r"полис\w{0,8}\s+омс"
    r"|\bомс\b"
    r"|медстраховк\w{0,8}"
    r"|страхов\w{0,12}\s+полис\w{0,8}"
    r"|свидетельств\w{0,12}\s+омс"
    r"|полис\w{0,8}\s+медицинск\w{0,16}\s+страхован\w{0,16}"
    r"|полис\w{0,8}\s+обязательн\w{0,16}\s+медицинск\w{0,16}\s+страхован\w{0,16}"
    r"|полис\w{0,8}\s+медицинск\w{0,16}\s+обязательн\w{0,16}\s+страхован\w{0,16}"
    r"|обязательн\w{0,16}\s+медицинск\w{0,16}\s+страхован\w{0,16}"
    r"|медполис\w{0,8}"
    r"|полис\w{0,8}\s+обязательн\w{0,16}\s+страхован\w{0,16}"
)

# ── Universal numeric candidates ────────────────────────────────

NUM_CANDIDATE_RE = re.compile(
    r"(?<![\d-])(?<!\d )(?:\d(?:[ \t/_\\-]{0,3}\d){9,18})(?![ \t/_\\-]*\d)",
    re.IGNORECASE,
)

NUM_CANDIDATE_WITH_SIGN_RE = re.compile(
    r"(?iu)№\s*(?:\d(?:[ \t/_\\-]{0,3}\d){9,18})(?![ \t/_\\-]*\d)"
)

SERIES_NUMBER_NUMERIC_RE = re.compile(
    r"(?iu)\b"
    r"сер(?:ия)?\s*(?:№\s*)?(?:\d(?:[ \t\n/._\\-]?\d){1,7})"
    r"[\s,;:\.\-]*"
    r"(?:номер|№|N)\s*(?:\d[ \t\n/._\\-]?){5,15}\d"
    r"\b"
)

# ── Generic candidate patterns (always applied AFTER entity-specific ones) ──

GENERIC_CANDIDATE_PATTERNS: tuple[re.Pattern, ...] = (
    SERIES_NUMBER_NUMERIC_RE,
    NUM_CANDIDATE_WITH_SIGN_RE,
    NUM_CANDIDATE_RE,
)


def build_candidate_patterns(
    registry,  # EntityClassifierRegistry (import-free to avoid cycles)
) -> tuple[re.Pattern, ...]:
    """Merge entity-specific patterns (from *registry*) + generic patterns.

    Entity-specific patterns come first (in classifier priority order),
    then the generic catch-all patterns.
    """
    entity_patterns = registry.all_candidate_patterns()
    return tuple(entity_patterns) + GENERIC_CANDIDATE_PATTERNS
