"""DATE_TIME entity — Russian dates, time, ISO formats, year/hours.

Self-contained: all patterns and Presidio ``PatternRecognizer``
registration in one file.
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

from pii_guard.framework.base import (
    register_conflict_handler,
    register_pairwise_conflict,
    register_regex_entity,
)


@register_regex_entity
def register_date_time(analyzer: AnalyzerEngine, language: str = "ru") -> None:
    """Create and register all DATE_TIME recognizers on *analyzer*."""

    # Russian dates with month names
    ru_date_pattern = Pattern(
        name="dd mm yy (года/год)",
        regex=(
            r"(?<!\d)(([1-9]|0[1-9]|[12][0-9]|3[01])\s+"
            r"(январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|"
            r"июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|"
            r"ноябр[ья]|декабр[ья])"
            r"(?:(?:\s+\d{4})(?:(?:\s+(?:года|году|год))(?!\w)|(?!\d))|"
            r"(?:\s+(?:года|году|год))(?!\w)|(?!\d)))"
        ),
        score=0.85,
    )

    # DD.MM.YY / DD.MM.YYYY
    date_dot_pattern = Pattern(
        name="date_dot_format",
        regex=(
            r"(?<!\d)(?:0?[1-9]|[12][0-9]|3[01])[ \t]*\.[ \t]*"
            r"(?:0?[1-9]|1[0-2])[ \t]*\.[ \t]*(?:19|20)\d{2}(?!\d)"
            r"(?!(?:[ \t]*[-–—][ \t]*\d{1,2}(?![ \t]*[.\d]|$)))"
        ),
        score=0.8,
    )

    # DD-MM-YY / DD-MM-YYYY
    date_dash_pattern = Pattern(
        name="date_dash_format",
        regex=(
            r"(?<!\d)(?:0?[1-9]|[12][0-9]|3[01])[ \t]*[-–—][ \t]*"
            r"(?:0?[1-9]|1[0-2])[ \t]*[-–—][ \t]*(?:19|20)\d{2}(?!\d)"
        ),
        score=0.8,
    )

    # HH:MM or HH:MM:SS
    time_pattern = Pattern(
        name="time_detection",
        regex=(
            r"(?<!\d)(?:[0-1]?[0-9]|2[0-3])[ \t]*:[ \t]*[0-5][0-9]"
            r"(?:[ \t]*:[ \t]*[0-5][0-9])?(?!\d)"
        ),
        score=0.75,
    )

    # MM/YYYY or MM/YY
    date_slash_pattern = Pattern(
        name="date_slash_format",
        regex=r"(?<!\d)(?:0?[1-9]|1[0-2])/(?:\d{2}|\d{4})(?!\d)",
        score=0.8,
    )

    # ISO 8601
    iso_date_pattern = Pattern(
        name="iso_date_format",
        regex=(
            r"(?<!\d)\d{4}-\d{2}-\d{2}"
            r"(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?)?(?!\d)"
        ),
        score=0.85,
    )

    # "YYYY года" / "N часов/часа"
    year_pattern = Pattern(
        name="year_pattern",
        regex=r"(?<!\d)\d{4}\s+года(?!\d)",
        score=0.8,
    )
    hours_pattern = Pattern(
        name="hours_pattern",
        regex=r"(?<!\d)\d{1,2}\s+час(?:ов|а|у|ами|ам)?\b(?!\d)",
        score=0.75,
    )

    # DD month. YYYY
    ru_abbrev_month_pattern = Pattern(
        name="ru_abbrev_month",
        regex=(
            r"(?<!\d)(?:0?[1-9]|[12][0-9]|3[01])\s+"
            r"(?:янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек)\b\.?"
            r"(?:(?:\s+\d{4})(?!\d)|(?!\d))"
        ),
        score=0.85,
    )

    # With suffix: DD-го / -e month YYYY
    ru_ordinal_date_pattern = Pattern(
        name="ru_ordinal_date",
        regex=(
            r"(?<!\d)(?:0?[1-9]|[12][0-9]|3[01])-(?:го|е|ой|ему|ем)\s+"
            r"(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|"
            r"июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|"
            r"ноябр[ья]|декабр[ья])"
            r"(?:(?:\s+\d{4})(?!\d)|(?!\d))"
        ),
        score=0.85,
    )

    # DD/MM/YYYY or DD/MM/YY
    date_slash_full_pattern = Pattern(
        name="date_slash_full",
        regex=(
            r"(?<!\d)(?:0?[1-9]|[12][0-9]|3[01])[ \t]*/[ \t]*"
            r"(?:0?[1-9]|1[0-2])[ \t]*/[ \t]*(?:\d{2}|\d{4})(?!\d)"
        ),
        score=0.8,
    )

    # YYYY.MM.DD or YYYY/MM/DD
    date_year_first_pattern = Pattern(
        name="date_year_first",
        regex=(
            r"(?<!\d)(?:19\d{2}|20\d{2}|2100)(?:[./]|[ \t]*[-–—][ \t]*)"
            r"(?:0?[1-9]|1[0-2])(?:[./]|[ \t]*[-–—][ \t]*)(?:0?[1-9]|"
            # A trailing `.dd` or `/dd` means the match started mid-range rather
            # than at a date: «01.02.2026-05.02.2026» let the dash branch match
            # «2026-05.02», which scores 0.85 and beat both correct 0.80 day-first
            # spans, so the output was «01.02.[DATE_TIME].2026» -- one wrong span
            # and two real dates lost, with «01.02.» and «.2026» left in the clear
            # under mode="mask".
            r"[12][0-9]|3[01])(?!\d)(?![./]\d)"
        ),
        score=0.85,
    )

    # name + year (no day): январь 2025
    ru_month_year_pattern = Pattern(
        name="ru_month_year",
        regex=(
            r"(?<!\d)(?:январ[ья]?|феврал[ья]?|март[а]?|апрел[ья]?|ма[йя]|"
            r"июн[ья]?|июл[ья]?|август[а]?|сентябр[ья]?|октябр[ья]?|"
            r"ноябр[ья]?|декабр[ья]?)"
            r"\s+\d{4}(?!\d)"
        ),
        score=0.8,
    )

    # English dates: January 15th, 2024
    en_date_full_pattern = Pattern(
        name="en_date_full",
        regex=(
            r"(?<!\d)(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}(?!\d)"
        ),
        score=0.85,
    )

    # Year-first with comma: YYYY, DD month
    ru_year_first_comma_pattern = Pattern(
        name="ru_year_first_comma",
        regex=(
            r"(?<!\d)\d{4},\s+(?:0?[1-9]|[12][0-9]|3[01])\s+"
            r"(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|"
            r"июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|"
            r"ноябр[ья]|декабр[ья])(?!\d)"
        ),
        score=0.85,
    )

    # -- register DATE_TIME recognizers --
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="RussianDateRecognizer",
            patterns=[
                ru_date_pattern,
                ru_abbrev_month_pattern,
                ru_ordinal_date_pattern,
            ],
            supported_language=language,
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="DateDotFormatRecognizer",
            patterns=[date_dot_pattern, date_dash_pattern],
            supported_language=language,
            context=[
                "дата",
                "подписан",
                "родился",
                "родилась",
                "создан",
                "изменен",
            ],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="TimeRecognizer",
            patterns=[time_pattern],
            supported_language=language,
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="DateSlashFormatRecognizer",
            patterns=[date_slash_pattern, date_slash_full_pattern],
            supported_language=language,
            context=["действителен", "истекает", "до", "срок", "валиден"],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="IsoDateFormatRecognizer",
            patterns=[iso_date_pattern, date_year_first_pattern],
            supported_language=language,
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="YearHoursRecognizer",
            patterns=[year_pattern, hours_pattern],
            supported_language=language,
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="RussianDateExtendedRecognizer",
            patterns=[ru_month_year_pattern, ru_year_first_comma_pattern],
            supported_language=language,
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="EnglishDateRecognizer",
            patterns=[en_date_full_pattern],
            supported_language=language,
        )
    )


# ── Conflict rules ──────────────────────────────

# IP_PORT wins, not DATE_TIME. The pairwise pass has no notion of transitivity:
# with `IP_PORT > IP_ADDRESS` declared in ip_port.py, a `DATE_TIME > IP_PORT` edge
# closes a cycle (IP_PORT > IP_ADDRESS > DATE_TIME > IP_PORT), and the outcome
# then depends on the order the three candidates happen to arrive in -- in two of
# six orders they annihilate each other and the address goes out unmasked.
# `сервер 10.0.5.21:45 работает` is enough to hit it: `21:45` is a valid time.
# Keep the graph acyclic, and keep the direction that over-masks: an ip:port
# swallowing a time is harmless, a leaked address is not.
register_pairwise_conflict(winner="IP_PORT", loser="DATE_TIME")
register_pairwise_conflict(winner="IP_ADDRESS", loser="DATE_TIME")
register_pairwise_conflict(winner="URL", loser="DATE_TIME")


@register_conflict_handler
def handle_date_time_ip_context(entities, text, already_removed):
    """Remove DATE_TIME when it's part of an IPv4 address or adjacent to IP_ADDRESS."""
    import re as _re

    extra: set[int] = set()

    def _is_part_of_ip(start: int, end: int) -> bool:
        search_start = max(0, start - 20)
        search_end = min(len(text), end + 20)
        ctx = text[search_start:search_end]
        ipv4_re = (
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?!\d)"
        )
        for m in _re.finditer(ipv4_re, ctx):
            ip_abs_s = search_start + m.start()
            ip_abs_e = search_start + m.end()
            if (ip_abs_s <= start < ip_abs_e) or (ip_abs_s < end <= ip_abs_e):
                return True
            if start <= ip_abs_s and end >= ip_abs_e:
                return True
        return False

    for i, e1 in enumerate(entities):
        if i in already_removed or e1.entity_type != "DATE_TIME":
            continue

        # DATE_TIME span inside IPv4 pattern → remove
        if _is_part_of_ip(e1.start, e1.end):
            extra.add(i)
            continue

        # DATE_TIME adjacent to IP_ADDRESS (gap ≤ 5, empty or ".") → remove
        for j, e2 in enumerate(entities):
            if (
                j != i
                and j not in already_removed
                and e2.entity_type == "IP_ADDRESS"
                and (abs(e1.start - e2.end) <= 5 or abs(e1.end - e2.start) <= 5)
            ):
                gap = (
                    text[e2.end : e1.start]
                    if e1.start >= e2.end
                    else text[e1.end : e2.start]
                )
                if len(gap) == 0 or gap.strip() == "" or gap == ".":
                    extra.add(i)
                    break

    return extra

@register_conflict_handler
def handle_date_time_normative_codes(entities, text, already_removed):
    """Remove DATE_TIME matches that are parts of regulatory code numbers.

    Example false positive we want to suppress: ``ГОСТ 1.02.2020-2016``, where
    ``1.02.2020`` is a plausible date but the ``ГОСТ`` prefix and the ``-2016``
    suffix together say it is a standard number. Codes whose middle group is not
    a plausible date at all (``СП 2.1.3678-20``) never reach this handler -- the
    classifier itself declines them.

    Important: keep true date ranges such as ``1.02.2026-5.02.2026``.
    """
    import re as _re

    extra: set[int] = set()

    date_dot_re = _re.compile(
        r"^\d{1,2}[ \t]*\.[ \t]*\d{1,2}[ \t]*\.[ \t]*\d{4}$",
        _re.IGNORECASE | _re.UNICODE,
    )
    # Typical suffix for standards/codes: "-20", "-2016", etc.
    # Do not fire on date ranges like "-5.02.2026" or "-05.02.2026".
    normative_suffix_re = _re.compile(r"^[ \t]*[-–—][ \t]*\d{2,4}(?![.\d])")
    normative_prefix_re = _re.compile(
        r"(?iu)(?:^|[\s(,;:№])(?:сп|снип|санпин|гост(?:\s*р)?|ту|тр\s*тс)\s*$"
    )

    for i, ent in enumerate(entities):
        if i in already_removed or ent.entity_type != "DATE_TIME":
            continue

        span_text = text[ent.start : ent.end]
        if not date_dot_re.fullmatch(span_text):
            continue

        tail = text[ent.end : min(len(text), ent.end + 8)]
        if not normative_suffix_re.match(tail):
            continue

        left = text[max(0, ent.start - 24) : ent.start]
        if normative_prefix_re.search(left):
            extra.add(i)

    return extra
