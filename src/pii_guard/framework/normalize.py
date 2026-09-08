"""
Text normalisation utilities for PII candidate extraction.

``normalize_safe``
    Position-preserving normalisation (1-to-1 char replacements).
    Used by the Presidio recognizer — reported positions match the
    original text exactly.

``normalize_text``
    Full normalisation including Russian word-to-digit conversion. Changes text
    length, so span offsets refer to its output -- this is the first step of the
    production pipeline (see ``Engine.anonymize``), not a testing helper.
"""

from __future__ import annotations

import re

# ================================================================
# Position-safe normalisation
# ================================================================


def normalize_safe(text: str) -> str:
    """Normalise *text* **without** changing its length.

    Only 1-to-1 character replacements (Unicode spaces → ASCII space,
    Unicode dashes → ASCII hyphen).
    """
    out = text
    for ch in ("\u00a0", "\u202f", "\u2009", "\u2007", "\u2060", "\t"):
        out = out.replace(ch, " ")
    for ch in ("\u2014", "\u2013", "\u2010", "\u2011", "\u2012", "\u2212"):
        out = out.replace(ch, "-")
    return out


# ================================================================
# Full normalisation (may change text length)
# ================================================================

_DIGIT_WORD_TO_DIGIT = {
    "ноль": "0", "нуль": "0",
    "один": "1", "одна": "1",
    "два": "2", "две": "2",
    "три": "3", "четыре": "4",
    "пять": "5", "шесть": "6",
    "семь": "7", "восемь": "8",
    "девять": "9",
}

_DIGIT_WORD_ALT = "|".join(
    sorted(map(re.escape, _DIGIT_WORD_TO_DIGIT.keys()), key=len, reverse=True)
)
_DIGIT_WORD_SEQ_RE = re.compile(
    rf"(?iu)\b(?:{_DIGIT_WORD_ALT})"
    rf"(?:(?:[\s,\.;:\-]+|\s+и\s+)(?:{_DIGIT_WORD_ALT}))*\b"
)

_NUMBER_UNITS = {
    "ноль": 0, "нуль": 0, "один": 1, "одна": 1,
    "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
    "девять": 9,
}
_NUMBER_TEENS = {
    "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19,
}
_NUMBER_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40,
    "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70,
    "восемьдесят": 80, "девяносто": 90,
}
_NUMBER_HUNDREDS = {
    "сто": 100, "двести": 200, "дваста": 200,
    "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
    "девятьсот": 900,
}
_NUMBER_THOUSANDS = {"тысяча", "тысячи", "тысяч"}

_NUMBER_WORDS = (
    set(_NUMBER_UNITS) | set(_NUMBER_TEENS)
    | set(_NUMBER_TENS) | set(_NUMBER_HUNDREDS)
    | _NUMBER_THOUSANDS
)
_NUMBER_WORD_ALT = "|".join(
    sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True)
)
_NUMBER_SEQ_RE = re.compile(
    rf"(?iu)\b(?:{_NUMBER_WORD_ALT})"
    rf"(?:(?:[\s,\.;:\-]+|\s+и\s+)(?:{_NUMBER_WORD_ALT}))*\b"
)
_NUMBER_TOKEN_RE = re.compile(rf"(?iu){_NUMBER_WORD_ALT}")


def _consume_ru_number_upto_1000(
    tokens: list[str], start: int,
) -> tuple[int | None, int]:
    """Consume one cardinal number (0..1000) from *tokens* at *start*."""
    i = start
    n = len(tokens)
    value = 0

    if i >= n:
        return None, 0

    t = tokens[i]
    if t in _NUMBER_THOUSANDS:
        return 1000, 1
    if t in ("один", "одна") and i + 1 < n and tokens[i + 1] in _NUMBER_THOUSANDS:
        return 1000, 2

    if t in _NUMBER_HUNDREDS:
        value += _NUMBER_HUNDREDS[t]
        i += 1
        if i >= n:
            return value, i - start
        t = tokens[i]

    if t in _NUMBER_TEENS:
        value += _NUMBER_TEENS[t]
        i += 1
        return value, i - start

    if t in _NUMBER_TENS:
        value += _NUMBER_TENS[t]
        i += 1
        if i < n and tokens[i] in _NUMBER_UNITS:
            value += _NUMBER_UNITS[tokens[i]]
            i += 1
        return value, i - start

    if t in _NUMBER_UNITS:
        value += _NUMBER_UNITS[t]
        i += 1
        return value, i - start

    return None, 0


def _parse_ru_number_stream(tokens: list[str]) -> list[int] | None:
    """Parse a stream of cardinal-number tokens into a value sequence."""
    out: list[int] = []
    i = 0
    while i < len(tokens):
        val, consumed = _consume_ru_number_upto_1000(tokens, i)
        if consumed == 0 or val is None or val < 0 or val > 1000:
            return None
        out.append(val)
        i += consumed
    return out if out else None


def normalize_text(text: str) -> str:
    """Full normalisation (may change text length).

    Includes everything from ``normalize_safe`` plus:

    * Russian digit-word sequences → digit strings.
    * Russian cardinal numbers → numeric strings.
    * collapses whitespace runs to a single space.
    """
    normalized = normalize_safe(text)

    normalized = re.sub(r"\u00ad", "", normalized)  # soft hyphen
    normalized = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", normalized)  # zero-width chars
    normalized = re.sub(r" {2,}", " ", normalized)

    def _neighbor_word(src: str, index: int, *, right: bool) -> str | None:
        if right:
            m = re.match(r"(?iu)\W*([а-яё]+)", src[index:])
            return m.group(1).lower() if m else None
        m = re.search(r"(?iu)([а-яё]+)\W*$", src[:index])
        return m.group(1).lower() if m else None

    def _repl_digit_words(match: re.Match[str]) -> str:
        raw = match.group(0)
        tokens = re.findall(rf"(?iu){_DIGIT_WORD_ALT}", raw)
        if len(tokens) < 4:
            return raw
        prev_word = _neighbor_word(match.string, match.start(), right=False)
        next_word = _neighbor_word(match.string, match.end(), right=True)
        if (
            prev_word in _NUMBER_WORDS and prev_word not in _DIGIT_WORD_TO_DIGIT
        ) or (
            next_word in _NUMBER_WORDS and next_word not in _DIGIT_WORD_TO_DIGIT
        ):
            return raw
        return "".join(_DIGIT_WORD_TO_DIGIT[t.lower()] for t in tokens)

    normalized = _DIGIT_WORD_SEQ_RE.sub(_repl_digit_words, normalized)

    def _repl_number_words(match: re.Match[str]) -> str:
        chunk = match.group(0)
        tokens = [t.lower() for t in _NUMBER_TOKEN_RE.findall(chunk)]
        if not tokens:
            return chunk
        if len(tokens) < 2:
            t = tokens[0]
            if (
                t not in _NUMBER_HUNDREDS
                and t not in _NUMBER_TENS
                and t not in _NUMBER_TEENS
                and t not in _NUMBER_THOUSANDS
            ):
                return chunk
        if all(t in _DIGIT_WORD_TO_DIGIT for t in tokens):
            return chunk
        numbers = _parse_ru_number_stream(tokens)
        if not numbers:
            return chunk
        if len(numbers) == 1:
            return str(numbers[0])
        return " ".join(str(x) for x in numbers)

    normalized = _NUMBER_SEQ_RE.sub(_repl_number_words, normalized)

    return normalized
