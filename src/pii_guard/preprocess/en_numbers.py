"""Нормализация английских числительных словами → цифры (перед PII-анализом).

Отдельный модуль под флагом ``PII_GUARD_ENABLE_EN_NUMBERS``. Решает ту же задачу,
что русская ветка в ``framework.normalize``, но не является её точным зеркалом:
множители здесь перемножаются (``three thousand`` → ``3000``), а в русской ветке
множитель и множимое выходят отдельными числами (``три тысячи`` → ``3 1000``).
Для детекции ПДн это безразлично — идентификаторы пишут разрядами, а не
множителями, и именно разрядные цепочки склеиваются ниже.

Логика
------------------------
* Последовательности «цифровых» слов (``zero/oh/one..nine``) длиной ≥4 →
  поразрядная склейка в строку цифр (телефоны, номера документов).
* Прочие кардинальные числа (``twenty five``, ``one hundred twenty three``) →
  числовое значение.
"""

from __future__ import annotations

import re

_DIGIT_WORD: dict[str, str] = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0", "naught": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_TEENS: dict[str, int] = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ONES_VAL: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "zero": 0,
}
_HUNDRED = "hundred"
_THOUSAND = "thousand"

_NUMBER_WORDS = (
    set(_DIGIT_WORD) | set(_TEENS) | set(_TENS) | {_HUNDRED, _THOUSAND}
)
_NUMBER_WORD_ALT = "|".join(
    sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True)
)
_NUMBER_SEQ_RE = re.compile(
    rf"(?i)\b(?:{_NUMBER_WORD_ALT})"
    rf"(?:(?:[\s,\-]+|\s+and\s+)(?:{_NUMBER_WORD_ALT}))*\b"
)
_TOKEN_RE = re.compile(rf"(?i){_NUMBER_WORD_ALT}")


def _consume_cardinal(tokens: list[str], start: int) -> tuple[int | None, int]:
    """Считывает одно кардинальное число (0..thousands) из *tokens* c *start*."""
    i, n, value = start, len(tokens), 0
    if i >= n:
        return None, 0

    matched = False
    if tokens[i] in _ONES_VAL and i + 1 < n and tokens[i + 1] == _HUNDRED:
        value += _ONES_VAL[tokens[i]] * 100
        i += 2
        matched = True
    elif tokens[i] == _HUNDRED:
        value += 100
        i += 1
        matched = True

    if i < n and tokens[i] in _TEENS:
        value += _TEENS[tokens[i]]
        i += 1
        matched = True
    elif i < n and tokens[i] in _TENS:
        value += _TENS[tokens[i]]
        i += 1
        matched = True
        if i < n and tokens[i] in _ONES_VAL and tokens[i] != "zero":
            value += _ONES_VAL[tokens[i]]
            i += 1
    elif i < n and tokens[i] in _ONES_VAL:
        value += _ONES_VAL[tokens[i]]
        i += 1
        matched = True

    if i < n and tokens[i] == _THOUSAND:
        value = (value or 1) * 1000
        i += 1
        matched = True

    if not matched:
        return None, 0
    return value, i - start


def _parse_cardinal_stream(tokens: list[str]) -> list[int] | None:
    out: list[int] = []
    i = 0
    while i < len(tokens):
        val, consumed = _consume_cardinal(tokens, i)
        if consumed == 0 or val is None:
            return None
        out.append(val)
        i += consumed
    return out or None


def _replace_run(match: re.Match[str]) -> str:
    chunk = match.group(0)
    tokens = [t.lower() for t in _TOKEN_RE.findall(chunk)]
    if not tokens:
        return chunk

    if len(tokens) >= 4 and all(t in _DIGIT_WORD for t in tokens):
        return "".join(_DIGIT_WORD[t] for t in tokens)

    # A lone digit word ("call me at one") is prose far more often than data.
    # Only runs are worth rewriting. `_DIGIT_WORD` holds units and the zero
    # spellings exclusively -- teens, tens, hundred and thousand are not in it,
    # so no extra exclusions are needed here.
    if len(tokens) == 1 and tokens[0] in _DIGIT_WORD:
        return chunk

    numbers = _parse_cardinal_stream(tokens)
    if not numbers:
        return chunk
    if len(numbers) == 1:
        return str(numbers[0])
    return " ".join(str(x) for x in numbers)


def normalize_english_numbers(text: str) -> str:
    """Приводит английские числительные словами к цифрам.

    Возвращает текст с заменёнными последовательностями; остальной текст,
    пунктуация и существующие цифры не меняются.
    """
    return _NUMBER_SEQ_RE.sub(_replace_run, text)
