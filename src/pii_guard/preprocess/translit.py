"""Детекция и обратная транслитерация русского текста, записанного латиницей."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TRANSLIT_PAIRS: list[tuple[str, str]] = [
    # 4–3 символа
    ("shch", "щ"),
    ("sch", "щ"),
    ("tch", "ч"),
    ("iya", "ия"),
    ("iyu", "ию"),
    ("iye", "ие"),
    # 2 символа (диграфы)
    ("yo", "ё"),
    ("ya", "я"),
    ("yu", "ю"),
    ("ye", "е"),
    ("yi", "и"),
    ("iy", "ий"),
    ("zh", "ж"),
    ("kh", "х"),
    ("ch", "ч"),
    ("sh", "ш"),
    ("ts", "ц"),
    ("ju", "ю"),
    ("ja", "я"),
    ("jo", "ё"),
    ("je", "е"),
    # 1 символ
    ("a", "а"),
    ("b", "б"),
    ("v", "в"),
    ("w", "в"),
    ("g", "г"),
    ("d", "д"),
    ("e", "е"),
    ("z", "з"),
    ("i", "и"),
    ("k", "к"),
    ("l", "л"),
    ("m", "м"),
    ("n", "н"),
    ("o", "о"),
    ("p", "п"),
    ("r", "р"),
    ("s", "с"),
    ("t", "т"),
    ("u", "у"),
    ("f", "ф"),
    ("h", "х"),
    ("c", "к"),
    ("j", "й"),
    ("y", "ы"),
    ("x", "кс"),
    ("q", "к"),
    ("'", "ь"),
    ("`", "ь"),
]
_TRANSLIT_PAIRS.sort(key=lambda pair: -len(pair[0]))

_LATIN_WORD_RE = re.compile(r"[A-Za-z'`]+")

_MAX_TRANSLIT_WORD = 20

_STRUCTURAL_RE = re.compile(
    r"\S*[@=+]\S*"                                   # email / base64 / закодированное
    r"|(?:https?://|www\.)\S+"                       # URL
    rf"|[A-Za-z0-9/]{{{_MAX_TRANSLIT_WORD + 1},}}"  # сверхдлинный run (> лимита)
)


def _mask_structural(text: str) -> str:
    """Заменяет «структурные» токены (email/url/base64/сверхдлинные) пробелами."""
    return _STRUCTURAL_RE.sub(lambda m: " " * (m.end() - m.start()), text)


def _structural_spans(text: str) -> list[tuple[int, int]]:
    """Координаты структурных токенов (email/url/base64/сверхдлинные)."""
    return [(m.start(), m.end()) for m in _STRUCTURAL_RE.finditer(text)]

# Характерные для русского транслита диграфы/паттерны.
_TRANSLIT_HINTS = (
    "ya", "yu", "yo", "zh", "kh", "ts", "sch", "shch", "tsya", "iy",
    "ij", "ju", "ja",
)

_RU_TRANSLIT_STOPWORDS = frozenset(
    {
        "ya", "menya", "mne", "moj", "moya", "moyo", "moi", "tvoj", "vash",
        "nomer", "zovut", "iz", "goroda", "gorod", "ulitsa", "dom", "kvartira",
        "telefon", "pochta", "let", "goda", "god", "rozhdeniya", "prozhivayu",
        "zhivu", "adres", "familiya", "imya", "otchestvo", "pasport", "seriya",
        "eto", "privet", "zdravstvujte", "spasibo", "pozhalujsta", "dobryj",
        "den", "nas", "vas", "ego", "ee", "svoj", "etot", "etogo", "kotoryj",
    }
)

_EN_STOPWORDS = frozenset(
    {
        "the", "and", "is", "are", "was", "were", "have", "has", "had", "will",
        "would", "should", "could", "this", "that", "these", "those", "with",
        "from", "your", "you", "their", "there", "here", "what", "when", "where",
        "which", "who", "whom", "about", "into", "over", "under", "than", "then",
        "them", "they", "she", "him", "her", "his", "its", "our", "for", "not",
        "but", "all", "any", "can", "may", "such", "been", "being", "does", "did",
    }
)


@dataclass(frozen=True)
class _Piece:
    """Cоответствие фрагмента кириллицы фрагменту источника.

    ``verbatim=True`` — фрагмент перенесён без изменений (пробелы, цифры,
    пунктуация, email/url), длины совпадают и отображение посимвольное 1:1.
    ``verbatim=False`` — транслитерированное латинское слово, длины могут
    отличаться (отображаются границы слова целиком).
    """

    dst_start: int
    dst_end: int
    src_start: int
    src_end: int
    verbatim: bool


@dataclass
class SpanMap:
    """Карта соответствия позиций кириллического текста исходному."""


    cyrillic: str
    pieces: list[_Piece]

    def _map_boundary(self, pos: int, *, is_end: bool) -> int | None:
        """Переводит граничную позицию *pos* кириллицы в позицию источника."""
        target = pos - 1 if is_end else pos
        for p in self.pieces:
            if p.dst_start <= target < p.dst_end:
                if p.verbatim:
                    return p.src_start + (pos - p.dst_start)
                return p.src_end if is_end else p.src_start
        if is_end and self.pieces:
            return self.pieces[-1].src_end
        return None

    def to_source_span(self, dst_start: int, dst_end: int) -> tuple[int, int] | None:
        """Переводит спан ``[dst_start, dst_end)`` кириллицы в координаты источника."""
        if dst_end <= dst_start or not self.pieces:
            return None
        src_start = self._map_boundary(dst_start, is_end=False)
        src_end = self._map_boundary(dst_end, is_end=True)
        if src_start is None or src_end is None or src_end <= src_start:
            return None
        return src_start, src_end


def _convert_word(word: str) -> str:
    """Жадная обратная транслитерация одного слова с сохранением регистра."""
    out: list[str] = []
    i = 0
    lower = word.lower()
    n = len(lower)
    while i < n:
        for src, dst in _TRANSLIT_PAIRS:
            if lower.startswith(src, i):
                if word[i].isupper():
                    dst = dst[:1].upper() + dst[1:]
                out.append(dst)
                i += len(src)
                break
        else:  # символ не покрыт картой — оставляем как есть
            out.append(word[i])
            i += 1
    return "".join(out)


def transliterate_to_cyrillic(text: str) -> SpanMap:
    """Транслитерирует *text* в кириллицу пословно, сохраняя структуру.

    Латинские слова заменяются кириллическими образами; пробелы, цифры,
    пунктуация и любые нелатинские символы переносятся без изменений (что
    сохраняет позиции номеров телефонов, паспортов и т.п. и позволяет
    контекстным распознавателям увидеть кириллические ключевые слова).
    """
    structural = _structural_spans(text)

    def _in_structural(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e in structural)

    pieces: list[_Piece] = []
    out: list[str] = []
    dst_len = 0
    last = 0

    def _emit(src_start: int, src_end: int, dst_text: str, verbatim: bool) -> None:
        nonlocal dst_len
        if not dst_text and src_start == src_end:
            return
        out.append(dst_text)
        pieces.append(_Piece(dst_len, dst_len + len(dst_text), src_start, src_end, verbatim))
        dst_len += len(dst_text)

    for match in _LATIN_WORD_RE.finditer(text):
        # неизменяемый кусок до слова (пробелы, цифры, пунктуация) — 1:1
        if match.start() > last:
            gap = text[last:match.start()]
            _emit(last, match.start(), gap, verbatim=True)

        word = match.group()
        ws, we = match.start(), match.end()
        last = we

        if _in_structural(ws, we):
            # структурный токен (email/url/...) — дословно, 1:1
            _emit(ws, we, word, verbatim=True)
        else:
            _emit(ws, we, _convert_word(word), verbatim=False)

    if last < len(text):
        _emit(last, len(text), text[last:], verbatim=True)

    return SpanMap(cyrillic="".join(out), pieces=pieces)


def looks_like_translit(text: str, *, min_latin_words: int = 2) -> bool:
    """Эвристически определяет, что *text* — русский транслит, а не английский.

    Критерии (все должны выполняться):
    * есть минимум ``min_latin_words`` латинских слов;
    * латиница доминирует над кириллицей (иначе текст уже частично русский);
    * присутствуют русские транслит-сигналы (служебные слова в транслите или
      характерные диграфы ``ya/yu/sch/...``);
    * доля английских стоп-слов мала (иначе это обычный английский).
    """

    text = _mask_structural(text)

    latin_words = [w for w in _LATIN_WORD_RE.findall(text) if len(w) <= _MAX_TRANSLIT_WORD]
    if len(latin_words) < min_latin_words:
        return False

    cyrillic_chars = sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch in "ёЁ")
    latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if latin_chars <= cyrillic_chars:
        return False

    lowered = [w.lower() for w in latin_words]
    en_hits = sum(1 for w in lowered if w in _EN_STOPWORDS)
    ru_hits = sum(1 for w in lowered if w in _RU_TRANSLIT_STOPWORDS)
    digraph_hits = sum(1 for w in lowered for h in _TRANSLIT_HINTS if h in w)

    # Сильный английский контекст — не транслит.
    if en_hits >= 2 and en_hits > ru_hits:
        return False

    # Нужен хотя бы один русский сигнал.
    return ru_hits >= 1 or digraph_hits >= 1
