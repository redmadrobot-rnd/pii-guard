"""
Generic context-analysis helpers for PII classification.

Provides distance / proximity helpers and context-window extraction
used by entity classifiers.  **No** entity-specific patterns or
keywords live here — those belong in ``entities/*.py``.
"""

from __future__ import annotations

import re


def has_keyword(regex: re.Pattern[str], text: str) -> bool:
    """Return *True* if *regex* matches anywhere in *text*."""
    return bool(regex.search(text))


def nearest_distance(
    regex: re.Pattern[str], text: str, center: int,
) -> int | None:
    """Minimum distance from any *regex* match centre to *center*."""
    best: int | None = None
    for m in regex.finditer(text):
        m_center = (m.start() + m.end()) // 2
        dist = abs(m_center - center)
        if best is None or dist < best:
            best = dist
    return best


def nearest_before_distance(
    regex: re.Pattern[str],
    text: str,
    candidate_start: int,
    max_dist: int | None = None,
) -> int | None:
    """Distance from nearest keyword **before** *candidate_start* (keyword-end → candidate-start).

    Parameters
    ----------
    candidate_start : int
        Start position of the candidate within *text*.  Passing the exact
        position (rather than the raw string) avoids the ambiguity when the
        same digit sequence appears more than once inside the context window.
    """
    best: int | None = None
    for m in regex.finditer(text):
        if m.end() > candidate_start:
            continue
        dist = candidate_start - m.end()
        if max_dist is not None and dist > max_dist:
            continue
        if best is None or dist < best:
            best = dist
    return best


def get_context(text: str, start: int, end: int, window: int = 80) -> tuple[str, int]:
    """Sentence-local context expanded by *window* characters.

    Returns
    -------
    tuple[str, int]
        ``(context_string, ctx_start)`` where *ctx_start* is the offset of
        the context window start within *text*.  Use it to compute the
        candidate's position inside the returned string:
        ``candidate_pos_in_ctx = start - ctx_start``.
    """
    left_marks = [
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
        text.rfind(";", 0, start),
        text.rfind("\n", 0, start),
    ]
    sent_start = max(left_marks) + 1

    right_candidates = []
    for mark in (".", "!", "?", ";", "\n"):
        pos = text.find(mark, end)
        if pos != -1:
            right_candidates.append(pos)
    sent_end = min(right_candidates) if right_candidates else len(text)

    local_start = max(0, sent_start - window)
    local_end = min(len(text), sent_end + window)
    return text[local_start:local_end], local_start


def get_wide_context(
    text: str, start: int, end: int, window: int = 260,
) -> tuple[str, int]:
    """Wider context window (*window* chars each side).

    Returns
    -------
    tuple[str, int]
        ``(context_string, ctx_start)`` — same convention as
        :func:`get_context`.
    """
    ctx_start = max(0, start - window)
    return text[ctx_start: min(len(text), end + window)], ctx_start
