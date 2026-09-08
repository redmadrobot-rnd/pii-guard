"""IP_ADDRESS entity — IPv4 and IPv6 addresses (full, partial, compressed).

Self-contained: patterns and Presidio ``PatternRecognizer``
registration in one file.
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

from pii_guard.framework.base import register_conflict_handler, register_pairwise_conflict, register_regex_entity
from pii_guard.framework.utils import drop_spans_without_digits


@register_regex_entity
def register_ip_address(analyzer: AnalyzerEngine, language: str = "ru") -> None:
    """Create and register IPv4/IPv6 recognizers on *analyzer*."""

    ipv6_full_pattern = Pattern(
        name="ipv6_full_pattern",
        regex=r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?:%[a-zA-Z0-9]+)?\b",
        score=0.9,
    )
    ipv6_partial_pattern = Pattern(
        name="ipv6_partial_pattern",
        regex=r"\b(?:[0-9a-fA-F]{1,4}:){5,6}[0-9a-fA-F]{1,4}(?:%[a-zA-Z0-9]+)?\b",
        score=0.85,
    )
    ipv6_compressed_pattern = Pattern(
        name="ipv6_compressed_pattern",
        regex=(
            r"(?<![0-9A-Fa-f:])(?=[0-9A-Fa-f:]*\d)(?:(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}::"
            r"(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}"
            r"|::(?:[0-9a-fA-F]{1,4}:){0,7}[0-9a-fA-F]{1,4}"
            r"|(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}::)(?:%[a-zA-Z0-9]+)?(?![0-9A-Fa-f:])"
        ),
        score=0.9,
    )
    ipv4_pattern = Pattern(
        name="ipv4_pattern",
        regex=(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        ),
        score=0.85,
    )

    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="IP_ADDRESS",
            name="IPv4Recognizer",
            patterns=[ipv4_pattern],
            supported_language=language,
            context=["ip", "адрес", "хост", "сервер", "подключение", "ip-адрес"],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="IP_ADDRESS",
            name="IPv6Recognizer",
            patterns=[ipv6_full_pattern, ipv6_partial_pattern, ipv6_compressed_pattern],
            supported_language=language,
            context=["ip", "адрес", "ipv6", "ip-адрес", "сервер"],
        )
    )


# ── Conflict rules ──────────────────────────────────────────────

register_pairwise_conflict(winner="URL", loser="IP_ADDRESS")
register_pairwise_conflict(winner="IP_ADDRESS", loser="PHONE_NUMBER")


@register_conflict_handler
def _drop_ip_address_without_digits(entities, text, already_removed):
    """Drop IP_ADDRESS spans that contain no digits (NER noise).

    Rules-based patterns always require digits; digit-free spans can
    only come from NER mis-classifications (e.g. 'Anya', 'Monday', 'xi').
    """
    return drop_spans_without_digits(entities, text, already_removed, "IP_ADDRESS")
