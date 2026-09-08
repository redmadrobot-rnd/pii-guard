"""IP_PORT entity — IPv4/IPv6 addresses with port number.

Self-contained: pattern and Presidio ``PatternRecognizer``
registration in one file.
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

from pii_guard.framework.base import register_pairwise_conflict, register_regex_entity


@register_regex_entity
def register_ip_port(analyzer: AnalyzerEngine, language: str = "ru") -> None:
    """Create and register the IP_PORT recognizer on *analyzer*."""

    ip_port_pattern = Pattern(
        name="IP_with_port",
        regex=(
            r"(?<!\w)(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
            r"|\[[0-9a-fA-F:\.]+\]):\d{1,5}\b"
        ),
        score=0.6,
    )

    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="IP_PORT",
            name="IP_PORT",
            patterns=[ip_port_pattern],
            supported_language=language,
            context=["ip", "адрес", "порт", "подключение", "сервер"],
        )
    )


# ── Conflict rules ──────────────────────────────────────────────
#
# Every IP_PORT match contains a full IPv4 (or a bracketed IPv6), so IP_ADDRESS
# always matches inside it. Without an explicit rule the two only meet in the
# score pass, where IP_ADDRESS scores 0.85 against 0.6 here and wins every time:
# the type was declared, tested at the pattern level and never reached the
# output, leaving the port in the clear (`[IP_ADDRESS]:8443`). The longer span is
# the right answer -- the port is part of the same locator.
register_pairwise_conflict(winner="IP_PORT", loser="IP_ADDRESS")
