"""EMAIL_ADDRESS entity — e-mail addresses (regex).

Self-contained: pattern and Presidio ``PatternRecognizer`` registration in
one file. Complements the NER model: BERT loses an e-mail next to a second
``@`` token ("Почта: hr@gmail.com, телеграм @nick"). This rule catches the
e-mail by its ``local@domain.tld`` shape and wins over NER on overlap
(auto-merge).

Whitespace around ``.``/``-``/``@`` is tolerated — in the dataset e-mails are
written tokenised (``ivanov@example . com``); in normal text this is zero spaces.
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

from pii_guard.framework.base import register_regex_entity

ENTITY_TYPE = "EMAIL_ADDRESS"


@register_regex_entity
def register_email(analyzer: AnalyzerEngine, language: str = "ru") -> None:
    """Create and register the EMAIL_ADDRESS recognizer on *analyzer*."""

    email_pattern = Pattern(
        name="email",
        regex=(
            r"(?<![\w@.])"
            r"[A-Za-z0-9_%+]+(?:\s*[.\-]\s*[A-Za-z0-9_%+]+)*"   # local part
            r"\s*@\s*"
            r"[A-Za-z0-9]+(?:\s*[.\-]\s*[A-Za-z0-9]+)*"          # domain labels
            r"\s*\.\s*[A-Za-z]{2,}"                              # TLD
        ),
        score=0.85,
    )

    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity=ENTITY_TYPE,
            name="EMAIL_ADDRESS",
            patterns=[email_pattern],
            supported_language=language,
            context=["почта", "email", "e-mail", "mail"],
        )
    )
