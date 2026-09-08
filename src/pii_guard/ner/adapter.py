"""NER adapter — thin wrapper around TransformerNERRecognizer.

Accepts raw text and returns a list of ``RecognizerResult`` objects
in the same format as the rules pipeline, ready for merging via
``resolve_ml_vs_rules_conflicts``.
"""

from __future__ import annotations

from presidio_analyzer import RecognizerResult

from pii_guard.ner.recognizer import TransformerNERRecognizer


class NERAdapter:
    """Wraps ``TransformerNERRecognizer`` and exposes a simple ``analyze`` call.

    Parameters
    ----------
    recognizer : TransformerNERRecognizer
        Pre-initialised NER recognizer (created once in ``Engine``).
    """

    def __init__(self, recognizer: TransformerNERRecognizer) -> None:
        self._recognizer = recognizer

    def analyze(self, text: str) -> list[RecognizerResult]:
        """Run NER on *text* and return all found entities."""
        entities = list(self._recognizer.supported_entities)
        return self._recognizer.analyze(text=text, entities=entities)
