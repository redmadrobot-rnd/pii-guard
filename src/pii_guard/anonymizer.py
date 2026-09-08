"""Public façade: detection + the chosen output mode + de-anonymisation.

This is the one class most callers need::

    from pii_guard import Anonymizer

    a = Anonymizer()
    r = a.anonymize("Иван Петров, ИНН 7707083893", mode="pseudonymize")
    llm_answer = call_your_llm(r.texts[0])
    original = a.deanonymize(llm_answer, r.mapping)

Deliberately stateless with respect to the placeholder mapping: it is returned to
the caller, never stored. That removes the mapping store, its TTL and the
correlation id the extracted service needed -- and with them the failure mode
where a mapping expired between the request and the response.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from pii_guard.config import Config
from pii_guard.engine import Engine
from pii_guard.operators import Mode, build_tagged_text, validate_mode

logger = logging.getLogger("pii_guard.anonymizer")


@dataclass
class AnonymizeResult:
    """Outcome of one :meth:`Anonymizer.anonymize` call.

    ``texts``
        Parallel to the input list, rendered per the requested mode.
    ``entities``
        One list per input text: ``{start, end, entity_type, score}``.
    ``normalized_texts``
        The text the spans refer to. **Offsets in** ``entities`` **index this
        string, not the input.** ``normalize_text`` and the English-numeral
        preprocessor can change length; the transliteration pass maps spans back,
        and base64 blobs collapse to the blob's own boundaries. Compare against
        ``normalized_texts`` when you slice.
    ``mapping``
        ``{tag: original}`` -- non-empty only for ``mode="pseudonymize"``. Pass it
        to :meth:`Anonymizer.deanonymize` to restore the original values.
    ``state``
        Pseudonymisation counters and dedup maps. Feed it back into the next call
        to keep ids stable across turns of the same conversation.
    """

    texts: list[str] = field(default_factory=list)
    entities: list[list[dict]] = field(default_factory=list)
    normalized_texts: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)
    mode: Mode = "mask"
    state: object | None = None

    @property
    def text(self) -> str:
        """First result — convenience for the single-text case."""
        return self.texts[0] if self.texts else ""


class Anonymizer:
    """Thread-safe wrapper around :class:`pii_guard.engine.Engine`.

    Detection is serialised through a lock. spaCy pipelines and a torch module
    are not safe to drive from several threads at once, and in the service this
    code came from the queue consumer enforced the same one-at-a-time discipline
    (``asyncio.Semaphore(1)``). Callers wanting throughput should run several
    processes, not several threads.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        ner: bool | None = None,
        engine: Engine | None = None,
    ) -> None:
        self.config = config or Config.load()
        self._lock = threading.Lock()
        self._engine = engine or Engine(self.config, ner=ner)

        # One config drives both halves: the engine's spaCy pipeline and the
        # morphology pipeline used for grammatical-case restoration.
        from pii_guard import pseudonymize as _pseudo

        _pseudo.configure_nlp(self.config.spacy_model)
        self._pseudo = _pseudo

    # ── properties ────────────────────────────────────────────────

    @property
    def ner_enabled(self) -> bool:
        """False when running rules-only.

        Without the model the four NER-only types -- see
        :data:`pii_guard.config.NER_ONLY_ENTITIES` -- yield nothing.
        """
        return self._engine.ner_enabled

    # ── anonymise ─────────────────────────────────────────────────

    def anonymize(
        self,
        texts: str | list[str],
        *,
        mode: Mode = "mask",
        entities: list[str] | None = None,
        state: object | None = None,
    ) -> AnonymizeResult:
        """Detect PII in *texts* and render it according to *mode*.

        *entities* is an allowlist of entity types; ``None`` uses
        ``config.entities``. The allowlist gates the reported spans as well as
        the masking, so an empty list yields the text unchanged *and* an empty
        entity list -- it is "detect nothing", not "detect without masking".
        *state* continues a previous pseudonymisation session so repeated values
        keep their ids.
        """
        validate_mode(mode)
        items = [texts] if isinstance(texts, str) else list(texts)
        result = AnonymizeResult(mode=mode)
        if not items:
            return result

        allowed = set(self.config.entities if entities is None else entities)
        allowed = {e.strip().upper() for e in allowed if str(e).strip()}

        masked: list[str] = []
        with self._lock:
            for text in items:
                anonymized_text, found, normalized = self._engine.anonymize(
                    text, allowed_entities=allowed
                )
                masked.append(anonymized_text)
                # Conflict resolution emits spans in resolution order; sort by
                # position so consumers can walk them alongside the text.
                result.entities.append(sorted(found, key=lambda e: (e["start"], e["end"])))
                result.normalized_texts.append(normalized)

        if mode == "mask":
            result.texts = masked
        elif mode == "tag":
            result.texts = [
                build_tagged_text(norm, ents)
                for norm, ents in zip(result.normalized_texts, result.entities, strict=True)
            ]
        else:  # pseudonymize
            pseudo_texts, mapping, new_state = self._pseudo.pseudonymize_batch(
                result.normalized_texts, result.entities, state=state
            )
            result.texts = pseudo_texts
            result.mapping = mapping
            result.state = new_state

        return result

    # ── de-anonymise ──────────────────────────────────────────────

    def deanonymize(self, texts: str | list[str], mapping: dict[str, str]) -> list[str]:
        """Restore original values in *texts* using *mapping*.

        PERSON and LOCATION are re-inflected to the grammatical case their new
        syntactic context requires; every other type is substituted verbatim.
        Tags absent from *mapping* -- typically hallucinated by a model -- are
        scrubbed to ``****`` rather than left in place.
        """
        items = [texts] if isinstance(texts, str) else list(texts)
        # An empty mapping is not a reason to skip the pass. The scrubbing of
        # unknown tags lives inside `deanonymize_batch`, so short-circuiting here
        # returned a model's invented ``<PII .../>`` tags to the caller verbatim --
        # exactly the case the scrub exists for.
        with self._lock:
            return self._pseudo.deanonymize_batch(
                items, mapping, self.config.spacy_model
            )

    # ── prompt helper ─────────────────────────────────────────────

    @staticmethod
    def pseudonymize_system_prompt() -> str:
        """System prompt instructing a model to keep ``<PII .../>`` tags intact.

        Prepend it when sending pseudonymised text to an LLM: without it models
        tend to rewrite or drop the tags, and de-anonymisation then has nothing
        to match.
        """
        from pathlib import Path

        path = Path(__file__).with_name("prompts") / "pseudonymize_system.txt"
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""


__all__ = ["AnonymizeResult", "Anonymizer"]
