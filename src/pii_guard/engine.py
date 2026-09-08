"""
Presidio-based PII anonymisation service.

Detection pipeline
------------------
Two branches run independently and are merged before conflict resolution:

**Branch A — rules** (deterministic):
  1. Numeric / document entities — auto-discovered from ``entities/``
     (checksum validation + keyword context).
  2. Regex entities — DATE_TIME, EMAIL_ADDRESS, IP_ADDRESS (IPv4/IPv6),
     IP_PORT via Presidio ``PatternRecognizer``.

**Branch B — NER** (ML model):
  - PERSON (FIRST_NAME, LAST_NAME, MIDDLE_NAME)
  - LOCATION (CITY, COUNTRY, DISTRICT, REGION, STREET, HOUSE)
  - EMAIL_ADDRESS, PHONE_NUMBER, URL, IP_ADDRESS
  - Documents in noisy / OCR-distorted format

Merge logic (``resolve_ml_vs_rules_conflicts``):
  - Conflicting spans → rules win (deterministic math beats ML).
  - Non-overlapping → results from both branches are kept.

After merging the existing 5-pass ``resolve_conflicts`` runs for
deduplication and fine-grained overlap resolution.
"""

from __future__ import annotations

import logging

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from pii_guard.config import NER_ONLY_ENTITIES, Config
from pii_guard.detect import (
    NumericPIIRecognizer,
    register_regex_recognizers,
    resolve_conflicts,
    resolve_ml_vs_rules_conflicts,
)
from pii_guard.entities._email_shape import filter_ner_email_spans
from pii_guard.framework.normalize import normalize_text
from pii_guard.preprocess import (
    find_base64_spans,
    is_confident_foreign,
    langguard_warmup,
    looks_like_translit,
    normalize_english_numbers,
    transliterate_to_cyrillic,
)

logger = logging.getLogger("pii_guard.engine")

# NER label -> Presidio entity type. This dict is the contract between the code
# and the published weights: it must stay in step with the model's ``id2label``.
NER_ENTITY_MAPPING: dict[str, str] = {
    "CITY": "LOCATION",
    "COUNTRY": "LOCATION",
    "DISTRICT": "LOCATION",
    "REGION": "LOCATION",
    "STREET": "LOCATION",
    "HOUSE": "LOCATION",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "MIDDLE_NAME": "PERSON",
    "EMAIL": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER",
    "URL": "URL",
    "IP_ADDRESS": "IP_ADDRESS",
    "PASSPORT": "PASSPORT",
    "CREDIT_CARD": "CREDIT_CARD",
    "DRIVER_LICENSE": "DRIVER_LICENSE",
    "INN": "INN",
    "SNILS": "SNILS",
    "MILITARY_ID": "MILITARY_ID",
    "BIRTH_CERTIFICATE": "BIRTH_CERTIFICATE",
    "OMS": "OMS",
}


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Заменяет участки *spans* пробелами равной длины (offset-safe).

    Используется, чтобы исключить base64-блобы из транслит-прохода, сохранив
    позиции остального текста.
    """
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = " "
    return "".join(chars)


# ── Бюджет на base64-блобы ──────────────────────────────────────
# Каждый блоб — отдельный вложенный проход. Текст на `max_text_chars` даёт 5882
# блоба и 28 с CPU только на правилах, то есть ~30 минут на запрос из 64 текстов.
# Реальные входы — единицы закодированных полей, не тысячи.
MAX_B64_BLOBS = 32
MAX_B64_DECODED_CHARS = 20_000


class Base64BudgetExceeded(ValueError):
    """Текст содержит больше base64, чем движок готов разобрать.

    Отказ вместо частичного разбора: пропустить блоб — значит вернуть его
    содержимое в открытом виде, то есть обойти маскирование. Сообщение содержит
    только счётчики, никогда сам текст.
    """


def _non_overlapping(
    candidates: list[RecognizerResult], kept: list[RecognizerResult]
) -> list[RecognizerResult]:
    """Из *candidates* оставляет то, что не перекрывается ни с одним *kept*."""
    out = []
    for cand in candidates:
        if not any(cand.start < k.end and k.start < cand.end for k in kept):
            out.append(cand)
    return out


class Engine:
    """Both detection branches plus conflict resolution.

    The NER branch (Branch B) is optional: with ``ner=False`` -- or with
    ``PII_GUARD_NER_DISABLED=1``, or no ``ner_model`` configured -- only the rules
    branch runs, and neither torch nor the model weights are touched. That keeps
    the package usable without a 700 MB download, at the cost of the four
    NER-only types (see :data:`pii_guard.config.NER_ONLY_ENTITIES`), which regex
    cannot find in free-form text.
    """

    def __init__(self, config: Config | None = None, *, ner: bool | None = None) -> None:
        self.config = config or Config.load()
        lang = self.config.language

        # ── SpaCy NLP engine (used by Presidio for PatternRecognizers) ─
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": lang, "model_name": self.config.spacy_model}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # Disable SpaCy NER — NER is handled by TransformerNERRecognizer
        nlp_engine.nlp[lang].disable_pipe("ner")

        # Rules analyzer: only our recognizers — clear Presidio built-ins.
        # Loading the registry logs a warning per built-in recognizer that does
        # not support `lang` (~18 lines for ru). They are irrelevant because the
        # registry is cleared on the next line, so keep them out of the log.
        presidio_log = logging.getLogger("presidio-analyzer")
        previous_level = presidio_log.level
        presidio_log.setLevel(logging.ERROR)
        try:
            self._rules_analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=[lang],
            )
        finally:
            presidio_log.setLevel(previous_level)
        self._rules_analyzer.registry.recognizers.clear()
        self.anonymizer = AnonymizerEngine()

        # Register rules pipeline (Branch A)
        self._setup_rules_pipeline()

        # Register NER pipeline (Branch B) — optional
        self._ner_adapter = None
        self.ner_enabled = self.config.ner_enabled if ner is None else ner
        if self.ner_enabled:
            self._setup_ner_pipeline()
        else:
            logger.warning(
                "event=ner_disabled entities_unavailable=%s "
                "(rules branch only; set PII_GUARD_NER_MODEL to enable)",
                ",".join(sorted(NER_ONLY_ENTITIES)),
            )

        if self.config.enable_translit:
            langguard_warmup()

    # ------------------------------------------------------------------
    # Pipeline setup
    # ------------------------------------------------------------------

    def _setup_rules_pipeline(self) -> None:
        """Branch A: numeric classifiers + regex PatternRecognizers."""
        lang = self.config.language

        # Numeric / document PII (checksums + keywords)
        self._numeric_recognizer = NumericPIIRecognizer(
            supported_language=lang,
        )
        self._rules_analyzer.registry.add_recognizer(self._numeric_recognizer)

        # Regex patterns: DATE_TIME, EMAIL_ADDRESS, IP_ADDRESS (IPv4/IPv6), IP_PORT
        register_regex_recognizers(self._rules_analyzer, language=lang)

    def _setup_ner_pipeline(self) -> None:
        """Branch B: transformer NER model.

        Imported here rather than at module scope so that a rules-only install
        (no ``ner`` extra) can import this module at all.
        """
        from pii_guard.ner.adapter import NERAdapter
        from pii_guard.ner.recognizer import TransformerNERConfig, TransformerNERRecognizer

        transformer_config = TransformerNERConfig(
            model_name=self.config.ner_model,
            revision=self.config.ner_revision,
            max_length=self.config.ner_max_length,
            stride=self.config.ner_stride,
            score=self.config.ner_score,
            supported_language=self.config.language,
            device=self.config.device,
        )

        ner_recognizer = TransformerNERRecognizer(
            config=transformer_config,
            entity_mapping=NER_ENTITY_MAPPING,
            name="TransformerNERRecognizer",
        )
        self._ner_adapter = NERAdapter(ner_recognizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _analyze_raw(
        self, text: str
    ) -> tuple[list[RecognizerResult], list[RecognizerResult]]:
        """Прогоняет обе ветки детекции по *text* без разрешения конфликтов.

        Возвращает ``(rules_results, ner_results)`` со спанами относительно
        *text*. Вынесено в отдельный метод, чтобы переиспользовать на
        кириллической версии при включённом транслит-препроцессоре.
        """
        rules_results = self._rules_analyzer.analyze(
            text=text, language=self.config.language
        )
        ner_results = self._ner_adapter.analyze(text) if self._ner_adapter else []
        # Shape-гейт: отсекаем NER-«email», не похожие на почту по форме
        # (соц-ники "@user_id", "instagram @gmail"), сохраняя обфусцированные
        # настоящие почты. Логика — в pii_guard/entities/_email_shape.py.
        ner_results = filter_ner_email_spans(ner_results, text)
        return list(rules_results), list(ner_results)

    @staticmethod
    def _remap_results(
        results: list[RecognizerResult], span_map
    ) -> list[RecognizerResult]:
        """Переносит спаны из кириллической версии в координаты исходного текста."""
        remapped: list[RecognizerResult] = []
        for res in results:
            mapped = span_map.to_source_span(res.start, res.end)
            if mapped is None:
                continue
            src_start, src_end = mapped
            remapped.append(
                RecognizerResult(
                    entity_type=res.entity_type,
                    start=src_start,
                    end=src_end,
                    score=res.score,
                )
            )
        return remapped

    def anonymize(
        self,
        text: str,
        mask_char: str | None = None,
        mask_chars_to_mask: int | None = None,
        mask_from_end: bool | None = None,
        allowed_entities: set[str] | None = None,
    ) -> tuple[str, list[dict[str, object]], str]:
        text = normalize_text(text)

        if self.config.enable_en_numbers:
            text = normalize_english_numbers(text)

        b64_spans: list[tuple[int, int]] = []
        b64_rules: list[RecognizerResult] = []
        b64_ner: list[RecognizerResult] = []
        if self.config.enable_base64:
            blobs = find_base64_spans(text)
            decoded_chars = sum(len(b.decoded) for b in blobs)
            # Refuse, never skip: skipping spends the budget in text order, so a
            # prefix of cheap blobs pushes a real encoded value out of analysis and
            # into the clear -- the budget would turn a DoS into a bypass. Finding
            # the blobs is cheap (5882 in 9 ms); analysing each is not.
            if len(blobs) > MAX_B64_BLOBS or decoded_chars > MAX_B64_DECODED_CHARS:
                raise Base64BudgetExceeded(
                    f"text carries {len(blobs)} base64 blobs / {decoded_chars} decoded "
                    f"characters, limits are {MAX_B64_BLOBS} / {MAX_B64_DECODED_CHARS}"
                )
            for blob in blobs:
                # The span is always recorded: it is what excludes the blob from
                # the transliteration pass, and it costs nothing.
                b64_spans.append((blob.start, blob.end))
                d_rules, d_ner = self._analyze_raw(blob.decoded)
                b64_rules += [
                    RecognizerResult(entity_type=r.entity_type,
                                     start=blob.start, end=blob.end, score=r.score)
                    for r in d_rules
                ]
                b64_ner += [
                    RecognizerResult(entity_type=r.entity_type,
                                     start=blob.start, end=blob.end, score=r.score)
                    for r in d_ner
                ]

        translit_fired = False
        if self.config.enable_translit:
            translit_src = _mask_spans(text, b64_spans) if b64_spans else text
            if looks_like_translit(translit_src) and not is_confident_foreign(
                translit_src, self.config.translit_lang_threshold
            ):
                translit_fired = True
                span_map = transliterate_to_cyrillic(translit_src)
                cyr_rules, cyr_ner = self._analyze_raw(span_map.cyrillic)
                rules_results = self._remap_results(cyr_rules, span_map)
                ner_results = self._remap_results(cyr_ner, span_map)

                # Additive, not a replacement. Taking only three "Latin-native"
                # types from the original text dropped real PII: «seriya IV-ZHA
                # 123456» is a birth certificate whose Roman numeral survives only
                # untransliterated (`IV` → `ИВ` stops matching), so it came back
                # unmasked. On overlap the Cyrillic pass wins -- higher recall is
                # the reason this branch exists.
                orig_rules, orig_ner = self._analyze_raw(text)
                rules_results += _non_overlapping(orig_rules, rules_results)
                ner_results += _non_overlapping(orig_ner, ner_results)

        if not translit_fired:
            rules_results, ner_results = self._analyze_raw(text)


        rules_results += b64_rules
        ner_results += b64_ner

        # Merge: rules win on overlap, NER fills the rest
        entities = resolve_ml_vs_rules_conflicts(rules_results, ner_results)

        # Existing 5-pass conflict resolution (dedup, nesting, score-based)
        entities = resolve_conflicts(
            entities, text, registry=self._numeric_recognizer.registry
        )

        # Keep only the entity types the caller asked to mask. ``None`` means
        # "no allowlist" → mask everything detected.
        if allowed_entities is not None:
            entities = [e for e in entities if e.entity_type in allowed_entities]

        # Build anonymisation operators
        operators = {
            "DEFAULT": OperatorConfig(
                "mask",
                {
                    "masking_char": mask_char
                    if mask_char is not None
                    else self.config.mask_char,
                    "chars_to_mask": mask_chars_to_mask
                    if mask_chars_to_mask is not None
                    else self.config.mask_chars_to_mask,
                    "from_end": mask_from_end
                    if mask_from_end is not None
                    else self.config.mask_from_end,
                },
            )
        }

        result = self.anonymizer.anonymize(
            text=text, analyzer_results=entities, operators=operators
        )

        entity_dicts: list[dict[str, object]] = [
            {
                "start": e.start,
                "end": e.end,
                "entity_type": e.entity_type,
                "score": getattr(e, "score", None),
            }
            for e in entities
        ]
        return result.text, entity_dicts, text
