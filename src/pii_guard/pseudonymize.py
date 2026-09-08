"""
Pseudonymization module — XML tag format.

Replaces PII entities in text with XML self-closing tags:
    PERSON  → <PII type="PERSON" gender="female" id="1" />
    Others  → <PII type="PHONE_NUMBER" id="1" />

PERSON specifics:
    - Consecutive PERSON tokens merged into one span (e.g. first name + last name),
      bounded by MAX_NAME_PARTS so an unpunctuated list of people cannot collapse
      into a single placeholder (see the note on that constant for the residue)
    - Gender detected via pymorphy3
    - Cross-text dedup: same name in a different grammatical form gets the same id
      (fuzzy match on nominative-normalised forms, threshold 85)

POST_CALL deanonymize:
    - Tags are found in LLM output via regex + XML parsing + fuzzy attribute matching
      (handles typos, reordered attributes, multi-line tags from the model)
    - PERSON: name is inserted in nominative, then inflected to the case
      required by its syntactic context (spacy + pymorphy3)
    - All other entity types: direct substitution

Usage
-----
>>> texts = ["У Татьяны Владимировны Козловой нет времени"]
>>> entities_per_text = [[
...     {"start": 2, "end": 31, "entity_type": "PERSON", "score": 0.9},
... ]]
>>> pseudo_texts, mapping, state = pseudonymize_batch(texts, entities_per_text)
>>> # pseudo_texts == ['У <PII type="PERSON" gender="female" id="1" /> нет времени']
>>> # mapping == {'<PII type="PERSON" gender="female" id="1" />': 'Татьяна Владимировна Козлова'}

>>> deanonymize_batch(pseudo_texts, mapping)
['У Татьяны Владимировны Козловой нет времени']
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import pymorphy3
import spacy
from thefuzz import fuzz
from thefuzz import process as fuzz_process

from pii_guard.models import DEFAULT_SPACY_MODEL

morph = pymorphy3.MorphAnalyzer()

_nlp_cache: dict[str, object] = {}
_nlp_model_name = DEFAULT_SPACY_MODEL


def configure_nlp(model_name: str) -> None:
    """Choose the spaCy pipeline used for POS tagging and dependency parsing.

    Called by :class:`pii_guard.anonymizer.Anonymizer` so that one config drives
    both the detection engine and this module -- otherwise a deployment could end
    up downloading two different Russian pipelines for overlapping purposes.

    Sets the process-wide default only. Pipelines are cached per model name, and
    :func:`deanonymize` takes an explicit *model_name* which ``Anonymizer`` fills
    in from its own config -- so a second instance configured with a different
    pipeline no longer changes what the first one uses. With a single cache slot
    and no explicit argument it did: the later constructor evicted the earlier
    pipeline and every subsequent restore silently ran under the wrong model.
    """
    global _nlp_model_name
    if model_name:
        _nlp_model_name = model_name


def _get_nlp(model_name: str | None = None):
    name = model_name or _nlp_model_name
    pipeline = _nlp_cache.get(name)
    if pipeline is None:
        pipeline = spacy.load(name)
        _nlp_cache[name] = pipeline
    return pipeline


# ── Gender detection ──────────────────────────────────────────────


def get_gender_ru(name: str) -> str:
    """Detect grammatical gender of a Russian name string."""
    # First pass: prefer Name/Patr tags (more unambiguous than Surn)
    for tag_priority in ({"Name", "Patr"}, {"Surn"}):
        for token in name.split():
            parses = morph.parse(token)
            for p in parses:
                if tag_priority & set(p.tag.grammemes):
                    if "femn" in p.tag.grammemes:
                        return "female"
                    if "masc" in p.tag.grammemes:
                        return "male"
    return "unknown"


def _get_gender_detector():
    """Return a gender_guesser detector, or ``None`` when it is not installed.

    ``gender-guesser`` is GPLv3 and therefore lives in the optional extra
    ``latin-gender``; it is deliberately absent from the default install and from
    the published images (see NOTICE). Without it Latin-script names get
    ``gender="unknown"`` in their tag, which is cosmetic: de-anonymisation
    re-derives gender from the actual tokens via pymorphy3
    (:func:`_detect_gender_for_inflection`) and never reads the tag attribute.
    """
    try:
        import gender_guesser.detector as gg
        return gg.Detector()
    except ImportError:
        return None


_gender_detector = None


def get_gender_en(name: str) -> str:
    """Detect gender of a Latin name using gender_guesser (token by token)."""
    global _gender_detector
    if _gender_detector is None:
        _gender_detector = _get_gender_detector()
    if _gender_detector is None:
        return "unknown"

    _GG_MAP = {
        "male": "male",
        "mostly_male": "male",
        "female": "female",
        "mostly_female": "female",
        "andy": "unknown",
        "unknown": "unknown",
    }

    for token in name.split():
        result = _gender_detector.get_gender(token)
        mapped = _GG_MAP.get(result, "unknown")
        if mapped != "unknown":
            return mapped
    return "unknown"


def get_gender_universal(name: str) -> str:
    """Try Russian morphology first, fall back to gender_guesser for Latin names."""
    ru_gender = get_gender_ru(name)
    if ru_gender != "unknown":
        return ru_gender
    return get_gender_en(name)


# ── Nominative normalisation ──────────────────────────────────────


def normalize_to_nominative(text: str, gender: str | None = None) -> str:
    """Normalise every token to nominative singular for dedup comparison."""
    morph_gender = {"male": "masc", "female": "femn"}.get(gender or "")
    result = []
    for token in text.split():
        parses = morph.parse(token)
        best = None
        for p in parses:
            if {"Name", "Patr", "Surn"} & set(p.tag.grammemes) and (
                morph_gender is None or morph_gender in p.tag.grammemes
            ):
                best = p
                break
        if best is None:
            for p in parses:
                if {"Name", "Patr", "Surn"} & set(p.tag.grammemes):
                    best = p
                    break
        if best is None:
            best = parses[0]
        inflected = best.inflect({"nomn"})
        result.append(inflected.word.capitalize() if inflected else token)
    return " ".join(result)


_DIGITS_RE = re.compile(r"\d")

# Both case-insensitive on purpose, matching `find_tag_span`: a model asked to
# preserve `<PII .../>` verbatim sometimes lowercases it, and a tag we can still
# recognise must be either restored or scrubbed -- never handed back as-is.
_TAG_OPEN_RE = re.compile(r"<PII\b", re.IGNORECASE)
_TAG_FULL_RE = re.compile(r"<PII\b[^>]*/>", re.IGNORECASE)


def _location_has_digits(text: str) -> bool:
    """Return True if the location value contains digits (house/apt/office numbers)."""
    return bool(_DIGITS_RE.search(text))



def normalize_location(text: str) -> str:
    """
    Normalise a location string for dedup comparison (lowercase nominative).

    Every token is brought to its nominative form via pymorphy3 — the same
    approach used for PERSON.  Digit-containing tokens are kept as-is so that
    house/apartment numbers are not mangled.
    """
    result = []
    for token in text.split():
        if _DIGITS_RE.search(token):
            result.append(token.lower())
        else:
            parses = morph.parse(token)
            inflected = parses[0].inflect({"nomn"}) if parses else None
            result.append(inflected.word.lower() if inflected else token.lower())
    return " ".join(result)


# ── Tag helpers ───────────────────────────────────────────────────


def _build_tag(entity_type: str, entity_id: int, gender: str | None = None) -> str:
    if entity_type == "PERSON" and gender:
        return f'<PII type="PERSON" gender="{gender}" id="{entity_id}" />'
    return f'<PII type="{entity_type}" id="{entity_id}" />'


@dataclass
class PseudonymizationState:
    """Accumulated state across multiple pseudonymize_batch calls in a session."""
    type_counters: dict[str, int] = field(default_factory=dict)
    exact_key_to_tag: dict[str, str] = field(default_factory=dict)
    person_nom_to_tag: dict[str, str] = field(default_factory=dict)
    location_nom_to_tag: dict[str, str] = field(default_factory=dict)
    mapping: dict[str, str] = field(default_factory=dict)


# ── PRE_CALL: pseudonymize ────────────────────────────────────────

_FUZZY_DEDUP_THRESHOLD = 85       # PERSON: min fuzz.ratio to treat as same entity
_LOCATION_FUZZY_THRESHOLD = 90    # LOCATION (text-only): higher bar to avoid false positives


# фамилия, имя, отчество — не больше трёх. NER помечает части отдельно, склейка
# собирает человека обратно, но без предела «Иван Петров Мария Сидорова» слипались
# в один плейсхолдер и восстанавливались одним склеенным именем на двоих.
# Предел это ограничивает, но не решает: без запятой двое всё равно делятся не
# там (3 + 1), потому что метки-компоненты, размечающие границу, схлопнуты в
# PERSON ещё до этой функции. Обе половины маскируются в любом случае — цена в
# идентичности плейсхолдера, не в утечке.
MAX_NAME_PARTS = 3


def _merge_consecutive_persons(entities: list[dict], text: str) -> list[dict]:
    """Merge adjacent PERSON spans separated only by whitespace, up to a full name."""
    if not entities:
        return []
    sorted_ents = sorted(entities, key=lambda e: e.get("start", 0))
    merged: list[dict] = []
    buf: dict | None = None
    parts = 0
    for ent in sorted_ents:
        if ent.get("entity_type") != "PERSON":
            if buf is not None:
                merged.append(buf)
                buf = None
            merged.append(ent)
            continue
        if buf is None:
            buf = dict(ent)
            parts = 1
            continue
        gap = text[buf["end"]: ent["start"]]
        if gap.strip() == "" and parts < MAX_NAME_PARTS:
            buf["end"] = ent["end"]
            buf["score"] = max(buf.get("score") or 0, ent.get("score") or 0)
            parts += 1
        else:
            merged.append(buf)
            buf = dict(ent)
            parts = 1
    if buf is not None:
        merged.append(buf)
    return merged


def _apply_placeholders(
    text: str,
    entities: list[dict],
    type_counters: dict[str, int],
    exact_key_to_tag: dict[str, str],
    person_nom_to_tag: dict[str, str],
    location_nom_to_tag: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """
    Replace entity spans with XML tags.

    *exact_key_to_tag*    — ``"TYPE::normalized_value"`` → canonical tag string.
                            Mutated in-place; carries state across texts in a batch.
    *person_nom_to_tag*   — normalized nominative name → tag (fuzzy dedup for PERSON).
    *location_nom_to_tag* — normalized nominative location → tag (fuzzy dedup for
                            text-only LOCATION values; numeric tokens use exact match).

    Returns ``(tagged_text, {tag_string: original_value})``.
    """
    merged = _merge_consecutive_persons(entities, text)
    merged.sort(key=lambda e: e.get("start", 0))

    assignments: list[tuple[int, int, str, str]] = []

    for ent in merged:
        etype = ent.get("entity_type") or "UNKNOWN"
        original = text[ent["start"]: ent["end"]]
        norm_ws = " ".join(original.split())  # collapse internal whitespace

        if etype == "PERSON":
            detected_gender = get_gender_universal(norm_ws)
            norm_nom = normalize_to_nominative(norm_ws, gender=detected_gender)
        elif etype == "LOCATION":
            detected_gender = None
            norm_nom = normalize_location(norm_ws)
        else:
            detected_gender = None
            norm_nom = norm_ws

        exact_key = f"{etype}::{norm_nom}"

        if exact_key in exact_key_to_tag:
            # Exact match (same type + same normalised value already seen)
            tag = exact_key_to_tag[exact_key]

        elif etype == "PERSON" and person_nom_to_tag:
            # Fuzzy match for PERSON
            known_noms = list(person_nom_to_tag.keys())
            best = fuzz_process.extractOne(norm_nom, known_noms, scorer=fuzz.ratio)
            tag = (
                person_nom_to_tag[best[0]]
                if best and best[1] >= _FUZZY_DEDUP_THRESHOLD
                else None
            )

            if tag is None:
                count = type_counters.get(etype, 0) + 1
                type_counters[etype] = count
                tag = _build_tag(etype, count, detected_gender)

            exact_key_to_tag[exact_key] = tag
            person_nom_to_tag[norm_nom] = tag

        elif etype == "LOCATION" and location_nom_to_tag and not _location_has_digits(norm_nom):
            # Fuzzy match for text-only LOCATION (no digits → safe to fuzzy)
            known_noms = list(location_nom_to_tag.keys())
            best = fuzz_process.extractOne(norm_nom, known_noms, scorer=fuzz.ratio)
            tag = (
                location_nom_to_tag[best[0]]
                if best and best[1] >= _LOCATION_FUZZY_THRESHOLD
                else None
            )

            if tag is None:
                count = type_counters.get(etype, 0) + 1
                type_counters[etype] = count
                tag = _build_tag(etype, count)

            exact_key_to_tag[exact_key] = tag
            location_nom_to_tag[norm_nom] = tag

        else:
            count = type_counters.get(etype, 0) + 1
            type_counters[etype] = count
            if etype == "PERSON":
                tag = _build_tag(etype, count, detected_gender)
                person_nom_to_tag[norm_nom] = tag
            elif etype == "LOCATION":
                tag = _build_tag(etype, count)
                if not _location_has_digits(norm_nom):
                    location_nom_to_tag[norm_nom] = tag
            else:
                tag = _build_tag(etype, count)
            exact_key_to_tag[exact_key] = tag

        # PERSON: store nominative so POST_CALL can inflect to the correct case.
        # LOCATION: store original text — POST_CALL uses spacy POS to inflect.
        # Others: store original text — direct substitution, no inflection.
        if etype == "PERSON":
            stored_value = norm_nom
        elif etype == "LOCATION":
            stored_value = norm_ws
        else:
            stored_value = norm_ws
        assignments.append((ent["start"], ent["end"], tag, stored_value))

    # Replace right-to-left to preserve earlier indices
    mapping: dict[str, str] = {}
    result = text
    for start, end, tag, stored_value in reversed(assignments):
        result = result[:start] + tag + result[end:]
        mapping[tag] = stored_value

    return result, mapping


def pseudonymize_batch(
    texts: list[str],
    entities_per_text: list[list[dict]],
    state: PseudonymizationState | None = None,
) -> tuple[list[str], dict[str, str], PseudonymizationState]:
    """
    Pseudonymize several texts with globally-unique tag numbering.

    Cross-text deduplication: the same entity in different grammatical forms
    receives the same XML tag within the batch.

    If *state* is provided, continues from existing session state (counters,
    dedup maps) so repeated calls across turns reuse the same ids.

    Returns ``(pseudonymized_texts, new_mapping, updated_state)`` where
    new_mapping contains only tags found in this batch, and updated_state
    carries the full accumulated session state.
    """
    if state is None:
        state = PseudonymizationState()

    combined_mapping: dict[str, str] = {}
    result_texts: list[str] = []

    for text, entities in zip(texts, entities_per_text, strict=False):
        pseudo_text, mapping = _apply_placeholders(
            text, entities, state.type_counters, state.exact_key_to_tag,
            state.person_nom_to_tag, state.location_nom_to_tag,
        )
        result_texts.append(pseudo_text)
        combined_mapping.update(mapping)

    state.mapping.update(combined_mapping)
    return result_texts, combined_mapping, state


# ── POST_CALL: deanonymize ────────────────────────────────────────

CASE_MAP = {
    "Nom": "nomn", "Gen": "gent", "Dat": "datv",
    "Acc": "accs", "Ins": "ablt", "Loc": "loct", "Voc": "voct",
}

PREP_CASE = {
    "о": "loct", "об": "loct", "при": "loct",
    "к": "datv", "ко": "datv", "по": "datv",
    "от": "gent", "до": "gent", "из": "gent", "у": "gent",
    "без": "gent", "для": "gent",
    "с": "ablt", "со": "ablt", "за": "ablt", "под": "ablt",
    "перед": "ablt", "над": "ablt",
    "в": "loct", "на": "loct",
    "через": "accs", "про": "accs",
}

DEP_CASE = {
    "nsubj": "nomn",
    "nsubj:pass": "nomn",
    "obj": "accs",
    "iobj": "datv",
    "obl": "ablt",
    "nmod": "gent",
}


def _detect_gender_for_inflection(tokens: list[str]) -> str | None:
    for tag_priority in ({"Name", "Patr"}, {"Surn"}):
        for token in tokens:
            for p in morph.parse(token):
                if tag_priority & set(p.tag.grammemes):
                    if "femn" in p.tag.grammemes:
                        return "femn"
                    if "masc" in p.tag.grammemes:
                        return "masc"
    return None


def _inflect_token(token: str, target_case: str, gender: str | None) -> str:
    parses = morph.parse(token)
    best = None
    for p in parses:
        if gender and gender not in p.tag.grammemes:
            continue
        if {"Name", "Patr", "Surn"} & set(p.tag.grammemes):
            best = p
            break
    if best is None:
        for p in parses:
            if gender and gender not in p.tag.grammemes:
                continue
            best = p
            break
    if best is None:
        best = parses[0]
    inflected = best.inflect({target_case})
    return inflected.word.capitalize() if inflected else token


def _inflect_name(name_tokens: list[str], target_case: str) -> str:
    gender = _detect_gender_for_inflection(name_tokens)
    return " ".join(_inflect_token(tok, target_case, gender) for tok in name_tokens)


def _is_personal_name(token: str) -> bool:
    """True if any pymorphy3 parse contains Surn/Name/Patr (personal name or patronymic)."""
    return any(
        {"Name", "Patr", "Surn"} & set(p.tag.grammemes) for p in morph.parse(token)
    )


def _is_frozen_genitive(token: str) -> bool:
    """True if the top pymorphy3 parse is already genitive.

    Street names like «пр. Мира», «ул. Победы» use a fixed genitive noun as
    their descriptor — it must stay in genitive regardless of sentence context.
    """
    parses = morph.parse(token)
    return bool(parses) and "gent" in parses[0].tag.grammemes


def _inflect_location_token(token: str, target_case: str) -> str:
    """Inflect a single location token via pymorphy3; digits, dots, dashes kept as-is."""
    if "." in token or "-" in token or _DIGITS_RE.search(token):
        return token
    parses = morph.parse(token)
    if not parses:
        return token
    inflected = parses[0].inflect({target_case})
    if not inflected:
        return token
    word = inflected.word
    return word.capitalize() if token[0].isupper() else word


def _fix_location_case(
    text: str, loc_span: tuple[int, int], model_name: str | None = None
) -> str:
    """
    Detect the grammatical case required by the syntactic context of a location
    span and inflect the location tokens accordingly.

    Freeze rules (token is kept as-is):
      PROPN or NOUN + _is_personal_name  → surname/name used as street label (Ленина, Пушкина)
      PROPN or NOUN + _is_frozen_genitive → fixed genitive descriptor (Мира, Победы)

    Inflect rules:
      Unfrozen PROPN/NOUN/ADJ → inflect to target_case (city names, street generic terms,
                                 adjectival names: Москва, улица, Садовая)

    Inter-token whitespace from spacy is preserved to avoid spurious spaces (e.g. «пр .»).
    """
    span_start, span_end = loc_span
    window, local_start, local_end = _extract_window(text, span_start, span_end)

    nlp = _get_nlp(model_name)
    doc = nlp(window)

    loc_tokens_in_doc = [
        tok for tok in doc
        if tok.idx >= local_start and tok.idx + len(tok.text) <= local_end
    ]
    if not loc_tokens_in_doc:
        return text

    first_tok = loc_tokens_in_doc[0]

    # 1. Preposition immediately before the location span
    prev_token = doc[first_tok.i - 1] if first_tok.i > 0 else None
    target_case = (
        PREP_CASE.get(prev_token.text.lower())
        if prev_token and prev_token.pos_ == "ADP"
        else None
    )

    # 2. appos: animate head → head's case; inanimate head → genitive
    if not target_case and first_tok.dep_ == "appos":
        head_feats = first_tok.head.morph.to_dict()
        if first_tok.head.morph.get("Animacy") == ["Anim"]:
            spacy_case = head_feats.get("Case", "Nom")
            target_case = CASE_MAP.get(spacy_case, "nomn")
        else:
            target_case = "gent"

    # 3. Dependency relation
    if not target_case:
        target_case = DEP_CASE.get(first_tok.dep_)

    # 4. Fallback: head's case
    if not target_case:
        head_feats = first_tok.head.morph.to_dict()
        spacy_case = head_feats.get("Case", "Nom")
        target_case = CASE_MAP.get(spacy_case, "nomn")

    # Reconstruct with inflection driven by spacy POS — no vocabulary lookup
    parts: list[str] = []
    for i, tok in enumerate(loc_tokens_in_doc):
        freeze = tok.pos_ in ("PROPN", "NOUN") and (
            _is_personal_name(tok.text) or _is_frozen_genitive(tok.text)
        )
        if freeze:
            inflected = tok.text
        elif tok.pos_ in ("PROPN", "NOUN", "ADJ"):
            inflected = _inflect_location_token(tok.text, target_case)
        else:
            inflected = tok.text
        parts.append(inflected)
        if i < len(loc_tokens_in_doc) - 1:
            parts.append(tok.whitespace_)

    return text[:span_start] + "".join(parts) + text[span_end:]


_CONTEXT_WINDOW = 150  # characters to take on each side of the name span


def _extract_window(text: str, span_start: int, span_end: int) -> tuple[str, int, int]:
    """
    Cut a fixed-size character window around the name span.
    Returns (window_text, local_start, local_end) with indices relative to the window.
    """
    win_start = max(0, span_start - _CONTEXT_WINDOW)
    win_end = min(len(text), span_end + _CONTEXT_WINDOW)
    window = text[win_start:win_end]
    return window, span_start - win_start, span_end - win_start


def _fix_name_case(
    text: str, name_span: tuple[int, int], model_name: str | None = None
) -> str:
    """
    Extract the sentence containing the name, detect the grammatical case
    required by its syntactic context, and inflect the name accordingly.
    Returns the full text with the name replaced in the correct case.
    """
    span_start, span_end = name_span
    window, local_start, local_end = _extract_window(text, span_start, span_end)

    nlp = _get_nlp(model_name)
    doc = nlp(window)

    name_tokens_in_doc = [
        tok for tok in doc
        if tok.idx >= local_start and tok.idx + len(tok.text) <= local_end
    ]
    if not name_tokens_in_doc:
        return text

    first_tok = name_tokens_in_doc[0]

    # 1. Preposition immediately before the name
    prev_token = doc[first_tok.i - 1] if first_tok.i > 0 else None
    target_case = (
        PREP_CASE.get(prev_token.text.lower())
        if prev_token and prev_token.pos_ == "ADP"
        else None
    )

    # 2. appos: animate head → head's case; inanimate head → genitive
    if not target_case and first_tok.dep_ == "appos":
        head_feats = first_tok.head.morph.to_dict()
        if first_tok.head.morph.get("Animacy") == ["Anim"]:
            spacy_case = head_feats.get("Case", "Nom")
            target_case = CASE_MAP.get(spacy_case, "nomn")
        else:
            target_case = "gent"

    # 3. Dependency relation
    if not target_case:
        dep = first_tok.dep_
        if dep == "nsubj":
            # If the head verb already has another nsubj, this name is likely
            # the direct object (spacy misparses nominative-form names as subjects).
            head_has_other_nsubj = any(
                t.dep_ == "nsubj" and t.i not in {tok.i for tok in name_tokens_in_doc}
                for t in first_tok.head.children
            )
            target_case = "accs" if head_has_other_nsubj else "nomn"
        else:
            target_case = DEP_CASE.get(dep)

    # 4. Fallback: head's case
    if not target_case:
        head_feats = first_tok.head.morph.to_dict()
        spacy_case = head_feats.get("Case", "Nom")
        target_case = CASE_MAP.get(spacy_case, "nomn")

    original_tokens = [tok.text for tok in name_tokens_in_doc]
    inflected = _inflect_name(original_tokens, target_case)
    return text[:span_start] + inflected + text[span_end:]


def find_tag_span(
    canonical_tag: str,
    text: str,
    threshold: int = 80,
) -> tuple[int, int] | None:
    """
    Find where *canonical_tag* appears in *text* using regex + XML parsing +
    fuzzy attribute matching.  Handles typos, reordered/extra attributes, and
    multi-line tags produced by the LLM.

    Returns ``(start, end)`` indices or ``None`` if not found.
    """
    try:
        local_elem = ET.fromstring(canonical_tag.strip())
    except ET.ParseError:
        return None

    tag_name = local_elem.tag
    local_attrs = local_elem.attrib
    # [^>]* matches any char except > including newlines — covers multi-line tags
    pattern = rf"<{tag_name}\b[^>]*/>"

    for match in re.finditer(pattern, text, re.IGNORECASE):
        gen_tag_str = match.group(0)

        # Normalise non-self-closing variants just in case
        stripped = gen_tag_str.strip()
        if not stripped.endswith("/>"):
            stripped = stripped.rstrip(">").rstrip() + " />"

        # LLM sometimes escapes inner quotes when tag is inside a double-quoted string
        stripped = stripped.replace('\\"', '"')

        try:
            gen_elem = ET.fromstring(stripped)
        except ET.ParseError:
            continue

        gen_attrs = gen_elem.attrib
        is_match = True
        for key, local_val in local_attrs.items():
            if key not in gen_attrs:
                is_match = False
                break
            gen_val = gen_attrs[key]
            # `type` and `id` decide *whose* value is restored, so they match
            # exactly. Fuzzily they did not: `fuzz.ratio("111", "11")` is exactly
            # 80, so entity 111 matched entity 11 and put a different person's
            # name in their slot. `gender` stays fuzzy — cosmetic, the restored
            # form re-derives it from the tokens.
            if key in ("type", "id"):
                if local_val.strip().lower() != gen_val.strip().lower():
                    is_match = False
                    break
                continue
            if fuzz.ratio(local_val.lower(), gen_val.lower()) < threshold:
                is_match = False
                break

        if is_match:
            return match.span()

    return None


def deanonymize(
    text: str, mapping: dict[str, str], model_name: str | None = None
) -> str:
    """
    Replace XML PII tags in *text* with original values.

    - PERSON:   the mapping holds the nominative, which is inserted and then
                inflected to the case its syntactic context requires
                (spacy + pymorphy3).
    - LOCATION: the mapping holds the **surface form as written**, not the
                nominative -- normalising place names costs more than it buys.
                It is inserted as stored and then inflected via
                ``_fix_location_case`` (same context logic, without the
                Name/Patr/Surn filter), so «Москве» in, «Москве» or «Москву»
                out depending on the new context.
    - All others: direct substitution.
    - Tags remaining after all substitutions (unknown / hallucinated) → ``****``.
    """
    result = text

    # One budget for the whole call: a legitimate mapping needs one substitution
    # per tag of the input. Per-key bounds let a chain whose values re-emit the
    # next key's tag double the count on every key -- 12 keys in ~1 KB cost 41 s,
    # and `mapping` is unauthenticated input processed in an uncancellable thread
    # holding the inference slot. Counted case-insensitively like `find_tag_span`,
    # or a `<pii .../>` tag would be neither restored nor scrubbed.
    budget = len(_TAG_OPEN_RE.findall(result))

    for canonical_tag, original_nom in mapping.items():
        if budget <= 0:
            break
        try:
            elem = ET.fromstring(canonical_tag.strip())
            entity_type = elem.attrib.get("type", "").upper()
        except ET.ParseError:
            entity_type = ""

        is_person = entity_type == "PERSON"
        is_location = entity_type == "LOCATION"

        while budget > 0:
            span = find_tag_span(canonical_tag, result)
            if span is None:
                break
            # Charged per substitution actually made. Charging per iteration
            # would spend the budget on the failed lookup that ends every key's
            # loop, leaving later keys unrestored.
            budget -= 1
            start, end = span

            if is_person:
                before = result[:start]
                after = result[end:]
                sentence_with_nom = before + original_nom + after
                name_span = (start, start + len(original_nom))
                result = _fix_name_case(sentence_with_nom, name_span, model_name)
            elif is_location:
                before = result[:start]
                after = result[end:]
                sentence_with_nom = before + original_nom + after
                loc_span = (start, start + len(original_nom))
                result = _fix_location_case(sentence_with_nom, loc_span, model_name)
            else:
                result = result[:start] + original_nom + result[end:]

    # Scrub any leftover PII tags that weren't in the mapping. Case-insensitive
    # for the same reason as the budget above.
    result = _TAG_FULL_RE.sub("****", result)
    return result


def deanonymize_batch(
    texts: list[str], mapping: dict[str, str], model_name: str | None = None
) -> list[str]:
    """Deanonymize several texts using the same mapping.

    *model_name* pins the spaCy pipeline for this call. `Anonymizer` passes its
    own configured model, so two instances with different pipelines no longer
    interfere: the module-level name is only the fallback for direct callers.
    """
    return [deanonymize(t, mapping, model_name) for t in texts]
