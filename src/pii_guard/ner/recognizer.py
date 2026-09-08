from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from presidio_analyzer import EntityRecognizer, RecognizerResult
from transformers import AutoModelForTokenClassification, AutoTokenizer

logger = logging.getLogger("pii_guard.ner")


class ModelUnavailableError(RuntimeError):
    """Raised when the NER weights can be neither found locally nor fetched.

    Carries the four ways out, because the bare ``OSError`` transformers raises
    tells the user nothing about which of them applies to their setup.
    """

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        pin = f" --revision {revision}" if revision else ""
        super().__init__(
            f"NER model {model_name!r} is unavailable. Options:\n"
            f"  1) allow network access -- the model is downloaded automatically\n"
            f"  2) warm the cache: huggingface-cli download {model_name}{pin}\n"
            f"  3) point at a local copy: PII_GUARD_NER_MODEL=/path/to/model\n"
            f"  4) run without NER: PII_GUARD_NER_DISABLED=1 "
            f"(rules branch only -- the NER-only types yield nothing)"
        )
        self.model_name = model_name
        self.revision = revision


def resolve_device(requested: str | None = None) -> str:
    """Resolve the torch device for NER inference.

    ``auto`` (the default) mirrors the behaviour this code has always had: CUDA
    when a GPU is visible, CPU otherwise. Apple ``mps`` is **not** auto-selected
    -- it is only used when asked for explicitly, so that a macOS run stays
    numerically comparable to a Linux CPU run and to the quality-gate baseline.

    Override with ``PII_GUARD_DEVICE=cpu|cuda|mps``. An explicit value is passed
    to torch as-is, so an unavailable device fails loudly instead of silently
    degrading -- if you asked for a GPU you want to know it is missing.
    """
    choice = (requested or os.getenv("PII_GUARD_DEVICE") or "auto").strip().lower()
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"

_SPLIT_RE = re.compile(
    r'([(){}\[\]«»"""“”‘’\'",;:!?])'
)

# Matches non-whitespace runs — used to find word positions in text
_WORD_RE = re.compile(r'\S+')

# Punctuation that can appear between two parts of the same entity
_BRIDGE_PUNCT = set(".,;:!?")

# NER labels for which punctuation bridging is disabled (names and locations
# should not be merged across commas/colons)
_NO_BRIDGE_ENTITIES = {
    "FIRST_NAME", "LAST_NAME", "MIDDLE_NAME",
    "CITY", "COUNTRY", "DISTRICT", "REGION", "STREET", "HOUSE",
}


def split_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split *text* on whitespace, returning ``(word, start, end)`` triples.

    Offsets are character positions in the original *text*.
    """
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def retokenize_with_offsets(
    words: list[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    """Split each word into sub-tokens, preserving character offsets.

    Splitting rules:
    - brackets, quotes, commas, colons, semicolons, and
      exclamation/question marks are split into their own tokens;
    - leading dots are split off;
    - trailing dots are split off (dots inside abbreviations like
      'ул.', 'д.' are left intact because the model sees them as one token).

    Each returned tuple is ``(sub_token, abs_start, abs_end)`` where
    *abs_start* / *abs_end* are character offsets in the original text.
    """
    result: list[tuple[str, int, int]] = []

    for word, word_start, _ in words:
        # _SPLIT_RE uses a capturing group, so split() includes delimiters.
        # All parts are contiguous sub-strings of `word`; we track our
        # position inside `word` with `local_offset`.
        parts = _SPLIT_RE.split(word)
        local_offset = 0

        for part in parts:
            # `re.split` with a capturing group yields empty strings around
            # adjacent delimiters. They advance the cursor by zero, so skipping
            # them outright is equivalent.
            if not part:
                continue

            p = part
            p_local = local_offset  # cursor to the start of `p` inside `word`

            # Strip leading dots and record each one with its absolute offset.
            lead: list[tuple[str, int, int]] = []
            while len(p) > 1 and p[0] == ".":
                abs_s = word_start + p_local
                lead.append((".", abs_s, abs_s + 1))
                p = p[1:]
                p_local += 1

            # Strip trailing dots; collected in reverse order so that after
            # reversed() they appear in left-to-right position order.
            trail: list[tuple[str, int, int]] = []
            while len(p) > 1 and p[-1] == ".":
                abs_s = word_start + p_local + len(p) - 1
                trail.append((".", abs_s, abs_s + 1))
                p = p[:-1]

            result.extend(lead)
            if p:
                abs_s = word_start + p_local
                result.append((p, abs_s, abs_s + len(p)))
            result.extend(reversed(trail))

            local_offset += len(part)

    return result


@dataclass(slots=True)
class TransformerNERConfig:
    """Inference settings for :class:`TransformerNERRecognizer`.

    *model_name* is either a Hugging Face repo id or a local directory.
    *revision* pins a commit sha on the Hub; leaving it ``None`` means "whatever
    ``main`` currently points at", which makes runs irreproducible -- always pin
    it for the model shipped with this project.
    *device* overrides :func:`resolve_device`; ``None`` means resolve it.
    """

    model_name: str
    revision: str | None = None
    max_length: int = 512
    stride: int = 128
    score: float = 0.70
    supported_language: str = "ru"
    device: str | None = None


class TransformerNERRecognizer(EntityRecognizer):
    def __init__(
        self,
        config: TransformerNERConfig,
        entity_mapping: dict[str, str],
        name: str = "TransformerNERRecognizer",
        supported_entities: Iterable[str] | None = None,
    ) -> None:
        self.config = config
        self.entity_mapping = entity_mapping

        if supported_entities is None:
            supported_entities = sorted(set(entity_mapping.values()))

        super().__init__(
            supported_entities=list(supported_entities),
            name=name,
            supported_language=config.supported_language,
        )

        self._device = resolve_device(config.device)
        logger.info(
            "event=ner_model_load model=%s revision=%s device=%s "
            "(first run downloads ~700MB into the Hugging Face cache)",
            config.model_name, config.revision or "unpinned", self._device,
        )
        if not config.revision and not os.path.isdir(config.model_name):
            # A Hub id without a sha resolves to whatever `main` points at now, so
            # results can change under a re-upload with no local change at all.
            # Local directories are exempt: the path *is* the version.
            logger.warning(
                "event=ner_revision_unpinned model=%s "
                "(results are not reproducible; set PII_GUARD_NER_REVISION)",
                config.model_name,
            )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, revision=config.revision
            )
            self._model = AutoModelForTokenClassification.from_pretrained(
                config.model_name, revision=config.revision
            ).to(self._device)
        except OSError as exc:
            raise ModelUnavailableError(config.model_name, config.revision) from exc
        self._model.eval()
        self._id2label = {
            int(k): v for k, v in self._model.config.id2label.items()
        }

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts=None
    ) -> list[RecognizerResult]:
        if not text:
            return []

        tokens_with_offsets = retokenize_with_offsets(split_with_offsets(text))
        if not tokens_with_offsets:
            return []

        word_tokens = [tok for tok, _, _ in tokens_with_offsets]
        offsets = [(s, e) for _, s, e in tokens_with_offsets]

        # Run model with sliding window
        tags = self._sliding_window_predict(word_tokens)

        # Bridge punctuation gaps between same-type entities (except names/locations)
        tags = self._bridge_punctuation(word_tokens, tags)

        # Map word tokens + tags back to char offsets and RecognizerResults
        return self._build_results(offsets, tags, entities)

    def _count_subwords(self, word_tokens: list[str]) -> list[int]:
        """Return the number of subword tokens each word produces."""
        encoding = self._tokenizer(
            word_tokens,
            is_split_into_words=True,
            add_special_tokens=False,
        )
        word_ids = encoding.word_ids(batch_index=0)
        counts = [0] * len(word_tokens)
        for wid in word_ids:
            if wid is not None:
                counts[wid] += 1
        return counts

    def _predict_window_words(self, word_tokens: list[str]) -> list[str]:
        """Run model on a window of word tokens. Returns one tag per word."""
        encoding = self._tokenizer(
            word_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**encoding).logits

        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        word_ids = encoding.word_ids(batch_index=0)

        preds: list[str] = []
        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                preds.append(self._id2label[pred_ids[idx]])
            prev_word_id = word_id

        while len(preds) < len(word_tokens):
            preds.append("O")
        return preds[: len(word_tokens)]

    def _sliding_window_predict(self, word_tokens: list[str]) -> list[str]:
        """Predict tags for all word tokens using sliding window."""
        subword_counts = self._count_subwords(word_tokens)
        total_subwords = sum(subword_counts)
        window_budget = self.config.max_length - 2  # space for [CLS] and [SEP]
        stride = self.config.stride

        # Fits in one window — simple path
        if total_subwords <= window_budget:
            return self._predict_window_words(word_tokens)

        # Sliding window over word tokens
        preds: list[str | None] = [None] * len(word_tokens)

        # First word no window has written yet. The margin below skips a window's
        # left edge assuming the previous centre covered it -- false right after an
        # oversized-token skip, where nothing did: those words stayed None and were
        # forced to "O", so PII right after a 600-underscore form blank vanished.
        next_unwritten = 0

        start_word = 0
        while start_word < len(word_tokens):
            # A single word whose subword count exceeds the model window
            # (e.g. a long run of underscores from a fillable form blank)
            # cannot carry a meaningful label. Tag it "O" and skip — otherwise
            # the step-forward logic below cannot make progress and the loop
            # spins forever.
            if subword_counts[start_word] > window_budget:
                preds[start_word] = "O"
                start_word += 1
                continue

            # Greedily fill the window with as many words as fit
            sw_count = 0
            end_word = start_word
            while (
                end_word < len(word_tokens)
                and sw_count + subword_counts[end_word] <= window_budget
            ):
                sw_count += subword_counts[end_word]
                end_word += 1

            # An oversized word ahead cannot be pulled into any window, so this
            # window is the last one before it: nothing to the right will own the
            # tail, and creeping toward it one word at a time costs one full model
            # pass per word (130 windows for 374 words, measured).
            blocked_ahead = (
                end_word < len(word_tokens)
                and subword_counts[end_word] > window_budget
            )

            window_tokens = word_tokens[start_word:end_word]
            window_preds = self._predict_window_words(window_tokens)

            # Each window is responsible only for its center.
            # Skip the first stride//2 subwords (left context, owned by the
            # previous window's center) and the last stride//2 subwords (right
            # context, will be owned by the next window's center).
            # The last window writes all the way to its end since there is no
            # next window.
            is_last = end_word >= len(word_tokens)
            owns_tail = is_last or blocked_ahead
            margin = stride // 2

            if start_word == 0:
                write_from = 0
            else:
                sw_overlap = 0
                write_from = 0
                for w in range(len(window_tokens)):
                    sw_overlap += subword_counts[start_word + w]
                    if sw_overlap >= margin:
                        write_from = w + 1
                        break

            if owns_tail:
                write_to = len(window_preds)
            else:
                sw_tail = 0
                write_to = len(window_tokens)
                for w in range(len(window_tokens) - 1, -1, -1):
                    sw_tail += subword_counts[start_word + w]
                    if sw_tail >= margin:
                        write_to = w + 1
                        break

            # Never skip words that no earlier window reached.
            write_from = max(0, min(write_from, next_unwritten - start_word))

            for w in range(write_from, write_to):
                word_idx = start_word + w
                if word_idx < len(preds):
                    preds[word_idx] = window_preds[w]
            next_unwritten = max(next_unwritten, start_word + write_to)

            if is_last:
                break

            if blocked_ahead:
                # Hand over to the oversized-token branch instead of creeping.
                start_word = end_word
                continue

            # Step forward
            sw_back = 0
            overlap_words = 0
            for w in range(end_word - 1, start_word - 1, -1):
                sw_back += subword_counts[w]
                overlap_words += 1
                if sw_back >= stride:
                    break
            # Guarantee forward progress: if the window was so narrow that
            # the back-scan covered all of it, advance past it anyway.
            start_word = max(end_word - overlap_words, start_word + 1)

        return [p if p is not None else "O" for p in preds]

    @staticmethod
    def _bridge_punctuation(
        tokens: list[str], tags: list[str]
    ) -> list[str]:
        """Fill O-tagged punctuation between two same-entity spans.

        Bridging is skipped for entities in _NO_BRIDGE_ENTITIES (names,
        locations) to avoid merging distinct people or places through commas.
        """
        tags = list(tags)
        for i in range(1, len(tags) - 1):
            if tags[i] != "O":
                continue
            if not all(c in _BRIDGE_PUNCT for c in tokens[i]):
                continue
            prev_tag, next_tag = tags[i - 1], tags[i + 1]
            if prev_tag == "O" or next_tag == "O":
                continue
            prev_ent = prev_tag.split("-", 1)[1] if "-" in prev_tag else None
            next_ent = next_tag.split("-", 1)[1] if "-" in next_tag else None
            if prev_ent and prev_ent == next_ent and prev_ent not in _NO_BRIDGE_ENTITIES:
                tags[i] = f"I-{prev_ent}"
        return tags

    def _build_results(
        self,
        offsets: list[tuple[int, int]],
        tags: list[str],
        requested_entities: list[str],
    ) -> list[RecognizerResult]:
        """Convert word-level tags to char-offset RecognizerResults."""

        # Collect contiguous entity spans
        entities: list[dict] = []
        current: dict | None = None

        # A run of same-type tags is one span, `B-` or `I-` alike. Deviates from
        # IOB2 on purpose: the model restarts with `B-` between a label and its
        # value («Полис ОМС: 9988776655443325» → B-OMS I-OMS B-OMS), and honouring
        # that splits the label off as its own entity. Strict B/I costs 40 false
        # positives and 0.0024 micro-F1 on the quality set, nine categories down
        # and none up. Adjacent people are separated by component type here and
        # bounded downstream by `pseudonymize.MAX_NAME_PARTS`.
        for i, tag in enumerate(tags):
            if tag.startswith("B-") or tag.startswith("I-"):
                ent_type = tag[2:]
                if current and current["type"] == ent_type:
                    current["end"] = offsets[i][1]
                else:
                    if current:
                        entities.append(current)
                    current = {
                        "type": ent_type,
                        "start": offsets[i][0],
                        "end": offsets[i][1],
                    }
            else:
                if current:
                    entities.append(current)
                    current = None

        if current:
            entities.append(current)

        # Map to RecognizerResult with entity_mapping
        results: list[RecognizerResult] = []
        for ent in entities:
            mapped = self.entity_mapping.get(ent["type"])
            if not mapped or mapped not in requested_entities:
                continue
            results.append(
                RecognizerResult(
                    entity_type=mapped,
                    start=ent["start"],
                    end=ent["end"],
                    score=self.config.score,
                )
            )

        return results
