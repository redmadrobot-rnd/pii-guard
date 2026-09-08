"""Single source of truth for the model artefacts this project defaults to.

Both the runtime and the Dockerfile read from here, so a model bump is a one-line
change with one place to review.

``DEFAULT_NER_REVISION`` pins a commit sha on the Hub. Without a pin, re-uploading
the model silently changes behaviour for everyone whose cache is cold while the
code stays byte-identical. Tags are not a substitute -- they can be moved, a sha
cannot. Bumping either constant is a pull request that must carry quality-gate
metrics before and after.
"""

from __future__ import annotations

DEFAULT_NER_MODEL = "redmadrobot-rnd/rubert-base-pii-ner"

# Pin this to a commit sha to make runs reproducible. ``None`` resolves to
# whatever ``main`` currently points at, so a re-upload of the weights silently
# changes results; the recognizer logs a warning when it loads unpinned. Set it
# via ``PII_GUARD_NER_REVISION`` for a deployment, or edit it here when cutting a
# release against published weights.
DEFAULT_NER_REVISION: str | None = None

# spaCy pipeline backing the Presidio pattern recognizers (tokenisation, lemmas)
# and the grammatical-case restoration in pseudonymize.py (POS, dependencies).
# Installed with `python -m spacy download`, never redistributed by this project.
DEFAULT_SPACY_MODEL = "ru_core_news_sm"

__all__ = [
    "DEFAULT_NER_MODEL",
    "DEFAULT_NER_REVISION",
    "DEFAULT_SPACY_MODEL",
]
