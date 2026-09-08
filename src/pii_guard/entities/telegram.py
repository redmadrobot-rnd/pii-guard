"""TELEGRAM (телеграм-идентификатор) classifier.

Self-contained entity definition: keyword context, classification —
all in one file.

Поддерживается ТОЛЬКО форма ``@username``. Остальные формы
(``t.me/…``, числовой ``user_id``, голый ник без ``@``) намеренно
не распознаются: они дают много ложных срабатываний (числовой id
пересекается с телефонами/кодами, голый ник — с локальной частью
e-mail вида ``maria_k@…``), а реальные телеграм-контакты в данных
почти всегда пишутся как ``@username``.
"""

from __future__ import annotations

import re

from pii_guard.framework.base import BaseEntityClassifier, register_entity
from pii_guard.framework.context import nearest_before_distance

ENTITY_TYPE = "TELEGRAM"


# ── Keyword patterns ────────────────────────────────────────────

# Инстаграм-ники пишутся так же (``@username``), поэтому исключаем их по
# предшествующему контексту. Только живые формы слова: стем `инст\w*` ставил вето
# на «институт», «инструкция», «инстанция» — и настоящий ник рядом с ними
# оставался в открытом виде.
KW_INSTA = re.compile(
    r"(?iu)\bинст(?:а|е|у|ы|ой)\b|\bинстаг\b|\bинстаграм\w{0,4}"
    r"|\binsta\b|\binstagram\w{0,3}"
)


@register_entity
class TelegramClassifier(BaseEntityClassifier):
    """Telegram ``@username``.

    Скоринг: ``@username`` (если рядом перед ним нет инстаграм-контекста)
    → ``0.90``; иначе → ``None``.
    """

    entity_type = ENTITY_TYPE
    priority = 35
    candidate_patterns = [
        re.compile(r"(?<![\w@.])@[A-Za-z][A-Za-z0-9_]{4,31}\b"),  # @username
    ]

    def classify(
        self, raw: str, digits: str, ctx: str, wide_ctx: str | None,
        ctx_pos: int = 0, wide_ctx_pos: int = 0,
    ) -> tuple[str, float] | None:
        s = raw.strip()

        if s.startswith("@"):  # @username, минус инстаграм
            if nearest_before_distance(KW_INSTA, ctx, ctx_pos, max_dist=40) is not None:
                return None
            return self.entity_type, 0.90

        return None
