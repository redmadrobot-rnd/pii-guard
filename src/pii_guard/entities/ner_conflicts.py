"""
Conflict rules for NER-produced entities (URL, EMAIL_ADDRESS).

These entities are detected by the Transformer NER recognizer (branch B),
not by entity classifiers in this package.  This file declares their
cross-entity conflict rules so the framework can resolve them
generically.
"""

from __future__ import annotations

from pii_guard.framework.base import (
    register_conflict_handler,
    register_pairwise_conflict,
)

# ── Pairwise rules ─────────────────────────────────────────────

register_pairwise_conflict(winner="EMAIL_ADDRESS", loser="URL")


# ── Custom handler: URL as domain part of an email ─────────────

@register_conflict_handler
def _handle_url_email_adjacency(entities, text, already_removed):
    """Remove URL when it's the domain part of an email address.

    Catches cases where URL and EMAIL_ADDRESS don't formally overlap
    but the URL is adjacent to an ``@`` sign or to an EMAIL_ADDRESS
    entity.
    """
    extra: set[int] = set()

    for i, e in enumerate(entities):
        if i in already_removed or e.entity_type != "URL":
            continue

        # Check for "@" sign shortly before the URL
        url_start = e.start
        search_start = max(0, url_start - 100)
        before = text[search_start:url_start]

        if "@" in before:
            at_pos = before.rfind("@") + search_start
            gap = text[at_pos + 1 : url_start]
            if len(gap) == 0 or all(c.isalnum() or c in "._-+" for c in gap):
                extra.add(i)
                continue

        # Check adjacency with EMAIL_ADDRESS entities
        for j, e2 in enumerate(entities):
            if j != i and j not in already_removed and e2.entity_type == "EMAIL_ADDRESS":
                if e.start == e2.end:
                    extra.add(i)
                    break
                if e.start > e2.end:
                    gap = text[e2.end : e.start]
                    if len(gap.strip()) == 0:
                        extra.add(i)
                        break

    return extra
