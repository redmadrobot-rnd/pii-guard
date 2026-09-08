"""
Entity classifier registry — auto-discovers all built-in classifiers.

Every ``.py`` file in this package that decorates a
:class:`~pii_guard.framework.base.BaseEntityClassifier` subclass with
:func:`~pii_guard.framework.base.register_entity` is picked up
automatically.  Regex entities using
:func:`~pii_guard.framework.base.register_regex_entity` are also
discovered.

To add a new entity type
------------------------
1. Create ``pii_guard/entities/my_entity.py``
2. Decorate the classifier with ``@register_entity`` (numeric) or
   ``@register_regex_entity`` (regex).
3. Done — no other file needs changing.
"""

from __future__ import annotations

from pii_guard.framework.base import auto_discover_entities

# ── Auto-discover all entity modules in this package ────────────
auto_discover_entities(__path__, __name__)
