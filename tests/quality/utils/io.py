"""Вспомогательные операции ввода-вывода для quality gate."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path


def atomic_write_text(path: Path, payload: str) -> None:
    """Атомарно записывает текст в файл через временный файл в той же директории."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = handle.name
        Path(temp_path).replace(path)
    finally:
        if temp_path and Path(temp_path).exists():
            with suppress(OSError):
                Path(temp_path).unlink()
