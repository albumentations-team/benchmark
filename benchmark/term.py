"""TTY colors and colored logging (respects NO_COLOR / FORCE_COLOR)."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",
}
_RESET = "\033[0m"


def use_color() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    try:
        return sys.stderr.isatty()
    except ValueError:
        return False


class _ColoredFormatter(logging.Formatter):
    def __init__(self, fmt: str, *, color: bool) -> None:
        super().__init__(fmt, datefmt="%H:%M:%S")
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self._color:
            return text
        prefix = _LEVEL_COLORS.get(record.levelno, "")
        if not prefix:
            return text
        return f"{prefix}{text}{_RESET}"


def configure_logging(level: int = logging.INFO, fmt: str | None = None) -> None:
    """Configure root logger once: stderr, optional ANSI level colors."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    use_c = use_color()
    pattern = fmt or "%(asctime)s %(levelname)s %(message)s"
    handler.setFormatter(_ColoredFormatter(pattern, color=use_c))
    root.addHandler(handler)


def tqdm_kwargs() -> dict[str, Any]:
    """Extra kwargs for tqdm when colors are enabled."""
    if use_color():
        return {"colour": "cyan"}
    return {}
