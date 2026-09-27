"""Setup logging: ringkas di terminal, lengkap di file."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_CONSOLE_FMT = "%(asctime)s %(levelname).1s [%(name)s] %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-7s [%(name)s:%(lineno)d] %(message)s"
_DATE_FMT = "%H:%M:%S"

_LEVEL_COLOR = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"

_configured = False


class _ColorFormatter(logging.Formatter):
    """Formatter dengan warna, otomatis mati kalau output bukan terminal."""

    def __init__(self, fmt: str, datefmt: str, use_color: bool) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        text = super().format(record)
        if not self.use_color:
            return text
        color = _LEVEL_COLOR.get(record.levelname)
        return f"{color}{text}{_RESET}" if color else text


def _color_supported(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | os.PathLike[str]] = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Pasang handler root sekali saja lalu kembalikan logger ``tukang_sayur``."""

    global _configured
    root = logging.getLogger()
    if _configured and not force:
        return logging.getLogger("tukang_sayur")

    for handler in list(root.handlers):
        root.removeHandler(handler)

    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(min(numeric, logging.INFO))

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(numeric)
    console.setFormatter(
        _ColorFormatter(_CONSOLE_FMT, _DATE_FMT, _color_supported(sys.stderr))
    )
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(_FILE_FMT))
            root.addHandler(file_handler)
        except OSError as exc:  # direktori read-only, dsb. -> jangan bikin crash
            root.warning("Tidak bisa menulis log ke %s: %s", path, exc)

    # yt-dlp & urllib3 terlalu berisik pada level DEBUG
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _configured = True
    return logging.getLogger("tukang_sayur")


def get_logger(name: str) -> logging.Logger:
    """Logger anak dengan prefiks ``tukang_sayur.``."""

    if name.startswith("tukang_sayur"):
        return logging.getLogger(name)
    return logging.getLogger(f"tukang_sayur.{name}")
