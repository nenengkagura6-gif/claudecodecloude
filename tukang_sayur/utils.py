"""Perkakas umum: jalankan proses, ffprobe, hash cache, format waktu, wrap teks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .errors import DependencyError, TukangSayurError
from .logging_utils import get_logger

log = get_logger("utils")

_TIMECODE_RE = re.compile(
    r"^\s*(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:[.,]\d+)?)\s*$"
)
_HUMAN_RE = re.compile(
    r"^\s*(?:(?P<h>\d+(?:\.\d+)?)\s*h)?\s*(?:(?P<m>\d+(?:\.\d+)?)\s*m(?!s))?"
    r"\s*(?:(?P<s>\d+(?:\.\d+)?)\s*s)?\s*$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Proses luar
# --------------------------------------------------------------------------
def which(name: str) -> Optional[str]:
    return shutil.which(name)


def ffmpeg_bin() -> str:
    """Path ffmpeg. Bisa ditimpa dengan env ``TUKANG_SAYUR_FFMPEG``."""

    override = os.environ.get("TUKANG_SAYUR_FFMPEG")
    if override:
        return override
    found = which("ffmpeg")
    if found:
        return found
    raise DependencyError(
        "ffmpeg tidak ditemukan di PATH. Windows: unduh dari "
        "https://www.gyan.dev/ffmpeg/builds/ lalu tambahkan folder bin ke PATH. "
        "Linux: sudo apt install ffmpeg"
    )


def ffprobe_bin() -> str:
    override = os.environ.get("TUKANG_SAYUR_FFPROBE")
    if override:
        return override
    found = which("ffprobe")
    if found:
        return found
    raise DependencyError(
        "ffprobe tidak ditemukan di PATH (biasanya satu paket dengan ffmpeg)."
    )


def has_ffmpeg() -> bool:
    try:
        ffmpeg_bin()
        ffprobe_bin()
        return True
    except DependencyError:
        return False


def run_command(
    cmd: Sequence[str],
    *,
    timeout: Optional[float] = None,
    check: bool = True,
    capture: bool = True,
    cwd: Optional[str | os.PathLike[str]] = None,
    error_cls: type[TukangSayurError] = TukangSayurError,
    label: str = "",
) -> subprocess.CompletedProcess:
    """Jalankan proses luar dengan log rapi dan pesan error yang informatif."""

    printable = " ".join(str(c) for c in cmd)
    log.debug("Menjalankan%s: %s", f" [{label}]" if label else "", printable)
    try:
        proc = subprocess.run(  # noqa: S603
            [str(c) for c in cmd],
            timeout=timeout,
            check=False,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise DependencyError(f"Program tidak ditemukan: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise error_cls(
            f"Proses kelamaan (> {timeout:.0f}s) lalu dihentikan: {printable}"
        ) from exc

    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
        detail = "\n".join(tail)
        raise error_cls(
            f"Perintah gagal (kode {proc.returncode}): {printable}\n{detail}"
        )
    return proc


# --------------------------------------------------------------------------
# Informasi media
# --------------------------------------------------------------------------
@dataclass
class MediaInfo:
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    audio_channels: int = 0
    audio_sample_rate: int = 0
    video_codec: str = ""
    audio_codec: str = ""

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 0.0


def _parse_fraction(value: Any) -> float:
    text = str(value or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            denominator = float(den)
            return float(num) / denominator if denominator else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def probe_media(path: str | os.PathLike[str]) -> MediaInfo:
    """Baca metadata media dengan ffprobe."""

    target = Path(path)
    if not target.is_file():
        raise TukangSayurError(f"File media tidak ada: {target}")

    proc = run_command(
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(target),
        ],
        label="ffprobe",
        timeout=120,
    )
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise TukangSayurError(f"Output ffprobe tidak bisa dibaca: {exc}") from exc

    info = MediaInfo(path=str(target))
    fmt = data.get("format") or {}
    try:
        info.duration = float(fmt.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        info.duration = 0.0

    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video" and not info.width:
            info.width = int(stream.get("width") or 0)
            info.height = int(stream.get("height") or 0)
            info.video_codec = str(stream.get("codec_name") or "")
            info.fps = _parse_fraction(
                stream.get("avg_frame_rate") or stream.get("r_frame_rate")
            )
            if not info.duration:
                try:
                    info.duration = float(stream.get("duration") or 0.0)
                except (TypeError, ValueError):
                    pass
        elif kind == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_codec = str(stream.get("codec_name") or "")
            info.audio_channels = int(stream.get("channels") or 0)
            try:
                info.audio_sample_rate = int(stream.get("sample_rate") or 0)
            except (TypeError, ValueError):
                info.audio_sample_rate = 0
    return info


# --------------------------------------------------------------------------
# Hash & cache
# --------------------------------------------------------------------------
def sha1_text(*parts: Any) -> str:
    digest = hashlib.sha1()  # noqa: S324 - cuma buat nama file cache
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


def file_fingerprint(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    """Sidik jari file yang murah: ukuran + potongan awal & akhir."""

    target = Path(path)
    stat = target.stat()
    digest = hashlib.sha1()  # noqa: S324
    digest.update(str(stat.st_size).encode())
    with target.open("rb") as handle:
        digest.update(handle.read(chunk))
        if stat.st_size > chunk * 2:
            handle.seek(-chunk, os.SEEK_END)
            digest.update(handle.read(chunk))
    return digest.hexdigest()


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Tulis file lewat file sementara supaya tidak ada file setengah jadi."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return target


def write_json(path: str | os.PathLike[str], data: Any) -> Path:
    return atomic_write_text(
        path, json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Teks & waktu
# --------------------------------------------------------------------------
def slugify(text: str, *, max_length: int = 60, default: str = "klip") -> str:
    """Ubah teks jadi nama file aman (ASCII, tanpa spasi)."""

    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    ascii_text = re.sub(r"-{2,}", "-", ascii_text)
    if not ascii_text:
        return default
    return ascii_text[:max_length].strip("-") or default


def human_duration(seconds: float) -> str:
    """``95`` -> ``01:35``; lebih dari sejam jadi ``1:02:03``."""

    total = max(0, int(round(float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float) -> str:
    """Format ``HH:MM:SS.mmm`` (dipakai di metadata & log)."""

    value = max(0.0, float(seconds))
    hours, rest = divmod(value, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def parse_timecode(value: Any) -> Optional[float]:
    """Terima 83, "83", "83.5s", "1:23", "00:01:23.5", "1m23s" -> detik.

    Balikan ``None`` kalau benar-benar tidak bisa dibaca (jangan diam-diam 0,
    karena 0 itu timestamp yang sah).
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number >= 0 else 0.0

    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".") if text.count(",") == 1 and ":" not in text else text

    match = _TIMECODE_RE.match(text)
    if match:
        hours = float(match.group("h") or 0)
        minutes = float(match.group("m") or 0)
        secs = float((match.group("s") or "0").replace(",", "."))
        return hours * 3600 + minutes * 60 + secs

    plain = text[:-1] if text.endswith("s") and text[:-1].replace(".", "", 1).isdigit() else text
    try:
        return max(0.0, float(plain))
    except ValueError:
        pass

    human = _HUMAN_RE.match(text)
    if human and any(human.group(g) for g in ("h", "m", "s")):
        return (
            float(human.group("h") or 0) * 3600
            + float(human.group("m") or 0) * 60
            + float(human.group("s") or 0)
        )
    return None


def wrap_text(text: str, max_chars: int) -> List[str]:
    """Bungkus teks per kata; kata super panjang dipotong paksa."""

    words = str(text or "").split()
    if not words:
        return []
    limit = max(4, int(max_chars))
    lines: List[str] = []
    current = ""
    for word in words:
        while len(word) > limit:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:limit])
            word = word[limit:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def truncate_words(text: str, max_words: int) -> str:
    """Batasi jumlah kata (dipakai untuk hook maks. 8 kata)."""

    words = str(text or "").split()
    if max_words <= 0 or len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def even(value: float) -> int:
    """Bulatkan ke bilangan bulat genap (banyak encoder butuh kelipatan 2)."""

    number = int(round(float(value)))
    return number - (number % 2)


def unique_path(path: str | os.PathLike[str]) -> Path:
    """Kalau file sudah ada, tambahkan ``-2``, ``-3``, dst."""

    target = Path(path)
    if not target.exists():
        return target
    stem, suffix, parent = target.stem, target.suffix, target.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def rmtree_quiet(path: str | os.PathLike[str]) -> None:
    shutil.rmtree(Path(path), ignore_errors=True)


def describe_dict(data: Dict[str, Any], keys: Sequence[str]) -> str:
    return ", ".join(f"{k}={data.get(k)}" for k in keys if k in data)
