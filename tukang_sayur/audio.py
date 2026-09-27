"""Audio: potong jeda hening dan normalisasi loudness ke -14 LUFS.

Dua hal yang perlu diingat:

* Memotong hening **menggeser** semua timestamp sesudahnya. Karena itu ada
  :func:`remap_words` - kalau lupa memakainya, subtitle jadi tidak sinkron.
* Loudness -14 LUFS adalah target yang dipakai banyak platform. ffmpeg bisa
  sekali jalan, tapi dua tahap (ukur dulu, baru terapkan) hasilnya lebih rapi.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Config
from .errors import RenderError
from .logging_utils import get_logger
from .models import Word
from .utils import ffmpeg_bin, run_command

log = get_logger("audio")

Range = Tuple[float, float]

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")
_LOUDNORM_KEYS = (
    "input_i", "input_tp", "input_lra", "input_thresh",
    "output_i", "output_tp", "output_lra", "output_thresh", "target_offset",
)


# --------------------------------------------------------------------------
# Deteksi hening
# --------------------------------------------------------------------------
def parse_silencedetect(output: str, duration: float = 0.0) -> List[Range]:
    """Baca output filter ``silencedetect`` menjadi daftar rentang hening."""

    starts = [float(m) for m in _SILENCE_START_RE.findall(output or "")]
    ends = [float(m) for m in _SILENCE_END_RE.findall(output or "")]

    ranges: List[Range] = []
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) else (duration or start)
        if end > start:
            ranges.append((max(0.0, start), end))
    return ranges


def detect_silences(media_path: str | Path, cfg: Config, duration: float = 0.0) -> List[Range]:
    """Jalankan ffmpeg silencedetect pada satu file."""

    threshold = float(cfg.get("audio.silence_threshold_db", -34))
    min_silence = float(cfg.get("audio.min_silence_duration", 0.45))
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(media_path),
        "-af", f"silencedetect=noise={threshold}dB:d={min_silence:g}",
        "-f", "null", "-",
    ]
    proc = run_command(cmd, error_cls=RenderError, label="silencedetect", timeout=900)
    return parse_silencedetect((proc.stderr or "") + (proc.stdout or ""), duration)


def keep_ranges_from_silences(
    silences: Sequence[Range],
    duration: float,
    *,
    keep_padding: float = 0.08,
    min_silence: float = 0.45,
    min_keep: float = 0.12,
) -> List[Range]:
    """Ubah daftar hening menjadi daftar bagian yang dipertahankan.

    ``keep_padding`` menyisakan sedikit hening di tepi supaya potongan tidak
    terdengar seperti terpotong paksa.
    """

    duration = max(0.0, float(duration))
    if duration <= 0:
        return []

    trimmed: List[Range] = []
    for start, end in sorted(silences):
        start = max(0.0, float(start) + keep_padding)
        end = min(duration, float(end) - keep_padding)
        if end - start >= min_silence * 0.5 and end > start:
            trimmed.append((start, end))

    if not trimmed:
        return [(0.0, duration)]

    keeps: List[Range] = []
    cursor = 0.0
    for start, end in trimmed:
        if start - cursor >= min_keep:
            keeps.append((round(cursor, 3), round(start, 3)))
        cursor = max(cursor, end)
    if duration - cursor >= min_keep:
        keeps.append((round(cursor, 3), round(duration, 3)))
    return keeps or [(0.0, duration)]


# --------------------------------------------------------------------------
# Pemetaan waktu setelah potong
# --------------------------------------------------------------------------
def total_kept_duration(keep_ranges: Sequence[Range]) -> float:
    return round(sum(max(0.0, end - start) for start, end in keep_ranges), 3)


def remap_time(time: float, keep_ranges: Sequence[Range]) -> Optional[float]:
    """Waktu asli -> waktu pada hasil potongan.

    Kalau ``time`` jatuh di bagian yang dibuang, dikembalikan batas terdekat
    (bukan ``None``), supaya subtitle tetap menempel di potongan berikutnya.
    ``None`` hanya untuk daftar kosong.
    """

    if not keep_ranges:
        return None
    offset = 0.0
    for start, end in keep_ranges:
        span = max(0.0, end - start)
        if time < start:
            return round(offset, 3)  # berada di dalam bagian yang dibuang
        if time <= end:
            return round(offset + (time - start), 3)
        offset += span
    return round(offset, 3)


def remap_words(words: Sequence[Word], keep_ranges: Sequence[Range]) -> List[Word]:
    """Geser timestamp kata mengikuti potongan hening; kata yang ikut terbuang dihapus."""

    if not keep_ranges:
        return list(words)

    out: List[Word] = []
    for word in words:
        inside = any(
            min(word.end, end) - max(word.start, start) > 0 for start, end in keep_ranges
        )
        if not inside:
            continue
        new_start = remap_time(word.start, keep_ranges)
        new_end = remap_time(word.end, keep_ranges)
        if new_start is None or new_end is None:
            continue
        if new_end <= new_start:
            new_end = new_start + 0.05
        out.append(
            Word(start=new_start, end=new_end, text=word.text, probability=word.probability)
        )
    return out


def build_cut_filters(keep_ranges: Sequence[Range]) -> Tuple[str, str]:
    """Filter ffmpeg untuk membuang bagian di luar ``keep_ranges``.

    Mengembalikan ``(filter_video, filter_audio)``. String kosong berarti tidak
    ada yang perlu dipotong.
    """

    if not keep_ranges or len(keep_ranges) <= 0:
        return "", ""
    conditions = "+".join(
        f"between(t,{start:.3f},{end:.3f})" for start, end in keep_ranges
    )
    video = f"select='{conditions}',setpts=N/FRAME_RATE/TB"
    audio = f"aselect='{conditions}',asetpts=N/SR/TB"
    return video, audio


# --------------------------------------------------------------------------
# Loudness
# --------------------------------------------------------------------------
def loudnorm_filter(
    cfg: Config, measured: Optional[Dict[str, float]] = None, *, linear: bool = True
) -> str:
    """Bangun string filter ``loudnorm``. Dengan ``measured`` = tahap kedua."""

    target = float(cfg.get("audio.target_lufs", -14))
    true_peak = float(cfg.get("audio.true_peak", -1.5))
    parts = [f"loudnorm=I={target:g}", f"TP={true_peak:g}", "LRA=11"]
    if measured:
        for key in ("input_i", "input_tp", "input_lra", "input_thresh"):
            if key in measured:
                parts.append(f"measured_{key.split('_', 1)[1]}={measured[key]:g}")
        if "target_offset" in measured:
            parts.append(f"offset={measured['target_offset']:g}")
        parts.append("linear=true" if linear else "linear=false")
    return ":".join(parts)


def measure_loudness(media_path: str | Path, cfg: Config) -> Optional[Dict[str, float]]:
    """Tahap pengukuran loudnorm; ``None`` kalau gagal (nanti pakai satu tahap)."""

    cmd = [
        ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(media_path),
        "-af", f"{loudnorm_filter(cfg)}:print_format=json",
        "-f", "null", "-",
    ]
    try:
        proc = run_command(cmd, error_cls=RenderError, label="loudnorm-measure", timeout=900)
    except RenderError as exc:
        log.warning("Pengukuran loudness gagal, dipakai satu tahap saja: %s", exc)
        return None

    text = (proc.stderr or "") + (proc.stdout or "")
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        log.warning("Output loudnorm tidak memuat JSON; dipakai satu tahap saja.")
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        log.warning("JSON loudnorm tidak bisa dibaca; dipakai satu tahap saja.")
        return None

    measured: Dict[str, float] = {}
    for key in _LOUDNORM_KEYS:
        value = data.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number or abs(number) == float("inf"):  # NaN / inf
            continue
        measured[key] = number
    if "input_i" not in measured:
        return None
    log.debug("Loudness terukur: %s", measured)
    return measured


def plan_silence_cuts(
    media_path: str | Path, cfg: Config, duration: float
) -> Tuple[List[Range], float]:
    """Rencanakan pemotongan hening untuk satu klip.

    Mengembalikan ``(keep_ranges, durasi_baru)``. ``keep_ranges`` kosong berarti
    tidak ada yang dipotong.
    """

    if not bool(cfg.get("audio.remove_silence", False)):
        return [], duration

    silences = detect_silences(media_path, cfg, duration)
    if not silences:
        log.info("Tidak ada jeda hening yang memenuhi syarat.")
        return [], duration

    keeps = keep_ranges_from_silences(
        silences,
        duration,
        keep_padding=float(cfg.get("audio.keep_padding", 0.08)),
        min_silence=float(cfg.get("audio.min_silence_duration", 0.45)),
    )
    new_duration = total_kept_duration(keeps)
    if new_duration <= 0 or new_duration >= duration - 0.15:
        log.info("Pemotongan hening dilewati (hemat waktunya tidak berarti).")
        return [], duration

    log.info(
        "Potong hening: %d bagian dipertahankan, %.1fs -> %.1fs (hemat %.1fs)",
        len(keeps), duration, new_duration, duration - new_duration,
    )
    return keeps, new_duration
