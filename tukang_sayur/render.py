"""Tahap 6 - Render dengan ffmpeg ``filter_complex``.

Dua kali jalan supaya hasilnya bisa diandalkan (dan mudah di-debug):

* **Pass A** :func:`extract_clip` - potong rentang [start, end] dari video
  sumber dengan re-encode, jadi potongannya akurat sampai frame.
* **Pass B** :func:`render_clip` - satu ``filter_complex`` yang mengerjakan:
  potong hening (opsional) -> crop 9:16 mengikuti wajah atau latar blur ->
  skala ke 1080x1920 -> bakar subtitle ASS + hook -> watermark (opsional) ->
  loudnorm audio.

Soal hook: hook adalah event di file ASS, bukan gambar beku. Video dan audio
jalan normal sejak detik 0; hook cuma teks di atasnya. Tidak ada freeze-frame
di mana pun di berkas ini - silakan cari kata "freeze", tidak akan ketemu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .audio import Range, build_cut_filters, loudnorm_filter, measure_loudness
from .config import Config
from .errors import RenderError
from .logging_utils import get_logger
from .reframe import MODE_BLUR, MODE_CENTER, MODE_TRACK, ReframePlan
from .utils import even, ffmpeg_bin, probe_media, run_command

log = get_logger("render")

WATERMARK_POSITIONS = {
    "top-left": ("{margin}", "{margin}"),
    "top-right": ("main_w-overlay_w-{margin}", "{margin}"),
    "bottom-left": ("{margin}", "main_h-overlay_h-{margin}"),
    "bottom-right": ("main_w-overlay_w-{margin}", "main_h-overlay_h-{margin}"),
}


# --------------------------------------------------------------------------
# Escaping untuk filtergraph
# --------------------------------------------------------------------------
def escape_filter_path(path: str | Path) -> str:
    """Amankan path untuk dipakai di dalam filter ffmpeg (``ass=filename=...``).

    Di Windows, ``C:\\kerja\\a.ass`` harus jadi ``C\\:/kerja/a.ass`` supaya
    titik dua tidak dianggap pemisah opsi filter.
    """

    text = str(path).replace("\\", "/")
    text = text.replace("'", r"\'")
    text = text.replace(":", r"\:")
    text = text.replace("[", r"\[").replace("]", r"\]")
    text = text.replace(",", r"\,").replace(";", r"\;")
    return text


# --------------------------------------------------------------------------
# Pass A: potong klip
# --------------------------------------------------------------------------
def extract_clip(
    source: str | Path,
    start: float,
    end: float,
    output: str | Path,
    cfg: Config,
) -> Path:
    """Potong [start, end] dari video sumber dengan re-encode (akurat per frame)."""

    duration = max(0.1, float(end) - float(start))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, float(start)):.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-c:v", str(cfg.get("render.video_codec", "libx264")),
        "-preset", str(cfg.get("render.preset", "veryfast")),
        "-crf", str(cfg.get("render.crf", 20)),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(output),
    ]
    run_command(cmd, timeout=None, error_cls=RenderError, label="extract")
    if not output.is_file() or output.stat().st_size == 0:
        raise RenderError(f"Potongan klip gagal dibuat: {output}")
    return output


# --------------------------------------------------------------------------
# Pass B: filtergraph
# --------------------------------------------------------------------------
@dataclass
class RenderRequest:
    """Semua bahan untuk merender satu klip jadi MP4 vertikal."""

    clip_path: Path
    output_path: Path
    cfg: Config
    plan: ReframePlan
    ass_path: Optional[Path] = None
    keep_ranges: List[Range] = field(default_factory=list)
    duration: float = 0.0
    watermark_path: Optional[Path] = None
    loudness: Optional[Dict[str, float]] = None


def build_video_chain(
    plan: ReframePlan,
    cfg: Config,
    *,
    ass_path: Optional[Path] = None,
    keep_ranges: Sequence[Range] = (),
    watermark: bool = False,
) -> Tuple[List[str], str]:
    """Susun bagian video dari ``filter_complex``.

    Mengembalikan ``(daftar_bagian_filter, label_keluaran)``.
    """

    width = even(cfg.get("render.width", 1080))
    height = even(cfg.get("render.height", 1920))
    blur_strength = int(cfg.get("reframe.blur_strength", 40) or 40)
    parts: List[str] = []
    current = "0:v"
    video_cut, _ = build_cut_filters(keep_ranges)

    # PENTING soal urutan:
    # Ekspresi crop pelacak wajah memakai variabel waktu `t`, sedangkan filter
    # select+setpts (pemotong hening) me-reset waktu. Jadi saat melacak wajah,
    # crop harus dipasang SEBELUM pemotongan; kalau tidak, gerakan kamera akan
    # bergeser terhadap gambarnya. Untuk mode blur/center (tanpa `t`), potong
    # dulu supaya frame yang dibuang tidak ikut diproses berat.
    if plan.is_tracking and plan.crop_width and plan.crop_height:
        parts.append(
            f"[{current}]crop=w={plan.crop_width}:h={plan.crop_height}:"
            f"x='{plan.crop_x_expression()}':y=(in_h-out_h)/2[vcrop]"
        )
        current = "vcrop"
        if video_cut:
            parts.append(f"[{current}]{video_cut}[vcut]")
            current = "vcut"
        parts.append(
            f"[{current}]scale={width}:{height}:flags=bicubic,setsar=1[vframe]"
        )
        current = "vframe"
    elif plan.mode == MODE_BLUR:
        if video_cut:
            parts.append(f"[{current}]{video_cut}[vcut]")
            current = "vcut"
        # latar: video di-zoom & di-blur; depan: video utuh di tengah
        parts.append(f"[{current}]split=2[vbg][vfg]")
        parts.append(
            f"[vbg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur={max(2, blur_strength // 2)}:1,"
            f"eq=brightness=-0.06:saturation=1.1,setsar=1[vbgb]"
        )
        parts.append(
            f"[vfg]scale={width}:-2:flags=bicubic,setsar=1[vfgs]"
        )
        parts.append(
            "[vbgb][vfgs]overlay=x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2:"
            "shortest=1[vframe]"
        )
        current = "vframe"
    else:  # MODE_CENTER atau fallback lain
        if video_cut:
            parts.append(f"[{current}]{video_cut}[vcut]")
            current = "vcut"
        parts.append(
            f"[{current}]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1[vframe]"
        )
        current = "vframe"

    # Subtitle ASS + hook (satu file, dua gaya)
    if ass_path is not None:
        parts.append(f"[{current}]ass=filename='{escape_filter_path(ass_path)}'[vsub]")
        current = "vsub"

    # Watermark
    if watermark:
        position = str(cfg.get("watermark.position", "top-right")).lower()
        margin = int(cfg.get("watermark.margin", 40))
        scale_width = int(cfg.get("watermark.scale_width", 160))
        opacity = float(cfg.get("watermark.opacity", 0.75))
        x_expr, y_expr = WATERMARK_POSITIONS.get(position, WATERMARK_POSITIONS["top-right"])
        parts.append(
            f"[1:v]scale={scale_width}:-1,format=rgba,"
            f"colorchannelmixer=aa={max(0.0, min(1.0, opacity)):g}[wm]"
        )
        parts.append(
            f"[{current}][wm]overlay=x={x_expr.format(margin=margin)}:"
            f"y={y_expr.format(margin=margin)}[vwm]"
        )
        current = "vwm"

    parts.append(f"[{current}]format=yuv420p[vout]")
    return parts, "vout"


def build_audio_chain(
    cfg: Config,
    *,
    keep_ranges: Sequence[Range] = (),
    loudness: Optional[Dict[str, float]] = None,
    has_audio: bool = True,
) -> Tuple[List[str], Optional[str]]:
    """Susun bagian audio dari ``filter_complex``."""

    if not has_audio:
        return [], None

    chain: List[str] = []
    _, audio_cut = build_cut_filters(keep_ranges)
    if audio_cut:
        chain.append(audio_cut)
    if bool(cfg.get("audio.loudnorm", True)):
        chain.append(loudnorm_filter(cfg, loudness))
    sample_rate = int(cfg.get("ingest.normalize.audio_sample_rate", 48000))
    chain.append(f"aresample={sample_rate}:async=1")

    return [f"[0:a]{','.join(chain)}[aout]"], "aout"


def build_ffmpeg_command(request: RenderRequest, *, has_audio: bool = True) -> List[str]:
    """Rangkai perintah ffmpeg lengkap untuk Pass B."""

    cfg = request.cfg
    video_parts, video_label = build_video_chain(
        request.plan,
        cfg,
        ass_path=request.ass_path,
        keep_ranges=request.keep_ranges,
        watermark=request.watermark_path is not None,
    )
    audio_parts, audio_label = build_audio_chain(
        cfg,
        keep_ranges=request.keep_ranges,
        loudness=request.loudness,
        has_audio=has_audio,
    )

    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(request.clip_path)]
    if request.watermark_path is not None:
        cmd += ["-i", str(request.watermark_path)]

    cmd += ["-filter_complex", ";".join(video_parts + audio_parts)]
    cmd += ["-map", f"[{video_label}]"]
    if audio_label:
        cmd += ["-map", f"[{audio_label}]"]
    else:
        cmd += ["-an"]

    threads = int(cfg.get("render.threads", 0) or 0)
    cmd += [
        "-r", str(cfg.get("render.fps", 30)),
        "-c:v", str(cfg.get("render.video_codec", "libx264")),
        "-preset", str(cfg.get("render.preset", "veryfast")),
        "-crf", str(cfg.get("render.crf", 20)),
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",
    ]
    if audio_label:
        cmd += [
            "-c:a", str(cfg.get("render.audio_codec", "aac")),
            "-b:a", str(cfg.get("render.audio_bitrate", "160k")),
            "-ar", str(cfg.get("ingest.normalize.audio_sample_rate", 48000)),
            "-ac", "2",
        ]
    if threads > 0:
        cmd += ["-threads", str(threads)]
    cmd += ["-movflags", "+faststart", str(request.output_path)]
    return cmd


def render_clip(request: RenderRequest) -> Path:
    """Jalankan Pass B dan kembalikan path hasil."""

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    info = probe_media(request.clip_path)

    if request.loudness is None and bool(request.cfg.get("audio.loudnorm", True)) and info.has_audio:
        request.loudness = measure_loudness(request.clip_path, request.cfg)

    cmd = build_ffmpeg_command(request, has_audio=info.has_audio)
    log.info(
        "Render klip -> %s (mode %s%s)",
        request.output_path.name,
        request.plan.describe(),
        ", potong hening" if request.keep_ranges else "",
    )
    run_command(cmd, timeout=None, error_cls=RenderError, label="render")

    if not request.output_path.is_file() or request.output_path.stat().st_size == 0:
        raise RenderError(f"Render gagal, file kosong: {request.output_path}")
    return request.output_path


def make_thumbnail(video_path: str | Path, output: str | Path, cfg: Config) -> Optional[Path]:
    """Ambil satu frame sebagai thumbnail (dari klip final, bukan sumber)."""

    if not bool(cfg.get("render.thumbnail", True)):
        return None

    video_path, output = Path(video_path), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        duration = probe_media(video_path).duration
    except Exception:  # pragma: no cover
        duration = 0.0
    ratio = float(cfg.get("render.thumbnail_offset_ratio", 0.25))
    offset = max(0.0, min(duration * max(0.0, min(1.0, ratio)), max(0.0, duration - 0.1)))

    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{offset:.3f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "3", str(output),
    ]
    try:
        run_command(cmd, timeout=300, error_cls=RenderError, label="thumbnail")
    except RenderError as exc:
        log.warning("Thumbnail gagal dibuat: %s", exc)
        return None
    return output if output.is_file() else None


def resolve_watermark(cfg: Config) -> Optional[Path]:
    """Cek watermark: aktif di config dan filenya benar-benar ada."""

    if not bool(cfg.get("watermark.enabled", False)):
        return None
    raw = cfg.get("watermark.path")
    if not raw:
        log.warning("watermark.enabled true tapi watermark.path kosong; dilewati.")
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_file():
        log.warning("File watermark tidak ditemukan: %s; dilewati.", path)
        return None
    return path
