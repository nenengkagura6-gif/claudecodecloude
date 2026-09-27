"""Tahap 1 - Ingest: unduh video (yt-dlp) atau pakai file lokal, lalu normalisasi.

Hasil tahap ini selalu satu file MP4 yang "jinak" untuk ffmpeg: H.264 + AAC,
fps tetap, tinggi maksimum sesuai config, dan audio 48 kHz stereo. Dengan begitu
tahap render tidak perlu lagi menebak-nebak format sumber.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .errors import DependencyError, IngestError
from .logging_utils import get_logger
from .utils import (
    MediaInfo,
    even,
    ffmpeg_bin,
    probe_media,
    run_command,
    slugify,
    which,
)

log = get_logger("ingest")

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_MEDIA_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".m4v", ".ts", ".mpg", ".mpeg",
    ".wmv", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
}


@dataclass
class IngestResult:
    """Video yang siap dipakai tahap berikutnya."""

    video_path: Path
    source: str
    title: str = ""
    is_url: bool = False
    info: Optional[MediaInfo] = None
    downloaded_path: Optional[Path] = None
    temp_files: List[Path] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.info.duration if self.info else 0.0

    def cleanup(self) -> None:
        for path in self.temp_files:
            try:
                if path.is_file():
                    path.unlink()
            except OSError as exc:  # pragma: no cover - best effort
                log.debug("Gagal hapus file sementara %s: %s", path, exc)


def is_url(source: str) -> bool:
    return bool(_URL_RE.match(str(source).strip()))


def looks_like_media_file(source: str) -> bool:
    return Path(str(source)).suffix.lower() in _MEDIA_SUFFIXES


# --------------------------------------------------------------------------
# Unduh
# --------------------------------------------------------------------------
def _ytdlp_options(cfg: Config, outtmpl: str) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "format": cfg.get("ingest.ytdlp_format"),
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
        "retries": 5,
        "fragment_retries": 5,
        "merge_output_format": "mp4",
    }
    browser = cfg.get("ingest.cookies_from_browser")
    if browser:
        options["cookiesfrombrowser"] = (str(browser),)
    return options


def _download_with_module(url: str, cfg: Config, work_dir: Path) -> tuple[Path, str]:
    import yt_dlp  # type: ignore[import-not-found]

    outtmpl = str(work_dir / "sumber-%(id)s.%(ext)s")
    options = _ytdlp_options(cfg, outtmpl)
    log.info("Mengunduh dengan yt-dlp (modul Python): %s", url)
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    if not info:
        raise IngestError(f"yt-dlp tidak mengembalikan info untuk: {url}")

    path: Optional[str] = None
    for entry in info.get("requested_downloads") or []:
        path = entry.get("filepath") or entry.get("_filename")
        if path:
            break
    if not path:
        path = info.get("filepath") or info.get("_filename")
    if not path or not Path(path).is_file():
        found = _find_downloaded(work_dir)
        if found is None:
            raise IngestError("Unduhan selesai tapi file hasilnya tidak ditemukan.")
        path = str(found)
    return Path(path), str(info.get("title") or "")


def _find_downloaded(work_dir: Path) -> Optional[Path]:
    files = [p for p in work_dir.glob("sumber-*") if p.is_file() and p.suffix != ".part"]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _download_with_cli(url: str, cfg: Config, work_dir: Path) -> tuple[Path, str]:
    binary = which("yt-dlp")
    if not binary:
        raise DependencyError(
            "yt-dlp tidak tersedia. Install dulu: pip install yt-dlp "
            "(atau winget install yt-dlp di Windows)."
        )
    outtmpl = str(work_dir / "sumber-%(id)s.%(ext)s")
    cmd = [
        binary,
        "--no-playlist",
        "--restrict-filenames",
        "--retries", "5",
        "--merge-output-format", "mp4",
        "-f", str(cfg.get("ingest.ytdlp_format")),
        "-o", outtmpl,
        "--no-simulate",
        "--print", "%(title)s",
    ]
    browser = cfg.get("ingest.cookies_from_browser")
    if browser:
        cmd += ["--cookies-from-browser", str(browser)]
    cmd.append(url)

    log.info("Mengunduh dengan yt-dlp (CLI): %s", url)
    proc = run_command(cmd, timeout=None, error_cls=IngestError, label="yt-dlp")
    title = ""
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line and not line.startswith("[") and not Path(line).exists():
            title = line
            break
    found = _find_downloaded(work_dir)
    if found is None:
        raise IngestError("yt-dlp selesai tapi file hasilnya tidak ditemukan.")
    return found, title


def download_video(url: str, cfg: Config, work_dir: Path) -> tuple[Path, str]:
    """Unduh video; pakai modul Python kalau ada, kalau tidak pakai CLI."""

    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return _download_with_cli(url, cfg, work_dir)
    try:
        return _download_with_module(url, cfg, work_dir)
    except IngestError:
        raise
    except Exception as exc:  # yt-dlp melempar banyak jenis error
        raise IngestError(
            f"Gagal mengunduh {url}: {exc}\n"
            "Cek koneksi, atau isi ingest.cookies_from_browser bila video butuh login."
        ) from exc


# --------------------------------------------------------------------------
# Normalisasi
# --------------------------------------------------------------------------
def needs_normalize(info: MediaInfo, cfg: Config) -> bool:
    """Lewati re-encode kalau sumbernya sudah sesuai target."""

    max_height = int(cfg.get("ingest.normalize.max_height", 1080))
    target_fps = float(cfg.get("ingest.normalize.fps", 30))
    sample_rate = int(cfg.get("ingest.normalize.audio_sample_rate", 48000))
    channels = int(cfg.get("ingest.normalize.audio_channels", 2))

    if not info.has_audio:
        return True
    if info.video_codec not in {"h264"} or info.audio_codec not in {"aac"}:
        return True
    if info.height > max_height:
        return True
    if info.fps and info.fps > target_fps + 0.5:
        return True
    if info.audio_sample_rate and info.audio_sample_rate != sample_rate:
        return True
    if info.audio_channels and info.audio_channels > channels:
        return True
    return False


def normalize_media(
    source: Path, cfg: Config, output: Path, info: Optional[MediaInfo] = None
) -> Path:
    """Re-encode ke MP4 H.264/AAC dengan fps & audio seragam."""

    info = info or probe_media(source)
    max_height = int(cfg.get("ingest.normalize.max_height", 1080))
    target_fps = float(cfg.get("ingest.normalize.fps", 30))
    sample_rate = int(cfg.get("ingest.normalize.audio_sample_rate", 48000))
    channels = int(cfg.get("ingest.normalize.audio_channels", 2))

    filters = [f"fps={target_fps:g}"]
    if info.height and info.height > max_height:
        width = even(info.width * max_height / info.height) if info.height else 0
        filters.append(f"scale={width}:{even(max_height)}:flags=bicubic")
    filters.append("format=yuv420p")

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-vf", ",".join(filters),
        "-c:v", str(cfg.get("render.video_codec", "libx264")),
        "-preset", str(cfg.get("render.preset", "veryfast")),
        "-crf", str(cfg.get("render.crf", 20)),
        "-pix_fmt", "yuv420p",
    ]
    if info.has_audio:
        cmd += [
            "-c:a", "aac", "-b:a", "192k",
            "-ar", str(sample_rate), "-ac", str(channels),
        ]
    else:
        # Tambahkan trek diam supaya filter audio di tahap render tidak gagal.
        log.warning("Sumber tanpa audio; menambahkan trek audio diam.")
        cmd = cmd[:1] + [
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}",
            "-shortest",
            "-vf", ",".join(filters),
            "-c:v", str(cfg.get("render.video_codec", "libx264")),
            "-preset", str(cfg.get("render.preset", "veryfast")),
            "-crf", str(cfg.get("render.crf", 20)),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-ar", str(sample_rate), "-ac", str(channels),
        ]
    cmd += ["-movflags", "+faststart", str(output)]

    log.info("Normalisasi video -> %s", output.name)
    run_command(cmd, timeout=None, error_cls=IngestError, label="normalize")
    if not output.is_file() or output.stat().st_size == 0:
        raise IngestError(f"Normalisasi gagal, file kosong: {output}")
    return output


# --------------------------------------------------------------------------
# Pintu masuk
# --------------------------------------------------------------------------
def ingest(source: str, cfg: Config, *, work_dir: Optional[Path] = None) -> IngestResult:
    """Siapkan video sumber dari URL atau file lokal."""

    source = str(source).strip()
    if not source:
        raise IngestError("Sumber video kosong.")

    work = Path(work_dir) if work_dir else cfg.work_dir
    work.mkdir(parents=True, exist_ok=True)

    temp_files: List[Path] = []
    from_url = is_url(source)

    if from_url:
        raw_path, title = download_video(source, cfg, work)
        temp_files.append(raw_path)
    else:
        candidate = Path(source).expanduser()
        if not candidate.is_file():
            raise IngestError(
                f"File tidak ditemukan: {candidate}\n"
                "Kalau ini URL, pastikan diawali http:// atau https://"
            )
        raw_path = candidate
        title = candidate.stem

    info = probe_media(raw_path)
    max_minutes = float(cfg.get("ingest.max_duration_minutes", 180) or 0)
    if max_minutes > 0 and info.duration > max_minutes * 60:
        raise IngestError(
            f"Durasi video {info.duration / 60:.1f} menit melewati batas "
            f"ingest.max_duration_minutes ({max_minutes:g}). Naikkan batasnya "
            "atau potong videonya dulu."
        )
    if info.duration <= 0:
        raise IngestError(f"Durasi video terbaca 0 detik; file mungkin rusak: {raw_path}")

    slug = slugify(title or raw_path.stem, default="sumber")
    if needs_normalize(info, cfg):
        normalized = work / f"{slug}-normalized.mp4"
        normalize_media(raw_path, cfg, normalized, info)
        temp_files.append(normalized)
        video_path = normalized
        info = probe_media(video_path)
    else:
        log.info("Sumber sudah sesuai target, normalisasi dilewati.")
        video_path = raw_path

    log.info(
        "Ingest siap: %s (%.1f detik, %dx%d, %.2f fps, audio=%s)",
        video_path.name, info.duration, info.width, info.height, info.fps,
        "ya" if info.has_audio else "tidak",
    )
    return IngestResult(
        video_path=video_path,
        source=source,
        title=title or slug,
        is_url=from_url,
        info=info,
        downloaded_path=raw_path if from_url else None,
        temp_files=[p for p in temp_files if p != video_path],
    )


def copy_into(path: Path, target_dir: Path) -> Path:
    """Salin file ke folder lain (dipakai UI Gradio untuk file unggahan)."""

    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / path.name
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    return destination
