"""Antarmuka terminal Tukang Sayur.

Contoh pemakaian::

    python -m tukang_sayur clip https://youtu.be/xxxx --max-clips 10
    python -m tukang_sayur clip video.mp4 --provider ollama --no-hook
    python -m tukang_sayur batch daftar.txt            # lanjut otomatis
    python -m tukang_sayur demo                        # uji tanpa internet
    python -m tukang_sayur doctor                      # cek kelengkapan alat
    python -m tukang_sayur ui                          # antarmuka web Gradio
    python -m tukang_sayur serve                       # endpoint untuk n8n
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from . import __version__
from .config import Config
from .errors import TukangSayurError
from .logging_utils import get_logger, setup_logging
from .models import JobResult
from .utils import human_duration

if TYPE_CHECKING:  # hanya untuk anotasi; impor asli dilakukan saat dipakai
    from .pipeline import ClipJobOptions

log = get_logger("cli")


# --------------------------------------------------------------------------
# Argumen
# --------------------------------------------------------------------------
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", help="Path config.yaml (default: ./config.yaml)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KUNCI=NILAI",
        help="Timpa satu nilai config, mis. --set render.crf=18 (bisa berulang)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Tingkat log di terminal (default dari config)",
    )
    parser.add_argument("--log-file", help="Tulis log lengkap ke file ini")


def _add_job_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-clips", type=int, help="Jumlah klip maksimum")
    parser.add_argument("-o", "--output", help="Folder keluaran (default: output/<judul>-<waktu>)")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "gemini", "ollama", "mock"],
        help="Provider LLM (menimpa config)",
    )
    parser.add_argument("--model", help="Nama model LLM (menimpa config)")
    parser.add_argument("--lang", help="Bahasa transkripsi: id, su, atau auto")
    parser.add_argument("--whisper-model", help="Model Whisper: tiny/base/small/medium")
    parser.add_argument("--no-subtitles", action="store_true", help="Tanpa subtitle")
    parser.add_argument("--no-hook", action="store_true", help="Tanpa teks hook")
    parser.add_argument(
        "--remove-silence", action="store_true", help="Potong jeda hening"
    )
    parser.add_argument("--watermark", help="Path gambar watermark (mengaktifkan watermark)")
    parser.add_argument(
        "--keep-intermediate", action="store_true", help="Simpan file antara untuk debug"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Berhenti sebelum render: cuma cari kandidat, nilai, dan tulis metadata",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tukang_sayur",
        description="Tukang Sayur - potong video panjang jadi klip vertikal 9:16 "
        "siap unggah (TikTok/Reels/Shorts).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-V", "--version", action="version", version=f"Tukang Sayur {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clip = subparsers.add_parser("clip", help="Proses satu URL atau file video")
    clip.add_argument("source", help="URL video (YouTube, dll.) atau path file lokal")
    _add_job_args(clip)
    _add_common(clip)

    batch = subparsers.add_parser("batch", help="Proses banyak sumber dari file daftar")
    batch.add_argument("list_file", help="File teks berisi satu URL/path per baris")
    batch.add_argument(
        "--no-resume",
        action="store_true",
        help="Jangan lanjutkan; proses ulang semua sumber",
    )
    _add_job_args(batch)
    _add_common(batch)

    transcribe = subparsers.add_parser("transcribe", help="Hanya transkripsi (hasil JSON)")
    transcribe.add_argument("source", help="URL atau file video/audio")
    transcribe.add_argument("-o", "--output", help="File JSON keluaran")
    transcribe.add_argument("--lang", help="Bahasa: id, su, auto")
    transcribe.add_argument("--whisper-model", help="Model Whisper")
    _add_common(transcribe)

    demo = subparsers.add_parser(
        "demo", help="Uji pipeline penuh tanpa internet (video contoh + LLM tiruan)"
    )
    demo.add_argument("--duration", type=float, default=90.0, help="Durasi video contoh")
    demo.add_argument("--max-clips", type=int, default=2, help="Jumlah klip")
    demo.add_argument("-o", "--output", help="Folder keluaran")
    _add_common(demo)

    ui = subparsers.add_parser("ui", help="Jalankan antarmuka web Gradio")
    ui.add_argument("--host")
    ui.add_argument("--port", type=int)
    ui.add_argument("--share", action="store_true", help="Buat tautan publik Gradio")
    _add_common(ui)

    serve = subparsers.add_parser("serve", help="Jalankan API/webhook (untuk n8n)")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    _add_common(serve)

    doctor = subparsers.add_parser("doctor", help="Cek ffmpeg, yt-dlp, model, kunci API")
    _add_common(doctor)

    return parser


# --------------------------------------------------------------------------
# Config dari argumen
# --------------------------------------------------------------------------
def config_from_args(args: argparse.Namespace) -> Config:
    overrides: Dict[str, Any] = {}
    if getattr(args, "provider", None):
        overrides["llm.provider"] = args.provider
    if getattr(args, "lang", None):
        overrides["transcription.language"] = args.lang
    if getattr(args, "whisper_model", None):
        overrides["transcription.model"] = args.whisper_model
    if getattr(args, "remove_silence", False):
        overrides["audio.remove_silence"] = True
    if getattr(args, "watermark", None):
        overrides["watermark.enabled"] = True
        overrides["watermark.path"] = args.watermark

    extra = Config.parse_cli_set(getattr(args, "set", []) or [])
    overrides.update(extra)

    cfg = Config.load(getattr(args, "config", None), overrides=overrides)

    # --model perlu tahu provider final, jadi diterapkan sesudah config dimuat
    if getattr(args, "model", None):
        cfg.set(f"llm.providers.{cfg.provider_name()}.model", args.model)

    level = getattr(args, "log_level", None) or cfg.get("logging.level", "INFO")
    log_file = getattr(args, "log_file", None) or cfg.get("logging.file")
    setup_logging(level, log_file, force=True)
    return cfg


def job_options_from_args(args: argparse.Namespace) -> "ClipJobOptions":
    from .pipeline import ClipJobOptions

    return ClipJobOptions(
        max_clips=getattr(args, "max_clips", None),
        output_dir=Path(args.output) if getattr(args, "output", None) else None,
        keep_intermediate=True if getattr(args, "keep_intermediate", False) else None,
        no_subtitles=bool(getattr(args, "no_subtitles", False)),
        no_hook=bool(getattr(args, "no_hook", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )


# --------------------------------------------------------------------------
# Tampilan hasil
# --------------------------------------------------------------------------
def print_job_summary(job: JobResult) -> None:
    print()
    print("=" * 62)
    print(f"  Selesai: {len(job.clips)} klip  |  skor terbaik: {job.best_score:.1f}")
    print(f"  Folder : {job.output_dir}")
    print("=" * 62)
    for clip in job.clips:
        name = Path(clip.video_path).name if clip.video_path else "(dry-run)"
        print(
            f"  [{clip.index:02d}] skor {clip.score:5.1f}  "
            f"{human_duration(clip.duration):>6}  {name}"
        )
        if clip.hook_text:
            print(f"        hook   : {clip.hook_text}")
        if clip.title:
            print(f"        judul  : {clip.title}")
        if clip.hashtags:
            print(f"        hashtag: {' '.join('#' + t.lstrip('#') for t in clip.hashtags[:8])}")
    if job.metadata_csv:
        print(f"\n  Metadata: {job.metadata_csv}")
    for warning in job.warnings:
        print(f"  ! {warning}")
    print()


# --------------------------------------------------------------------------
# Perintah
# --------------------------------------------------------------------------
def cmd_clip(args: argparse.Namespace) -> int:
    from .pipeline import run_clip_job

    cfg = config_from_args(args)
    job = run_clip_job(args.source, cfg, options=job_options_from_args(args))
    print_job_summary(job)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from .batch import run_batch

    cfg = config_from_args(args)
    result = run_batch(
        args.list_file,
        cfg,
        options=job_options_from_args(args),
        resume=not args.no_resume,
    )
    print()
    print("=" * 62)
    print(
        f"  Batch: {result.done} sukses, {result.skipped} dilewati, "
        f"{result.failed} gagal dari {result.total} sumber"
    )
    print("=" * 62)
    for source, error in result.errors.items():
        print(f"  ! {source}\n    {error.splitlines()[0]}")
    print()
    return 0 if result.failed == 0 else 1


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .ingest import ingest
    from .transcribe import transcribe_media, transcript_stats
    from .utils import write_json

    cfg = config_from_args(args)
    cfg.ensure_dirs()
    ingested = ingest(args.source, cfg)
    transcript = transcribe_media(ingested.video_path, cfg)
    target = Path(args.output) if args.output else cfg.output_dir / "transcript.json"
    write_json(target, transcript.to_dict())
    print(f"Transkrip disimpan: {target}")
    for key, value in transcript_stats(transcript).items():
        print(f"  {key:16}: {value}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import DEMO_SCRIPT, make_dummy_video
    from .llm.providers import MockProvider
    from .pipeline import run_clip_job
    from .transcribe import DummyTranscriber

    cfg = config_from_args(args)
    cfg.set("llm.provider", "mock")
    cfg.set("transcription.engine", "dummy")
    cfg.set("candidates.min_duration", 12)
    cfg.set("candidates.max_duration", 45)
    cfg.set("candidates.min_clips", 2)
    cfg.ensure_dirs()

    video = cfg.work_dir / "demo-tukang-sayur.mp4"
    print(f"Membuat video contoh {args.duration:g} detik ...")
    make_dummy_video(video, duration=args.duration, script=DEMO_SCRIPT)

    options = job_options_from_args(args)
    options.max_clips = args.max_clips
    job = run_clip_job(
        str(video),
        cfg,
        options=options,
        provider=MockProvider(),
        transcriber=DummyTranscriber(cfg, script=DEMO_SCRIPT),
    )
    print_job_summary(job)
    print("Demo dipakai untuk memastikan ffmpeg, subtitle, dan render jalan.")
    print("Untuk video sungguhan: python -m tukang_sayur clip <url> --max-clips 10")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .webapp import launch_ui

    cfg = config_from_args(args)
    return launch_ui(cfg, host=args.host, port=args.port, share=bool(args.share))


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import run_server

    cfg = config_from_args(args)
    return run_server(cfg, host=args.host, port=args.port)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "TIDAK"
    print(f"  [{mark}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util

    from .utils import which

    cfg = config_from_args(args)
    print("\nPemeriksaan Tukang Sayur")
    print("-" * 62)

    essentials = [
        _check("ffmpeg", bool(which("ffmpeg")), which("ffmpeg") or "install ffmpeg dulu"),
        _check("ffprobe", bool(which("ffprobe")), which("ffprobe") or "satu paket dengan ffmpeg"),
    ]
    for module, note in [
        ("yt_dlp", "untuk unduh dari URL: pip install yt-dlp"),
        ("faster_whisper", "untuk transkripsi: pip install faster-whisper"),
        ("requests", "untuk provider LLM: pip install requests"),
        ("cv2", "untuk face tracking: pip install opencv-python-headless"),
        ("mediapipe", "opsional, deteksi wajah lebih akurat"),
        ("gradio", "opsional, untuk perintah 'ui'"),
        ("fastapi", "opsional, untuk perintah 'serve'"),
    ]:
        found = importlib.util.find_spec(module) is not None
        _check(f"paket {module}", found, "" if found else note)

    provider = cfg.provider_name()
    conf = cfg.provider_config()
    key = cfg.api_key()
    needs_key = provider not in {"ollama", "mock"}
    _check(
        f"LLM {provider} ({conf.get('model')})",
        (not needs_key) or bool(key),
        "kunci API terbaca dari .env" if key else
        ("tidak perlu kunci API" if not needs_key else
         f"isi {conf.get('api_key_env')} di file .env"),
    )
    print("-" * 62)
    print(f"  work_dir  : {cfg.work_dir}")
    print(f"  output_dir: {cfg.output_dir}")
    print(f"  cache_dir : {cfg.cache_dir}")
    print(f"  config    : {cfg.path or '(default bawaan)'}")
    print()
    if not all(essentials):
        print("ffmpeg wajib ada. Windows: https://www.gyan.dev/ffmpeg/builds/ , "
              "Linux: sudo apt install ffmpeg\n")
        return 1
    print("Siap dipakai. Coba dulu: python -m tukang_sayur demo\n")
    return 0


COMMANDS = {
    "clip": cmd_clip,
    "batch": cmd_batch,
    "transcribe": cmd_transcribe,
    "demo": cmd_demo,
    "ui": cmd_ui,
    "serve": cmd_serve,
    "doctor": cmd_doctor,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse sudah menjaga
        parser.error(f"Perintah tidak dikenal: {args.command}")

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nDihentikan.", file=sys.stderr)
        return 130
    except TukangSayurError as exc:
        log.error("Gagal pada tahap '%s': %s", getattr(exc, "stage", "umum"), exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
