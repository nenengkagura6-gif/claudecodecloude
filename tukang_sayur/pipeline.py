"""Perakit seluruh tahap: dari URL/file sampai klip + metadata siap unggah.

Satu pemanggilan :func:`run_clip_job` = satu video sumber. Setiap tahap punya
penanganan error sendiri: kegagalan di satu klip tidak membatalkan klip lain,
dan pesan errornya selalu menyebut tahap mana yang bermasalah.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from .audio import plan_silence_cuts, remap_words
from .candidates import find_candidates
from .config import Config
from .errors import RenderError, TukangSayurError
from .ingest import ingest
from .llm.base import LLMProvider
from .llm.factory import build_provider
from .logging_utils import get_logger
from .metadata import generate_metadata, write_metadata_files
from .models import ClipResult, JobResult, ScoredCandidate, Transcript
from .reframe import plan_reframe
from .render import RenderRequest, extract_clip, make_thumbnail, render_clip, resolve_watermark
from .scoring import best_score, score_candidates, select_clips
from .subtitles import HookStyle, SubtitleStyle, relativize_words, write_ass
from .transcribe import Transcriber, transcribe_media
from .utils import human_duration, probe_media, rmtree_quiet, slugify, write_json

log = get_logger("pipeline")

#: ``progress(tahap, pesan, 0.0-1.0)``
ProgressFn = Callable[[str, str, float], None]


@dataclass
class ClipJobOptions:
    """Pilihan yang biasanya datang dari CLI/UI."""

    max_clips: Optional[int] = None
    output_dir: Optional[Path] = None
    job_name: Optional[str] = None
    keep_intermediate: Optional[bool] = None
    skip_metadata_llm: bool = False
    no_subtitles: bool = False
    no_hook: bool = False
    dry_run: bool = False


@dataclass
class _ClipContext:
    """Data kerja per klip (tidak dipakai di luar modul ini)."""

    item: ScoredCandidate
    position: int
    clip_path: Path
    ass_path: Optional[Path] = None
    keep_ranges: list = field(default_factory=list)
    duration: float = 0.0


def _noop_progress(stage: str, message: str, fraction: float) -> None:
    return None


def job_output_dir(cfg: Config, title: str, options: ClipJobOptions) -> Path:
    """Folder keluaran: ``output/<slug>-<tanggal-jam>``."""

    if options.output_dir:
        return Path(options.output_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = options.job_name or f"{slugify(title, default='klip')}-{stamp}"
    return cfg.output_dir / name


def run_clip_job(
    source: str,
    cfg: Config,
    *,
    options: Optional[ClipJobOptions] = None,
    provider: Optional[LLMProvider] = None,
    transcriber: Optional[Transcriber] = None,
    progress: Optional[ProgressFn] = None,
) -> JobResult:
    """Jalankan seluruh pipeline untuk satu video sumber."""

    options = options or ClipJobOptions()
    report = progress or _noop_progress
    started = time.time()
    cfg.ensure_dirs()

    # ------------------------------------------------------------ 1. ingest
    report("ingest", "Menyiapkan video sumber", 0.02)
    ingested = ingest(source, cfg)
    output_dir = job_output_dir(cfg, ingested.title, options)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Folder keluaran: %s", output_dir)

    job = JobResult(
        source=source,
        video_path=str(ingested.video_path),
        output_dir=str(output_dir),
        duration=ingested.duration,
    )

    # -------------------------------------------------------- 2. transkripsi
    report("transcribe", "Transkripsi audio (bisa lama di CPU)", 0.08)
    transcript: Transcript = transcribe_media(
        ingested.video_path, cfg, transcriber=transcriber
    )
    job.language = transcript.language
    if transcript.duration <= 0:
        transcript.duration = ingested.duration
    transcript_path = output_dir / "transcript.json"
    write_json(transcript_path, transcript.to_dict())
    job.transcript_path = str(transcript_path)

    # ------------------------------------------------- 3. provider + kandidat
    if provider is None and not options.dry_run:
        provider = build_provider(cfg)

    report("candidates", "Mencari segmen menarik", 0.35)
    candidates = find_candidates(transcript, cfg, provider, wanted=options.max_clips)
    job.candidate_count = len(candidates)

    # ------------------------------------------------------------ 4. skoring
    report("scoring", f"Menilai {len(candidates)} kandidat", 0.45)
    scored = score_candidates(
        candidates, cfg, provider, language=transcript.language or "id"
    )
    # best_score diambil dari SELURUH kandidat yang dinilai (bukan hanya yang
    # dirender), lewat satu fungsi yang sama dengan yang diuji di test.
    job.best_score = best_score(scored)
    selected = select_clips(scored, cfg, max_clips=options.max_clips)

    if options.dry_run:
        log.info("Mode dry-run: berhenti sebelum render.")
        for position, item in enumerate(selected, start=1):
            job.clips.append(
                ClipResult(
                    index=position,
                    video_path="",
                    source_start=item.start,
                    source_end=item.end,
                    duration=item.duration,
                    score=item.score,
                    hook_text=item.hook_text,
                    transcript_text=item.candidate.transcript_text,
                    score_breakdown=item.breakdown.to_dict(),
                )
            )
        paths = write_metadata_files(job, cfg, output_dir)
        job.metadata_json, job.metadata_csv = paths["json"], paths["csv"]
        return job

    # ------------------------------------------------------------- 5. render
    subtitle_style = SubtitleStyle.from_config(cfg)
    hook_style = HookStyle.from_config(cfg)
    if options.no_hook:
        hook_style.enabled = False
    watermark = resolve_watermark(cfg)
    work_dir = cfg.work_dir / f"job-{slugify(ingested.title, default='klip')}"
    work_dir.mkdir(parents=True, exist_ok=True)

    total = max(1, len(selected))
    rendered: List[ClipResult] = []
    for position, item in enumerate(selected, start=1):
        fraction = 0.5 + 0.45 * (position - 1) / total
        report(
            "render",
            f"Klip {position}/{total} (skor {item.score:.0f})",
            fraction,
        )
        try:
            clip = _render_one(
                item=item,
                position=position,
                cfg=cfg,
                transcript=transcript,
                source_video=ingested.video_path,
                work_dir=work_dir,
                output_dir=output_dir,
                subtitle_style=subtitle_style,
                hook_style=hook_style,
                watermark=watermark,
                options=options,
            )
        except TukangSayurError as exc:
            message = f"Klip {position} gagal dirender: {exc}"
            log.error("%s", message)
            job.warnings.append(message)
            continue
        rendered.append(clip)

    if not rendered:
        raise RenderError(
            "Tidak ada klip yang berhasil dirender. Lihat pesan error di atas "
            "(biasanya ffmpeg atau font subtitle)."
        )

    # ----------------------------------------------------------- 6. metadata
    report("metadata", "Menulis judul, caption, hashtag", 0.96)
    provider_for_meta = None if options.skip_metadata_llm else provider
    metas = generate_metadata(
        [selected[clip.index - 1] for clip in rendered], cfg, provider_for_meta
    )
    for clip, meta in zip(rendered, metas):
        clip.title = meta.get("title", clip.hook_text)
        clip.caption = meta.get("caption", "")
        clip.hashtags = list(meta.get("hashtags", []))

    job.clips = rendered
    paths = write_metadata_files(job, cfg, output_dir)
    job.metadata_json, job.metadata_csv = paths["json"], paths["csv"]

    # ------------------------------------------------------------ 7. beresin
    keep = options.keep_intermediate
    if keep is None:
        keep = bool(cfg.get("project.keep_intermediate", False))
    if not keep:
        rmtree_quiet(work_dir)
        ingested.cleanup()
    else:
        log.info("File antara disimpan di %s", work_dir)

    report("done", f"Selesai: {len(rendered)} klip", 1.0)
    log.info(
        "Job selesai dalam %s: %d klip, skor terbaik %.1f, folder %s",
        human_duration(time.time() - started), len(rendered), job.best_score, output_dir,
    )
    return job


def _render_one(
    *,
    item: ScoredCandidate,
    position: int,
    cfg: Config,
    transcript: Transcript,
    source_video: Path,
    work_dir: Path,
    output_dir: Path,
    subtitle_style: SubtitleStyle,
    hook_style: HookStyle,
    watermark: Optional[Path],
    options: ClipJobOptions,
) -> ClipResult:
    """Render satu klip: potong -> reframe -> subtitle -> render -> thumbnail."""

    slug = slugify(item.hook_text or f"klip-{position}", default=f"klip-{position}")
    base_name = f"{position:02d}-{slug}"
    raw_clip = work_dir / f"{base_name}-raw.mp4"
    final_clip = output_dir / f"{base_name}.mp4"

    # Potong dari sumber (Pass A)
    extract_clip(source_video, item.start, item.end, raw_clip, cfg)
    clip_info = probe_media(raw_clip)
    clip_duration = clip_info.duration or item.duration

    # Rencana reframe 9:16 (dianalisis pada potongan, bukan video penuh)
    plan = plan_reframe(
        raw_clip, cfg, frame_width=clip_info.width, frame_height=clip_info.height
    )

    # Potong hening (opsional) -> geser timestamp kata
    keep_ranges, new_duration = plan_silence_cuts(raw_clip, cfg, clip_duration)
    words = relativize_words(transcript.words_between(item.start, item.end), item.start)
    if keep_ranges:
        words = remap_words(words, keep_ranges)
    final_duration = new_duration if keep_ranges else clip_duration

    # Subtitle + hook dalam satu file ASS
    ass_path: Optional[Path] = None
    subtitles_on = bool(cfg.get("subtitle.enabled", True)) and not options.no_subtitles
    if subtitles_on or (hook_style.enabled and item.hook_text):
        ass_path = write_ass(
            work_dir / f"{base_name}.ass",
            words,
            style=subtitle_style,
            hook_style=hook_style,
            hook_text=item.hook_text,
            width=int(cfg.get("render.width", 1080)),
            height=int(cfg.get("render.height", 1920)),
            clip_duration=final_duration,
            subtitles_enabled=subtitles_on,
        )

    render_clip(
        RenderRequest(
            clip_path=raw_clip,
            output_path=final_clip,
            cfg=cfg,
            plan=plan,
            ass_path=ass_path,
            keep_ranges=keep_ranges,
            duration=final_duration,
            watermark_path=watermark,
        )
    )

    thumbnail = make_thumbnail(final_clip, output_dir / f"{base_name}.jpg", cfg)
    final_info = probe_media(final_clip)

    return ClipResult(
        index=position,
        video_path=str(final_clip),
        source_start=item.start,
        source_end=item.end,
        duration=final_info.duration or final_duration,
        score=item.score,
        hook_text=item.hook_text,
        title=item.hook_text,
        thumbnail_path=str(thumbnail) if thumbnail else "",
        subtitle_path=str(ass_path) if ass_path else "",
        transcript_text=item.candidate.transcript_text,
        score_breakdown=item.breakdown.to_dict(),
        reframe_mode=plan.mode,
        width=final_info.width,
        height=final_info.height,
    )
