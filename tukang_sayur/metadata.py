"""Tahap 7 - Metadata unggah: judul, caption, hashtag, lalu tulis JSON + CSV.

CSV-nya sengaja dibuat gampang dibuka di Excel/Sheets: satu baris per klip,
lengkap dengan skor dan timestamp sumbernya, jadi enak untuk penjadwalan
unggahan atau audit hasil.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import Config
from .errors import LLMError
from .llm.base import LLMProvider
from .llm.parsing import parse_metadata_items
from .logging_utils import get_logger
from .models import ClipResult, JobResult, ScoredCandidate
from .prompts import SYSTEM_METADATA, metadata_prompt
from .utils import atomic_write_text, format_timestamp, write_json

log = get_logger("metadata")

DEFAULT_HASHTAGS = [
    "fyp", "foryou", "viral", "shorts", "reels", "tiktok", "videoviral", "clip",
]

CSV_COLUMNS = [
    "index", "file", "title", "caption", "hashtags", "score",
    "score_hook", "score_emotion", "score_clarity", "score_standalone",
    "hook_text", "duration", "source_start", "source_end",
    "source_start_tc", "source_end_tc", "thumbnail", "resolution", "reframe_mode",
]


# --------------------------------------------------------------------------
# Cadangan tanpa LLM
# --------------------------------------------------------------------------
def _first_sentence(text: str, limit: int = 140) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    match = re.search(r"^(.{20,%d}?[.!?])\s" % limit, cleaned)
    if match:
        return match.group(1)
    return cleaned[:limit].rsplit(" ", 1)[0]


def fallback_metadata(item: ScoredCandidate, cfg: Config) -> Dict[str, Any]:
    """Metadata tanpa LLM: dari hook dan kalimat pertama transkrip."""

    hook = item.hook_text.strip()
    snippet = _first_sentence(item.candidate.transcript_text)
    title = (hook or snippet or "Klip menarik")[:60]
    caption_parts = [part for part in (hook, snippet) if part]
    caption = " ".join(caption_parts) if caption_parts else "Klip pilihan dari video panjang."
    if not caption.rstrip().endswith("?"):
        caption = f"{caption.rstrip('.')}. Setuju nggak?"
    count = int(cfg.get("metadata.hashtag_count", 8))
    return {
        "title": title,
        "caption": caption,
        "hashtags": DEFAULT_HASHTAGS[: max(1, count)],
    }


# --------------------------------------------------------------------------
# Dengan LLM
# --------------------------------------------------------------------------
def generate_metadata(
    items: Sequence[ScoredCandidate],
    cfg: Config,
    provider: Optional[LLMProvider],
) -> List[Dict[str, Any]]:
    """Buat metadata tiap klip; LLM kalau bisa, cadangan kalau gagal."""

    if not items:
        return []

    results = [fallback_metadata(item, cfg) for item in items]
    if not bool(cfg.get("metadata.use_llm", True)) or provider is None:
        log.info("Metadata dibuat tanpa LLM (%d klip).", len(items))
        return results

    prompt = metadata_prompt(
        [item.candidate for item in items],
        language=str(cfg.get("metadata.language", "id")),
        hashtag_count=int(cfg.get("metadata.hashtag_count", 8)),
    )
    try:
        response = provider.complete(prompt, system=SYSTEM_METADATA)
        parsed = parse_metadata_items(response.text, len(items))
    except (LLMError, ValueError) as exc:
        log.warning("Metadata LLM gagal (%s); memakai versi cadangan.", exc)
        return results

    count = int(cfg.get("metadata.hashtag_count", 8))
    for position, entry in enumerate(parsed):
        if not entry:
            continue
        if entry.get("title"):
            results[position]["title"] = str(entry["title"])[:80]
        if entry.get("caption"):
            results[position]["caption"] = str(entry["caption"])
        tags = [t for t in entry.get("hashtags") or [] if t]
        if tags:
            results[position]["hashtags"] = tags[: max(1, count)]
    log.info("Metadata dibuat lewat LLM untuk %d klip.", len(items))
    return results


# --------------------------------------------------------------------------
# Penulisan berkas
# --------------------------------------------------------------------------
def clip_row(clip: ClipResult) -> Dict[str, Any]:
    breakdown = clip.score_breakdown or {}
    return {
        "index": clip.index,
        "file": Path(clip.video_path).name,
        "title": clip.title,
        "caption": clip.caption,
        "hashtags": " ".join(f"#{tag.lstrip('#')}" for tag in clip.hashtags),
        "score": round(float(clip.score), 2),
        "score_hook": breakdown.get("hook", ""),
        "score_emotion": breakdown.get("emotion", ""),
        "score_clarity": breakdown.get("clarity", ""),
        "score_standalone": breakdown.get("standalone", ""),
        "hook_text": clip.hook_text,
        "duration": round(float(clip.duration), 2),
        "source_start": round(float(clip.source_start), 2),
        "source_end": round(float(clip.source_end), 2),
        "source_start_tc": format_timestamp(clip.source_start),
        "source_end_tc": format_timestamp(clip.source_end),
        "thumbnail": Path(clip.thumbnail_path).name if clip.thumbnail_path else "",
        "resolution": f"{clip.width}x{clip.height}" if clip.width else "",
        "reframe_mode": clip.reframe_mode,
    }


def write_metadata_files(
    job: JobResult, cfg: Config, output_dir: str | Path
) -> Dict[str, str]:
    """Tulis ``metadata.json`` dan ``metadata.csv`` di folder keluaran job."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_name = str(cfg.get("metadata.json_filename", "metadata.json"))
    csv_name = str(cfg.get("metadata.csv_filename", "metadata.csv"))
    json_path = output_dir / json_name
    csv_path = output_dir / csv_name

    write_json(json_path, job.to_dict())

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for clip in job.clips:
        writer.writerow(clip_row(clip))
    atomic_write_text(csv_path, buffer.getvalue())

    log.info("Metadata ditulis: %s dan %s", json_path.name, csv_path.name)
    return {"json": str(json_path), "csv": str(csv_path)}
