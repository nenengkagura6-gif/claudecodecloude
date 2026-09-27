"""Tahap 3 - Candidate Finder: LLM mencari 5-15 segmen menarik.

Selain memanggil LLM, modul ini yang memastikan potongan **benar-benar** jatuh
di batas kalimat, panjangnya masuk rentang config, tidak tumpang tindih, dan
tidak keluar dari durasi video. Semua perhitungan waktunya fungsi murni supaya
bisa diuji tanpa LLM.
"""

from __future__ import annotations

import bisect
from typing import List, Optional, Sequence, Tuple

from .config import Config
from .errors import CandidateError, LLMError
from .llm.base import LLMProvider
from .llm.parsing import parse_candidates
from .logging_utils import get_logger
from .models import Candidate, Transcript
from .prompts import SYSTEM_CANDIDATE, candidate_prompt, chunk_segments
from .utils import truncate_words

log = get_logger("candidates")

#: Tanda akhir kalimat yang kita percaya sebagai titik potong.
SENTENCE_END_CHARS = ".!?…"

#: Toleransi geser (detik) saat mencari batas kalimat terdekat.
SNAP_TOLERANCE = 2.5


# --------------------------------------------------------------------------
# Batas kalimat
# --------------------------------------------------------------------------
def sentence_boundaries(transcript: Transcript) -> Tuple[List[float], List[float]]:
    """Kumpulkan waktu awal dan akhir kalimat dari transkrip.

    Awal kalimat = awal segmen, dan awal kata sesudah kata bertanda akhir.
    Akhir kalimat = akhir segmen, dan akhir kata bertanda akhir kalimat.
    """

    starts: set[float] = set()
    ends: set[float] = set()

    for segment in transcript.segments:
        starts.add(round(segment.start, 3))
        ends.add(round(segment.end, 3))
        words = segment.words
        for position, word in enumerate(words):
            text = word.text.strip()
            if not text:
                continue
            if text[-1] in SENTENCE_END_CHARS:
                ends.add(round(word.end, 3))
                if position + 1 < len(words):
                    starts.add(round(words[position + 1].start, 3))

    return sorted(starts), sorted(ends)


def nearest_boundary(
    value: float,
    boundaries: Sequence[float],
    *,
    tolerance: float = SNAP_TOLERANCE,
    prefer: str = "nearest",
) -> Optional[float]:
    """Batas terdekat dari ``value`` dalam jarak ``tolerance``.

    ``prefer='before'`` mengutamakan batas sebelum ``value`` (dipakai untuk awal
    klip supaya kata pertama tidak terpotong), ``prefer='after'`` mengutamakan
    batas sesudahnya (untuk akhir klip).
    """

    if not boundaries:
        return None

    position = bisect.bisect_left(boundaries, value)
    before = boundaries[position - 1] if position > 0 else None
    after = boundaries[position] if position < len(boundaries) else None
    if after is not None and abs(after - value) < 1e-9:
        return after

    options: List[float] = []
    if prefer == "before":
        options = [b for b in (before, after) if b is not None]
    elif prefer == "after":
        options = [b for b in (after, before) if b is not None]
    else:
        options = sorted(
            [b for b in (before, after) if b is not None], key=lambda b: abs(b - value)
        )

    for option in options:
        if abs(option - value) <= tolerance:
            return option
    return None


def snap_range(
    start: float,
    end: float,
    starts: Sequence[float],
    ends: Sequence[float],
    *,
    min_duration: float,
    max_duration: float,
    video_duration: float,
    tolerance: float = SNAP_TOLERANCE,
) -> Tuple[float, float]:
    """Geser (start, end) ke batas kalimat lalu paksa masuk rentang durasi."""

    video_duration = max(0.0, float(video_duration))
    start = max(0.0, float(start))
    end = float(end)
    if video_duration:
        end = min(end, video_duration)
    if end <= start:
        end = min(video_duration or (start + min_duration), start + min_duration)

    snapped_start = nearest_boundary(start, starts, tolerance=tolerance, prefer="before")
    if snapped_start is not None:
        start = max(0.0, snapped_start)
    snapped_end = nearest_boundary(end, ends, tolerance=tolerance, prefer="after")
    if snapped_end is not None:
        end = snapped_end
    if video_duration:
        end = min(end, video_duration)

    # Terlalu pendek -> perpanjang ke akhir kalimat berikutnya.
    if end - start < min_duration:
        for boundary in ends:
            if boundary >= start + min_duration and (
                not video_duration or boundary <= video_duration
            ):
                end = boundary
                break
        else:
            end = min(start + min_duration, video_duration or (start + min_duration))

    # Masih pendek (mentok ujung video) -> tarik awalnya ke belakang.
    if end - start < min_duration:
        for boundary in reversed(starts):
            if boundary <= end - min_duration:
                start = boundary
                break
        else:
            start = max(0.0, end - min_duration)

    # Terlalu panjang -> pendekkan ke akhir kalimat terakhir yang masih muat.
    if end - start > max_duration:
        limit = start + max_duration
        fitting = [b for b in ends if start + min_duration <= b <= limit]
        end = fitting[-1] if fitting else limit

    if video_duration:
        end = min(end, video_duration)
        start = max(0.0, min(start, max(0.0, end - 0.5)))
    return round(start, 3), round(end, 3)


def apply_padding(
    start: float,
    end: float,
    *,
    pad_start: float,
    pad_end: float,
    video_duration: float,
) -> Tuple[float, float]:
    """Beri jeda kecil di depan/belakang supaya kata tidak terpotong encoder."""

    new_start = max(0.0, start - max(0.0, pad_start))
    new_end = end + max(0.0, pad_end)
    if video_duration:
        new_end = min(new_end, video_duration)
    if new_end <= new_start:
        new_end = min(video_duration or (new_start + 0.5), new_start + 0.5)
    return round(new_start, 3), round(new_end, 3)


def drop_overlaps(
    candidates: Sequence[Candidate], max_overlap_ratio: float
) -> List[Candidate]:
    """Buang kandidat yang terlalu mirip dengan yang sudah diterima.

    Urutan masuk dianggap urutan prioritas (kandidat awal lebih dipercaya).
    """

    kept: List[Candidate] = []
    for candidate in candidates:
        clash = next(
            (
                other
                for other in kept
                if candidate.overlap_ratio(other) > max_overlap_ratio
            ),
            None,
        )
        if clash is not None:
            log.debug(
                "Kandidat %.1f-%.1f dibuang: tumpang tindih %.0f%% dengan %.1f-%.1f",
                candidate.start, candidate.end,
                candidate.overlap_ratio(clash) * 100, clash.start, clash.end,
            )
            continue
        kept.append(candidate)
    return kept


def normalize_candidates(
    candidates: Sequence[Candidate],
    transcript: Transcript,
    cfg: Config,
) -> List[Candidate]:
    """Rapikan kandidat mentah dari LLM menjadi potongan yang layak dirender."""

    min_duration = float(cfg.get("candidates.min_duration", 20))
    max_duration = float(cfg.get("candidates.max_duration", 90))
    snap = bool(cfg.get("candidates.snap_to_sentence", True))
    pad_start = float(cfg.get("candidates.padding_start", 0.25))
    pad_end = float(cfg.get("candidates.padding_end", 0.35))
    max_overlap = float(cfg.get("candidates.max_overlap_ratio", 0.35))
    max_hook_words = int(cfg.get("candidates.max_hook_words", 8))
    video_duration = float(transcript.duration or 0.0)

    starts, ends = sentence_boundaries(transcript) if snap else ([], [])

    cleaned: List[Candidate] = []
    for candidate in candidates:
        start, end = candidate.start, candidate.end
        if snap:
            start, end = snap_range(
                start, end, starts, ends,
                min_duration=min_duration,
                max_duration=max_duration,
                video_duration=video_duration,
            )
        else:
            end = min(end, video_duration) if video_duration else end
        start, end = apply_padding(
            start, end, pad_start=pad_start, pad_end=pad_end,
            video_duration=video_duration,
        )
        duration = end - start
        if duration < min_duration * 0.6:
            log.debug(
                "Kandidat %.1f-%.1f dibuang: durasi %.1fs terlalu pendek",
                start, end, duration,
            )
            continue

        text = transcript.text_between(start, end)
        hook = truncate_words(candidate.hook_text, max_hook_words)
        if not hook:
            hook = truncate_words(text, max_hook_words)
        cleaned.append(
            Candidate(
                start=start,
                end=end,
                hook_text=hook,
                reason=candidate.reason,
                title=candidate.title,
                transcript_text=text,
                index=candidate.index,
            )
        )

    cleaned = drop_overlaps(cleaned, max_overlap)
    cleaned.sort(key=lambda c: c.start)
    for position, candidate in enumerate(cleaned, start=1):
        candidate.index = position
    return cleaned


# --------------------------------------------------------------------------
# Pemanggilan LLM
# --------------------------------------------------------------------------
def find_candidates(
    transcript: Transcript,
    cfg: Config,
    provider: LLMProvider,
    *,
    wanted: Optional[int] = None,
) -> List[Candidate]:
    """Minta LLM mencari kandidat klip, lalu rapikan hasilnya."""

    if not transcript.segments:
        raise CandidateError("Transkrip kosong, tidak ada yang bisa dicari.")

    min_clips = int(cfg.get("candidates.min_clips", 5))
    max_clips = int(cfg.get("candidates.max_clips", 15))
    target = int(wanted or max_clips)
    target = max(min_clips, min(max_clips, target))

    min_duration = float(cfg.get("candidates.min_duration", 20))
    max_duration = float(cfg.get("candidates.max_duration", 90))
    max_hook_words = int(cfg.get("candidates.max_hook_words", 8))
    language = transcript.language or str(cfg.get("transcription.language", "id"))

    chunks = chunk_segments(transcript.segments)
    total_parts = len(chunks)
    # Minta lebih banyak dari target: nanti disaring oleh normalisasi & skor.
    per_chunk = max(2, -(-int(target * 1.4) // total_parts))

    raw_candidates: List[Candidate] = []
    for part, chunk in enumerate(chunks, start=1):
        if not chunk:
            continue
        prompt = candidate_prompt(
            chunk,
            wanted=per_chunk,
            min_duration=min_duration,
            max_duration=max_duration,
            language=language,
            video_duration=transcript.duration,
            max_hook_words=max_hook_words,
            part=part,
            total_parts=total_parts,
        )
        log.info(
            "Mencari kandidat (bagian %d/%d, minta %d) lewat %s ...",
            part, total_parts, per_chunk, provider.describe(),
        )
        try:
            result = provider.complete(prompt, system=SYSTEM_CANDIDATE)
            found = parse_candidates(
                result.text,
                max_hook_words=max_hook_words,
                source_duration=transcript.duration,
            )
        except LLMError as exc:
            if total_parts == 1:
                raise CandidateError(f"Gagal mencari kandidat: {exc}") from exc
            log.warning("Bagian %d/%d gagal, dilanjutkan: %s", part, total_parts, exc)
            continue
        log.info("Bagian %d/%d: %d kandidat mentah", part, total_parts, len(found))
        raw_candidates.extend(found)

    if not raw_candidates:
        raise CandidateError(
            "LLM tidak memberi kandidat yang bisa dipakai. Coba turunkan "
            "candidates.min_duration atau pakai model LLM yang lebih pintar."
        )

    cleaned = normalize_candidates(raw_candidates, transcript, cfg)
    if not cleaned:
        raise CandidateError(
            "Semua kandidat gugur saat dirapikan (durasi terlalu pendek atau "
            "tumpang tindih). Coba longgarkan candidates.min_duration."
        )

    if len(cleaned) > max_clips:
        log.debug("Kandidat %d > max_clips %d; sisanya dinilai juga lalu disaring.",
                  len(cleaned), max_clips)
    log.info(
        "Kandidat siap dinilai: %d (durasi %.0f-%.0fs)",
        len(cleaned),
        min(c.duration for c in cleaned),
        max(c.duration for c in cleaned),
    )
    return cleaned
