"""Membaca balasan LLM dengan sabar.

LLM kadang membalas JSON bersih, kadang JSON dibungkus ```json, kadang diawali
kalimat basa-basi, kadang pakai nama kunci yang beda ("mulai" bukan "start"),
kadang ada koma berlebih. Modul ini menormalkan semua itu menjadi objek
:class:`~tukang_sayur.models.Candidate` dan
:class:`~tukang_sayur.models.ScoredCandidate`.

Aturan penting: **jangan pernah diam-diam mengembalikan 0**. Kalau sebuah nilai
tidak ditemukan, fungsi di sini mengembalikan ``None`` atau melempar
:class:`LLMParseError`, supaya pemanggilnya bisa memilih fallback yang benar.
Inilah cara kita mencegah bug klasik "best_score = 0 karena salah path".
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..errors import LLMParseError
from ..logging_utils import get_logger
from ..models import SCORE_DIMENSIONS, Candidate, ScoreBreakdown, ScoredCandidate
from ..utils import parse_timecode, truncate_words

log = get_logger("llm.parsing")

# ------------------------------- alias kunci ------------------------------
START_KEYS = ("start", "start_time", "start_seconds", "start_sec", "from", "begin",
              "mulai", "awal", "start_ts", "t_start")
END_KEYS = ("end", "end_time", "end_seconds", "end_sec", "to", "stop", "finish",
            "selesai", "akhir", "end_ts", "t_end")
DURATION_KEYS = ("duration", "durasi", "length", "dur", "panjang")
HOOK_KEYS = ("hook_text", "hook", "hooktext", "hook_line", "teks_hook", "judul_hook",
             "hook_title")
REASON_KEYS = ("reason", "alasan", "why", "rationale", "explanation", "note", "catatan",
               "justification")
TITLE_KEYS = ("title", "judul", "headline")
TEXT_KEYS = ("transcript", "transcript_text", "text", "quote", "kutipan", "isi")
INDEX_KEYS = ("index", "idx", "id", "candidate", "candidate_index", "candidate_id",
              "no", "nomor", "number", "clip", "clip_index", "rank")

TOTAL_KEYS = ("score", "total", "total_score", "final_score", "overall", "overall_score",
              "viral_score", "skor", "nilai", "weighted_score", "score_total")

BREAKDOWN_CONTAINERS = ("scores", "score", "breakdown", "score_breakdown", "scoring",
                        "detail", "details", "dimensions", "rincian", "nilai",
                        "sub_scores", "subscores")

DIMENSION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hook": ("hook", "hook_score", "hook_strength", "kekuatan_hook", "hooking",
             "attention", "opening"),
    "emotion": ("emotion", "emotion_score", "emotional", "emotional_impact", "emosi",
                "feeling", "emotive"),
    "clarity": ("clarity", "clarity_score", "clear", "clearness", "kejelasan",
                "understandability", "coherence"),
    "standalone": ("standalone", "standalone_score", "self_contained", "selfcontained",
                   "standalone_value", "berdiri_sendiri", "mandiri", "context_free",
                   "completeness"),
}

LIST_KEYS = ("candidates", "clips", "segments", "items", "results", "scores", "data",
             "kandidat", "klip", "output", "list", "entries", "metadata")

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_LINE_COMMENT_RE = re.compile(r"(^|\s)//[^\n\r]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SMART_QUOTES = {
    "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'",
    "«": '"', "»": '"',
}


# --------------------------------------------------------------------------
# Ekstraksi JSON
# --------------------------------------------------------------------------
def strip_fences(text: str) -> str:
    """Ambil isi blok ``` kalau ada; kalau tidak, kembalikan teks apa adanya."""

    if "```" not in text:
        return text
    matches = _FENCE_RE.findall(text)
    if matches:
        # pilih blok terpanjang: biasanya itu JSON-nya, bukan contoh singkat
        return max(matches, key=len).strip()
    return text.replace("```", " ")


def _repair(text: str) -> str:
    """Perbaikan ringan yang aman untuk JSON hasil LLM."""

    for bad, good in _SMART_QUOTES.items():
        text = text.replace(bad, good)
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _LINE_COMMENT_RE.sub(r"\1", text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    text = re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "0", text)
    text = re.sub(r"(?<![\w\"'])None(?![\w\"'])", "null", text)
    text = re.sub(r"(?<![\w\"'])True(?![\w\"'])", "true", text)
    text = re.sub(r"(?<![\w\"'])False(?![\w\"'])", "false", text)
    return text


def _balanced_candidates(text: str) -> Iterable[str]:
    """Hasilkan potongan ``{...}``/``[...]`` yang seimbang, terpanjang dulu."""

    openers = {"{": "}", "[": "]"}
    found: List[str] = []
    for index, char in enumerate(text):
        if char not in openers:
            continue
        closer = openers[char]
        depth = 0
        in_string = False
        escaped = False
        for cursor in range(index, len(text)):
            current = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == char:
                depth += 1
            elif current == closer:
                depth -= 1
                if depth == 0:
                    found.append(text[index : cursor + 1])
                    break
    found.sort(key=len, reverse=True)
    return found


def extract_json(text: Any) -> Any:
    """Cari objek/array JSON pertama yang valid di dalam balasan LLM."""

    if isinstance(text, (dict, list)):
        return text
    raw = str(text or "").strip()
    if not raw:
        raise LLMParseError("Balasan LLM kosong, tidak ada JSON yang bisa dibaca.")

    attempts = [raw, strip_fences(raw)]
    attempts.append(_repair(attempts[-1]))
    for attempt in attempts:
        attempt = attempt.strip()
        if not attempt:
            continue
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    for chunk in _balanced_candidates(attempts[-1]):
        for variant in (chunk, _repair(chunk)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                continue

    preview = raw[:300].replace("\n", " ")
    raise LLMParseError(
        f"Tidak menemukan JSON yang valid di balasan LLM. Awal balasan: {preview!r}"
    )


def find_items(payload: Any, *, keys: Sequence[str] = LIST_KEYS) -> List[Dict[str, Any]]:
    """Ambil daftar objek dari payload, seberapa dalam pun ia disembunyikan."""

    if isinstance(payload, list):
        items = [item for item in payload if isinstance(item, Mapping)]
        if items:
            return [dict(item) for item in items]
        return []

    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload:
                found = find_items(payload[key], keys=keys)
                if found:
                    return found
        # telusuri semua nilai (breadth pertama pada list, lalu dict)
        for value in payload.values():
            if isinstance(value, list):
                found = find_items(value, keys=keys)
                if found:
                    return found
        for value in payload.values():
            if isinstance(value, Mapping):
                found = find_items(value, keys=keys)
                if found:
                    return found
        # objek tunggal (bukan daftar) juga sah
        if any(key in payload for key in START_KEYS + TOTAL_KEYS + HOOK_KEYS):
            return [dict(payload)]
    return []


# --------------------------------------------------------------------------
# Pembaca nilai
# --------------------------------------------------------------------------
def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def pick(entry: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    """Ambil nilai pertama yang ada (cocok tanpa peduli huruf besar/spasi)."""

    normalized = {_normalize_key(k): v for k, v in entry.items()}
    for key in keys:
        value = normalized.get(_normalize_key(key))
        if value is not None and value != "":
            return value
    return None


def pick_number(entry: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    value = pick(entry, keys)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def pick_text(entry: Mapping[str, Any], keys: Sequence[str], default: str = "") -> str:
    value = pick(entry, keys)
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value).strip()
    return str(value).strip()


def pick_time(entry: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    """Waktu bisa 12.5, "12.5", "00:00:12.5", atau "12s"."""

    value = pick(entry, keys)
    return parse_timecode(value) if value is not None else None


# --------------------------------------------------------------------------
# Kandidat
# --------------------------------------------------------------------------
def parse_candidates(
    raw: Any,
    *,
    max_hook_words: int = 8,
    source_duration: Optional[float] = None,
) -> List[Candidate]:
    """Ubah balasan LLM menjadi daftar :class:`Candidate`.

    Entri yang tidak punya start/end (atau start/duration) yang masuk akal
    dilewati dengan catatan di log, bukan bikin seluruh proses gagal.
    """

    payload = extract_json(raw)
    entries = find_items(payload)
    if not entries:
        raise LLMParseError(
            "Balasan LLM tidak memuat daftar kandidat. "
            f"Kunci yang ada: {list(payload)[:8] if isinstance(payload, Mapping) else type(payload).__name__}"
        )

    candidates: List[Candidate] = []
    for position, entry in enumerate(entries, start=1):
        start = pick_time(entry, START_KEYS)
        end = pick_time(entry, END_KEYS)
        if start is None and end is None:
            log.debug("Kandidat #%d dilewati: tidak ada start/end (%s)", position, entry)
            continue
        if end is None:
            duration = pick_number(entry, DURATION_KEYS)
            if duration is None or duration <= 0:
                log.debug("Kandidat #%d dilewati: tidak ada end maupun duration", position)
                continue
            end = float(start or 0.0) + float(duration)
        if start is None:
            duration = pick_number(entry, DURATION_KEYS)
            if duration is None or duration <= 0:
                log.debug("Kandidat #%d dilewati: tidak ada start", position)
                continue
            start = max(0.0, float(end) - float(duration))
        if float(end) <= float(start):
            log.debug("Kandidat #%d dilewati: end <= start (%.2f <= %.2f)", position, end, start)
            continue
        if source_duration and float(start) >= float(source_duration):
            log.debug("Kandidat #%d dilewati: start di luar durasi video", position)
            continue

        hook = truncate_words(pick_text(entry, HOOK_KEYS), max_hook_words)
        title = pick_text(entry, TITLE_KEYS)
        if not hook and title:
            hook = truncate_words(title, max_hook_words)

        candidates.append(
            Candidate(
                start=float(start),
                end=float(end),
                hook_text=hook,
                reason=pick_text(entry, REASON_KEYS),
                title=title,
                transcript_text=pick_text(entry, TEXT_KEYS),
                index=int(pick_number(entry, INDEX_KEYS) or position),
            )
        )

    if not candidates:
        raise LLMParseError(
            f"LLM mengirim {len(entries)} entri tapi tidak ada yang punya "
            "start/end yang valid."
        )
    return candidates


# --------------------------------------------------------------------------
# Skor
# --------------------------------------------------------------------------
def extract_breakdown(entry: Mapping[str, Any]) -> Tuple[Optional[ScoreBreakdown], int]:
    """Cari nilai per dimensi, baik yang rata maupun yang bersarang.

    Mengembalikan ``(breakdown, jumlah_dimensi_ditemukan)``. ``breakdown`` bisa
    ``None`` kalau tidak satu pun dimensi ditemukan - sengaja, supaya pemanggil
    tidak menyangka skor 0 itu penilaian sungguhan.
    """

    sources: List[Mapping[str, Any]] = [entry]
    for container in BREAKDOWN_CONTAINERS:
        value = pick(entry, (container,))
        if isinstance(value, Mapping):
            sources.append(value)

    values: Dict[str, float] = {}
    for dimension, aliases in DIMENSION_ALIASES.items():
        for source in sources:
            number = pick_number(source, aliases)
            if number is not None:
                values[dimension] = number
                break

    if not values:
        return None, 0
    return ScoreBreakdown(**{name: values.get(name, 0.0) for name in SCORE_DIMENSIONS}), len(values)


def _explicit_total(entry: Mapping[str, Any]) -> Optional[float]:
    """Total yang ditulis LLM sendiri (kalau ada dan bukan objek)."""

    normalized = {_normalize_key(k): v for k, v in entry.items()}
    for key in TOTAL_KEYS:
        value = normalized.get(_normalize_key(key))
        if value is None or isinstance(value, (Mapping, list, bool)):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if match:
            return float(match.group())
    return None


def score_from_entry(
    entry: Mapping[str, Any], weights: Mapping[str, float]
) -> Tuple[Optional[float], ScoreBreakdown]:
    """Hitung skor akhir satu entri.

    Urutan yang dipakai:

    1. rata-rata berbobot dari dimensi (paling dipercaya, konsisten dg config),
    2. total yang ditulis LLM - dipakai kalau dimensi tidak ada, atau kalau
       dimensi semuanya 0 padahal totalnya tidak 0,
    3. ``None`` kalau memang tidak ada angka apa pun (bukan 0!).
    """

    breakdown, found = extract_breakdown(entry)
    explicit = _explicit_total(entry)

    if breakdown is not None and found:
        weighted = breakdown.weighted_total(weights)
        if weighted > 0:
            return weighted, breakdown
        if explicit is not None and explicit > 0:
            return max(0.0, min(100.0, explicit)), breakdown
        return weighted, breakdown

    if explicit is not None:
        total = max(0.0, min(100.0, explicit))
        # sebar rata ke dimensi supaya CSV tetap terisi sesuatu yang jujur
        return total, ScoreBreakdown(**{name: total for name in SCORE_DIMENSIONS})

    return None, ScoreBreakdown()


def _align_indices(entries: Sequence[Mapping[str, Any]], count: int) -> List[Optional[int]]:
    """Tentukan kandidat ke-berapa yang dimaksud setiap entri skor.

    LLM kadang memakai nomor mulai 1, kadang mulai 0, kadang tidak menulis
    nomor sama sekali (urutannya saja).
    """

    raw: List[Optional[int]] = []
    for entry in entries:
        number = pick_number(entry, INDEX_KEYS)
        raw.append(int(number) if number is not None else None)

    known = [n for n in raw if n is not None]
    if not known:
        return [position if position < count else None for position in range(len(entries))]

    offset = 0
    if min(known) >= 1 and max(known) <= count:
        offset = -1  # penomoran mulai dari 1
    elif min(known) >= 0 and max(known) <= count - 1:
        offset = 0
    else:
        # nomor tidak masuk akal -> pakai urutan kemunculan
        return [position if position < count else None for position in range(len(entries))]

    resolved: List[Optional[int]] = []
    for position, number in enumerate(raw):
        if number is None:
            resolved.append(position if position < count else None)
            continue
        target = number + offset
        resolved.append(target if 0 <= target < count else None)
    return resolved


def parse_scores(
    raw: Any,
    candidates: Sequence[Candidate],
    weights: Mapping[str, float],
) -> List[ScoredCandidate]:
    """Gabungkan balasan penilaian LLM dengan daftar kandidat.

    Kandidat yang tidak dinilai LLM tetap dikembalikan, tapi dengan
    ``score = None`` yang diwakili skor 0 **dan** ditandai di log; pemanggil
    (lihat :mod:`tukang_sayur.scoring`) yang memutuskan fallback-nya.
    """

    if not candidates:
        return []

    payload = extract_json(raw)
    entries = find_items(payload)
    if not entries:
        raise LLMParseError("Balasan penilaian LLM tidak memuat daftar skor.")

    targets = _align_indices(entries, len(candidates))
    scored: Dict[int, ScoredCandidate] = {}

    for entry, target in zip(entries, targets):
        if target is None:
            log.debug("Entri skor tanpa pasangan kandidat diabaikan: %s", entry)
            continue
        total, breakdown = score_from_entry(entry, weights)
        if total is None:
            log.warning(
                "Kandidat #%d tidak punya angka skor apa pun di balasan LLM.", target + 1
            )
            continue
        if target in scored:
            log.debug("Skor ganda untuk kandidat #%d, yang pertama dipakai.", target + 1)
            continue
        scored[target] = ScoredCandidate(
            candidate=candidates[target],
            breakdown=breakdown,
            score=total,
            reason=pick_text(entry, REASON_KEYS),
        )

    if not scored:
        raise LLMParseError(
            "Tidak ada satu pun skor yang bisa dibaca dari balasan LLM."
        )

    missing = [i for i in range(len(candidates)) if i not in scored]
    if missing:
        log.warning(
            "%d kandidat tidak dinilai LLM (nomor: %s).",
            len(missing), ", ".join(str(i + 1) for i in missing),
        )
    return [scored[index] for index in sorted(scored)]


# --------------------------------------------------------------------------
# Metadata unggah
# --------------------------------------------------------------------------
def parse_metadata_items(raw: Any, count: int) -> List[Dict[str, Any]]:
    """Baca judul/caption/hashtag untuk tiap klip."""

    payload = extract_json(raw)
    entries = find_items(payload, keys=("items", "clips", "metadata", "results", "data"))
    if not entries:
        raise LLMParseError("Balasan metadata LLM tidak memuat daftar item.")

    targets = _align_indices(entries, count)
    output: List[Dict[str, Any]] = [{} for _ in range(count)]
    for entry, target in zip(entries, targets):
        if target is None:
            continue
        hashtags = pick(entry, ("hashtags", "tags", "hashtag", "tagar"))
        if isinstance(hashtags, str):
            tags = [t.strip() for t in re.split(r"[,\s]+", hashtags) if t.strip()]
        elif isinstance(hashtags, (list, tuple)):
            tags = [str(t).strip() for t in hashtags if str(t).strip()]
        else:
            tags = []
        output[target] = {
            "title": pick_text(entry, TITLE_KEYS + ("hook_text",)),
            "caption": pick_text(entry, ("caption", "description", "deskripsi", "caption_text")),
            "hashtags": [t.lstrip("#") for t in tags],
        }
    return output
