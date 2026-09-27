"""Struktur data inti: transkrip, kandidat klip, skor, dan hasil render.

Semua kelas punya ``to_dict()``/``from_dict()`` supaya gampang di-cache ke JSON
dan diuji tanpa harus menjalankan ffmpeg atau memanggil LLM.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# Nama atribut skor final. Dipakai bersama oleh scoring, pipeline, dan metadata
# supaya tidak ada lagi kasus "best_score = 0 karena salah path properti".
SCORE_ATTR = "score"

#: Dimensi penilaian LLM (urutannya dipakai di prompt & CSV).
SCORE_DIMENSIONS = ("hook", "emotion", "clarity", "standalone")


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


# --------------------------------------------------------------------------
# Transkrip
# --------------------------------------------------------------------------
@dataclass
class Word:
    """Satu kata beserta timestamp-nya (detik, relatif awal video sumber)."""

    start: float
    end: float
    text: str
    probability: float = 1.0

    def __post_init__(self) -> None:
        self.start = max(0.0, float(self.start))
        self.end = max(self.start, float(self.end))
        self.text = str(self.text)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def shifted(self, offset: float) -> "Word":
        return Word(
            start=max(0.0, self.start + offset),
            end=max(0.0, self.end + offset),
            text=self.text,
            probability=self.probability,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": _round(self.start),
            "end": _round(self.end),
            "text": self.text,
            "probability": _round(self.probability, 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Word":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", data.get("start", 0.0))),
            text=str(data.get("text", data.get("word", ""))),
            probability=float(data.get("probability", data.get("prob", 1.0)) or 1.0),
        )


@dataclass
class Segment:
    """Satu segmen (biasanya satu kalimat) hasil Whisper."""

    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.start = max(0.0, float(self.start))
        self.end = max(self.start, float(self.end))
        self.text = str(self.text).strip()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": _round(self.start),
            "end": _round(self.end),
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Segment":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", data.get("start", 0.0))),
            text=str(data.get("text", "")),
            words=[Word.from_dict(w) for w in data.get("words", []) or []],
        )


@dataclass
class Transcript:
    """Hasil transkripsi lengkap satu video."""

    language: str
    duration: float
    segments: List[Segment] = field(default_factory=list)
    model: str = ""
    source: str = ""

    @property
    def words(self) -> List[Word]:
        """Semua kata dari semua segmen, terurut menurut waktu."""

        out: List[Word] = []
        for segment in self.segments:
            out.extend(segment.words)
        out.sort(key=lambda w: (w.start, w.end))
        return out

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text).strip()

    def words_between(self, start: float, end: float) -> List[Word]:
        """Kata yang bagian terbesarnya berada di dalam rentang [start, end]."""

        selected: List[Word] = []
        for word in self.words:
            if word.end <= start or word.start >= end:
                continue
            overlap = min(word.end, end) - max(word.start, start)
            if word.duration <= 0:
                selected.append(word)
            elif overlap >= word.duration * 0.5:
                selected.append(word)
        return selected

    def text_between(self, start: float, end: float) -> str:
        return " ".join(w.text.strip() for w in self.words_between(start, end)).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "duration": _round(self.duration),
            "model": self.model,
            "source": self.source,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Transcript":
        return cls(
            language=str(data.get("language", "")),
            duration=float(data.get("duration", 0.0)),
            segments=[Segment.from_dict(s) for s in data.get("segments", []) or []],
            model=str(data.get("model", "")),
            source=str(data.get("source", "")),
        )


# --------------------------------------------------------------------------
# Kandidat & skor
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    """Satu calon klip yang diusulkan LLM (sudah dirapikan ke batas kalimat)."""

    start: float
    end: float
    hook_text: str
    reason: str = ""
    title: str = ""
    transcript_text: str = ""
    index: int = 0

    def __post_init__(self) -> None:
        self.start = max(0.0, float(self.start))
        self.end = max(self.start, float(self.end))
        self.hook_text = str(self.hook_text or "").strip()
        self.reason = str(self.reason or "").strip()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap_with(self, other: "Candidate") -> float:
        """Durasi tumpang tindih (detik) dengan kandidat lain."""

        return max(0.0, min(self.end, other.end) - max(self.start, other.start))

    def overlap_ratio(self, other: "Candidate") -> float:
        """Rasio tumpang tindih terhadap kandidat yang lebih pendek (0..1)."""

        shortest = min(self.duration, other.duration)
        if shortest <= 0:
            return 0.0
        return self.overlap_with(other) / shortest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start": _round(self.start),
            "end": _round(self.end),
            "duration": _round(self.duration),
            "hook_text": self.hook_text,
            "title": self.title,
            "reason": self.reason,
            "transcript_text": self.transcript_text,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Candidate":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            hook_text=str(data.get("hook_text", "")),
            reason=str(data.get("reason", "")),
            title=str(data.get("title", "")),
            transcript_text=str(data.get("transcript_text", "")),
            index=int(data.get("index", 0) or 0),
        )


@dataclass
class ScoreBreakdown:
    """Skor per dimensi, masing-masing 0-100."""

    hook: float = 0.0
    emotion: float = 0.0
    clarity: float = 0.0
    standalone: float = 0.0

    def __post_init__(self) -> None:
        for name in SCORE_DIMENSIONS:
            setattr(self, name, self._clamp(getattr(self, name)))

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return max(0.0, min(100.0, number))

    def weighted_total(self, weights: Mapping[str, float]) -> float:
        """Rata-rata berbobot; bobot dinormalisasi supaya hasil tetap 0-100."""

        total_weight = 0.0
        accumulated = 0.0
        for name in SCORE_DIMENSIONS:
            weight = float(weights.get(name, 0.0) or 0.0)
            if weight <= 0:
                continue
            total_weight += weight
            accumulated += weight * getattr(self, name)
        if total_weight <= 0:
            # Tidak ada bobot yang valid -> pakai rata-rata biasa, jangan 0.
            return _round(sum(getattr(self, n) for n in SCORE_DIMENSIONS) / 4.0, 2)
        return _round(accumulated / total_weight, 2)

    def to_dict(self) -> Dict[str, float]:
        return {name: _round(getattr(self, name), 2) for name in SCORE_DIMENSIONS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScoreBreakdown":
        return cls(**{name: data.get(name, 0.0) for name in SCORE_DIMENSIONS})


@dataclass
class ScoredCandidate:
    """Kandidat + hasil penilaian. Atribut skor final bernama ``score``."""

    candidate: Candidate
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    score: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        self.score = ScoreBreakdown._clamp(self.score)

    # Delegasi supaya pemakai tidak perlu menulis .candidate.start terus-terusan
    @property
    def start(self) -> float:
        return self.candidate.start

    @property
    def end(self) -> float:
        return self.candidate.end

    @property
    def duration(self) -> float:
        return self.candidate.duration

    @property
    def hook_text(self) -> str:
        return self.candidate.hook_text

    def to_dict(self) -> Dict[str, Any]:
        data = self.candidate.to_dict()
        data.update(
            {
                "score": _round(self.score, 2),
                "score_breakdown": self.breakdown.to_dict(),
                "score_reason": self.reason,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScoredCandidate":
        return cls(
            candidate=Candidate.from_dict(data),
            breakdown=ScoreBreakdown.from_dict(data.get("score_breakdown", {}) or {}),
            score=float(data.get("score", 0.0) or 0.0),
            reason=str(data.get("score_reason", "")),
        )


# --------------------------------------------------------------------------
# Hasil render
# --------------------------------------------------------------------------
@dataclass
class ClipResult:
    """Satu klip jadi: file video, thumbnail, dan metadata unggahnya."""

    index: int
    video_path: str
    source_start: float
    source_end: float
    duration: float
    score: float
    hook_text: str
    title: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    thumbnail_path: str = ""
    subtitle_path: str = ""
    transcript_text: str = ""
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    reframe_mode: str = ""
    width: int = 0
    height: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("source_start", "source_end", "duration", "score"):
            data[key] = _round(data[key], 2)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClipResult":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class JobResult:
    """Ringkasan satu job (satu video sumber)."""

    source: str
    video_path: str
    output_dir: str
    clips: List[ClipResult] = field(default_factory=list)
    transcript_path: str = ""
    metadata_json: str = ""
    metadata_csv: str = ""
    language: str = ""
    duration: float = 0.0
    best_score: float = 0.0
    candidate_count: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "video_path": self.video_path,
            "output_dir": self.output_dir,
            "language": self.language,
            "duration": _round(self.duration, 2),
            "best_score": _round(self.best_score, 2),
            "candidate_count": self.candidate_count,
            "clip_count": len(self.clips),
            "transcript_path": self.transcript_path,
            "metadata_json": self.metadata_json,
            "metadata_csv": self.metadata_csv,
            "warnings": list(self.warnings),
            "clips": [c.to_dict() for c in self.clips],
        }


def sort_by_score(items: Iterable[ScoredCandidate]) -> List[ScoredCandidate]:
    """Urutkan dari skor tertinggi; kalau seri, yang lebih awal menang."""

    return sorted(items, key=lambda item: (-item.score, item.start))


def as_sequence(items: Optional[Sequence[Any]]) -> List[Any]:
    return list(items) if items else []
