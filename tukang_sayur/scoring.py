"""Tahap 4 - Viral Scoring Judge: skor 0-100 per kandidat.

Empat dimensi (hook, emotion, clarity, standalone) digabung dengan bobot dari
``config.yaml`` menjadi satu skor akhir.

Soal ``best_score``
-------------------
Bug klasik pada tool sejenis: ``best_score`` selalu 0 karena kode membaca
properti yang salah (misalnya ``item["score"]`` padahal LLM mengirim
``item["scores"]["total"]``, atau membaca ``item.total`` padahal atributnya
bernama ``score``). Cara modul ini mencegahnya:

1. Nama atribut skor akhir hanya satu: :data:`~tukang_sayur.models.SCORE_ATTR`
   (``"score"``), dipakai bersama oleh scoring, pipeline, dan metadata.
2. :func:`get_score` menerima objek maupun dict, mencoba beberapa nama yang
   wajar, dan **melempar error** kalau tidak menemukan apa pun - tidak pernah
   diam-diam mengembalikan 0.
3. :func:`repair_zero_scores` menyalakan lampu merah kalau skor akhir 0 padahal
   rincian dimensinya jelas-jelas positif, lalu menghitung ulang.
4. Ada unit test khusus di ``tests/test_scoring.py`` untuk ketiga hal di atas.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import Config
from .errors import LLMError, ScoringError
from .llm.base import LLMProvider
from .llm.parsing import parse_scores
from .logging_utils import get_logger
from .models import (
    SCORE_ATTR,
    SCORE_DIMENSIONS,
    Candidate,
    ScoreBreakdown,
    ScoredCandidate,
    sort_by_score,
)
from .prompts import SYSTEM_JUDGE, judge_prompt

log = get_logger("scoring")

#: Nama alternatif skor akhir yang kita terima saat membaca data luar/lama.
SCORE_KEY_ALIASES = (SCORE_ATTR, "best_score", "total", "total_score", "final_score",
                     "overall", "viral_score", "skor")

_EMOTION_WORDS = {
    "kaget", "gila", "parah", "nangis", "sedih", "marah", "takut", "lucu", "ngeri",
    "bangga", "kecewa", "terharu", "syok", "heran", "bahaya", "rahasia", "jangan",
    "ternyata", "gagal", "sukses", "cinta", "benci", "malu", "bohong", "jujur",
    "shock", "anjir", "wow", "banget", "bikin", "kesel", "seram", "hebat",
}
_CURIOSITY_WORDS = {
    "kenapa", "gimana", "bagaimana", "kok", "rahasia", "trik", "cara", "ternyata",
    "jangan", "stop", "hati-hati", "awas", "ini", "inilah", "penting", "salah",
}
_WEAK_STARTERS = {
    "tapi", "namun", "jadi", "dan", "atau", "terus", "lalu", "kemudian", "nah",
    "karena", "soalnya", "makanya", "yang", "kalau", "sedangkan", "padahal",
}


# --------------------------------------------------------------------------
# Pembaca skor yang tidak pernah menebak 0
# --------------------------------------------------------------------------
def get_score(item: Any) -> float:
    """Ambil skor akhir dari :class:`ScoredCandidate`, objek lain, atau dict.

    Melempar :class:`ScoringError` kalau tidak ada jalur properti yang cocok -
    lebih baik berhenti dan berisik daripada melaporkan 0 yang menyesatkan.
    """

    if isinstance(item, (int, float)) and not isinstance(item, bool):
        return float(item)

    if isinstance(item, Mapping):
        for key in SCORE_KEY_ALIASES:
            if key in item and isinstance(item[key], (int, float)) and not isinstance(
                item[key], bool
            ):
                return float(item[key])
        breakdown = item.get("score_breakdown") or item.get("breakdown")
        if isinstance(breakdown, Mapping):
            values = [
                float(breakdown[name])
                for name in SCORE_DIMENSIONS
                if isinstance(breakdown.get(name), (int, float))
            ]
            if values:
                return sum(values) / len(values)
        raise ScoringError(
            f"Tidak menemukan skor pada dict dengan kunci: {sorted(item)[:10]}"
        )

    for attribute in SCORE_KEY_ALIASES:
        value = getattr(item, attribute, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    breakdown = getattr(item, "breakdown", None)
    if isinstance(breakdown, ScoreBreakdown):
        values = [getattr(breakdown, name) for name in SCORE_DIMENSIONS]
        if any(values):
            return sum(values) / len(values)

    raise ScoringError(
        f"Objek {type(item).__name__} tidak punya atribut skor "
        f"(dicoba: {', '.join(SCORE_KEY_ALIASES)})."
    )


def best_score(items: Iterable[Any]) -> float:
    """Skor tertinggi dari kumpulan kandidat. Kosong -> 0.0."""

    values = [get_score(item) for item in items]
    return round(max(values), 2) if values else 0.0


def average_score(items: Iterable[Any]) -> float:
    values = [get_score(item) for item in items]
    return round(sum(values) / len(values), 2) if values else 0.0


def repair_zero_scores(
    items: Sequence[ScoredCandidate], weights: Mapping[str, float]
) -> List[ScoredCandidate]:
    """Hitung ulang skor yang 0 padahal rincian dimensinya positif.

    Ini jaring pengaman terakhir untuk bug "best_score selalu 0".
    """

    repaired: List[ScoredCandidate] = []
    for item in items:
        dimension_values = [getattr(item.breakdown, name) for name in SCORE_DIMENSIONS]
        if item.score <= 0 and any(value > 0 for value in dimension_values):
            recomputed = item.breakdown.weighted_total(weights)
            log.error(
                "Skor kandidat #%d terbaca 0 padahal rinciannya %s -> dihitung "
                "ulang jadi %.2f (cek pemetaan properti skor!)",
                item.candidate.index, item.breakdown.to_dict(), recomputed,
            )
            item.score = recomputed
        repaired.append(item)
    return repaired


# --------------------------------------------------------------------------
# Penilaian cadangan tanpa LLM
# --------------------------------------------------------------------------
def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^\w']+", str(text).lower()) if t]


def heuristic_breakdown(candidate: Candidate, cfg: Config) -> ScoreBreakdown:
    """Penilaian berbasis aturan sederhana - dipakai kalau LLM gagal.

    Nilainya sengaja dijaga di rentang 25-85 supaya jelas "ini perkiraan",
    dan tidak pernah 0 (0 akan tampak seperti kandidat rusak).
    """

    min_duration = float(cfg.get("candidates.min_duration", 20))
    max_duration = float(cfg.get("candidates.max_duration", 90))
    ideal = (min_duration + max_duration) / 2.0

    words = _tokens(candidate.transcript_text)
    hook_words = _tokens(candidate.hook_text)
    duration = max(1e-6, candidate.duration)
    wps = len(words) / duration

    # clarity: kepadatan bicara wajar (2-3.5 kata/detik) dan durasi dekat ideal
    density_score = 100.0 - min(100.0, abs(wps - 2.7) * 45.0)
    duration_score = 100.0 - min(100.0, abs(duration - ideal) / max(1.0, ideal) * 90.0)
    clarity = 0.6 * density_score + 0.4 * duration_score

    # hook: panjang hook pas (3-8 kata), ada angka/pertanyaan/kata pemancing
    hook_length = len(hook_words)
    hook = 55.0 if 3 <= hook_length <= 8 else 35.0
    if any(word in _CURIOSITY_WORDS for word in hook_words):
        hook += 18.0
    if any(char.isdigit() for char in candidate.hook_text):
        hook += 8.0
    if "?" in candidate.hook_text:
        hook += 6.0

    # emotion: kerapatan kata bermuatan emosi
    emotional = sum(1 for word in words if word in _EMOTION_WORDS)
    emotion = 35.0 + min(45.0, emotional * 9.0)

    # standalone: tidak dibuka kata sambung, dan diakhiri kalimat utuh
    standalone = 60.0
    if words and words[0] in _WEAK_STARTERS:
        standalone -= 22.0
    if candidate.transcript_text.strip().endswith((".", "!", "?", "…")):
        standalone += 12.0
    if len(words) < 25:
        standalone -= 12.0

    clamp = lambda value: max(25.0, min(85.0, value))  # noqa: E731
    return ScoreBreakdown(
        hook=clamp(hook),
        emotion=clamp(emotion),
        clarity=clamp(clarity),
        standalone=clamp(standalone),
    )


def heuristic_scores(
    candidates: Sequence[Candidate], cfg: Config
) -> List[ScoredCandidate]:
    weights = dict(cfg.get("scoring.weights", {}) or {})
    out: List[ScoredCandidate] = []
    for candidate in candidates:
        breakdown = heuristic_breakdown(candidate, cfg)
        out.append(
            ScoredCandidate(
                candidate=candidate,
                breakdown=breakdown,
                score=breakdown.weighted_total(weights),
                reason="Skor perkiraan (tanpa LLM).",
            )
        )
    return out


# --------------------------------------------------------------------------
# Juri LLM
# --------------------------------------------------------------------------
def score_candidates(
    candidates: Sequence[Candidate],
    cfg: Config,
    provider: Optional[LLMProvider],
    *,
    language: str = "id",
    allow_fallback: bool = True,
) -> List[ScoredCandidate]:
    """Nilai semua kandidat. Kalau LLM gagal, pakai heuristik (bukan 0)."""

    if not candidates:
        return []

    weights = dict(cfg.get("scoring.weights", {}) or {})

    if provider is None:
        log.warning("Tidak ada provider LLM; memakai skor heuristik.")
        return heuristic_scores(candidates, cfg)

    prompt = judge_prompt(candidates, language=language)
    log.info("Menilai %d kandidat lewat %s ...", len(candidates), provider.describe())
    try:
        result = provider.complete(prompt, system=SYSTEM_JUDGE)
        scored = parse_scores(result.text, candidates, weights)
    except (LLMError, ScoringError) as exc:
        if not allow_fallback:
            raise ScoringError(f"Penilaian gagal: {exc}") from exc
        log.warning("Penilaian LLM gagal (%s); memakai skor heuristik.", exc)
        return heuristic_scores(candidates, cfg)

    scored = repair_zero_scores(scored, weights)

    # Kandidat yang tidak dinilai LLM tetap ikut, dengan skor heuristik.
    scored_starts = {round(item.candidate.start, 2) for item in scored}
    missing = [c for c in candidates if round(c.start, 2) not in scored_starts]
    if missing:
        log.info("%d kandidat dinilai heuristik karena dilewati LLM.", len(missing))
        scored.extend(heuristic_scores(missing, cfg))

    highest = best_score(scored)
    if highest <= 0:
        # Seharusnya tidak mungkin sampai sini; kalau iya, jangan diamkan.
        log.error(
            "Semua skor 0 setelah penilaian. Beralih ke heuristik supaya "
            "best_score tetap bermakna."
        )
        return heuristic_scores(candidates, cfg)

    log.info(
        "Penilaian selesai: tertinggi %.1f, rata-rata %.1f",
        highest, average_score(scored),
    )
    return sort_by_score(scored)


# --------------------------------------------------------------------------
# Pemilihan klip
# --------------------------------------------------------------------------
def select_clips(
    scored: Sequence[ScoredCandidate],
    cfg: Config,
    *,
    max_clips: Optional[int] = None,
) -> List[ScoredCandidate]:
    """Ambil klip terbaik: urut skor, saring skor minimum, batasi jumlah."""

    if not scored:
        return []

    limit = int(max_clips or cfg.get("candidates.max_clips", 15))
    limit = max(1, limit)
    minimum = float(cfg.get("scoring.min_score", 0) or 0)

    ranked = sort_by_score(scored)
    passing = [item for item in ranked if item.score >= minimum]
    if not passing:
        fallback = ranked[: min(limit, len(ranked))]
        log.warning(
            "Tidak ada klip yang mencapai scoring.min_score=%.0f "
            "(tertinggi %.1f). Mengambil %d klip terbaik apa adanya.",
            minimum, ranked[0].score, len(fallback),
        )
        passing = fallback

    selected = passing[:limit]
    log.info(
        "Klip terpilih: %d dari %d kandidat (skor %.1f - %.1f)",
        len(selected), len(scored), selected[-1].score, selected[0].score,
    )
    return selected


def scoring_summary(scored: Sequence[ScoredCandidate]) -> Dict[str, Any]:
    return {
        "count": len(scored),
        "best_score": best_score(scored),
        "average_score": average_score(scored),
        "dimensions": {
            name: round(
                sum(getattr(item.breakdown, name) for item in scored) / len(scored), 1
            )
            for name in SCORE_DIMENSIONS
        }
        if scored
        else {},
    }
