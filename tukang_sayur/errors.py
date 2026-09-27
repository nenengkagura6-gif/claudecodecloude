"""Kumpulan error khusus Tukang Sayur.

Semua tahap pipeline memakai turunan :class:`TukangSayurError` supaya CLI bisa
menampilkan pesan yang enak dibaca (bukan traceback panjang) dan batch mode
bisa memutuskan lanjut atau berhenti.
"""

from __future__ import annotations


class TukangSayurError(Exception):
    """Induk semua error yang kita lempar sendiri."""

    #: nama tahap pipeline, dipakai untuk log & state batch
    stage = "umum"


class ConfigError(TukangSayurError):
    stage = "config"


class DependencyError(TukangSayurError):
    """Program luar / paket Python yang dibutuhkan tidak ada."""

    stage = "dependency"


class IngestError(TukangSayurError):
    stage = "ingest"


class TranscriptionError(TukangSayurError):
    stage = "transcribe"


class LLMError(TukangSayurError):
    stage = "llm"


class LLMParseError(LLMError):
    """Balasan LLM tidak bisa dipahami sebagai JSON yang kita minta."""

    stage = "llm-parse"


class CandidateError(TukangSayurError):
    stage = "candidates"


class ScoringError(TukangSayurError):
    stage = "scoring"


class ReframeError(TukangSayurError):
    stage = "reframe"


class RenderError(TukangSayurError):
    stage = "render"
