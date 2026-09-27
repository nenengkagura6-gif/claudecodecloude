"""Tukang Sayur - pemotong video otomatis jadi klip vertikal 9:16.

Alur singkat::

    ingest -> transcribe -> candidates (LLM) -> scoring (LLM)
           -> reframe 9:16 -> render (ASS + hook) -> metadata

Pemakaian paling umum::

    python -m tukang_sayur clip <url|file> --max-clips 10
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
