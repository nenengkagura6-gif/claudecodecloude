"""Kontrak dasar provider LLM: satu metode ``complete()`` + retry berjenjang."""

from __future__ import annotations

import abc
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..errors import LLMError
from ..logging_utils import get_logger

log = get_logger("llm")


class RetryableLLMError(LLMError):
    """Error sementara (jaringan, rate limit, 5xx) - layak dicoba ulang."""


@dataclass
class LLMResult:
    """Balasan LLM beserta info pemakaian bila tersedia."""

    text: str
    provider: str = ""
    model: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0

    def __str__(self) -> str:  # supaya bisa dipakai seperti string biasa
        return self.text


class LLMProvider(abc.ABC):
    """Induk semua provider.

    Turunan hanya perlu mengisi :meth:`_request`. Retry, backoff, logging, dan
    pemeriksaan balasan kosong ditangani di sini.
    """

    name: str = "base"
    #: apakah provider ini butuh kunci API
    requires_api_key: bool = True

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 4000,
        timeout_seconds: float = 120,
        max_retries: int = 3,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = str(model)
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/") or None
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        self.extra = dict(extra or {})

    # ------------------------------------------------------------------ api
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        """Kirim prompt, kembalikan teks balasan. Melempar :class:`LLMError`."""

        temperature = self.temperature if temperature is None else float(temperature)
        max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._request(
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not result.text.strip():
                    raise RetryableLLMError("Balasan LLM kosong.")
                log.debug(
                    "%s/%s menjawab %d karakter (percobaan %d)",
                    self.name, self.model, len(result.text), attempt,
                )
                return result
            except RetryableLLMError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(30.0, (2 ** (attempt - 1))) + random.uniform(0, 0.4)
                log.warning(
                    "LLM %s gagal (%s). Coba ulang %d/%d dalam %.1fs",
                    self.name, exc, attempt, self.max_retries, delay,
                )
                time.sleep(delay)
            except LLMError:
                raise
            except Exception as exc:  # pragma: no cover - jaring pengaman
                raise LLMError(f"Provider {self.name} error tak terduga: {exc}") from exc

        raise LLMError(
            f"Provider {self.name} gagal setelah {self.max_retries} percobaan: "
            f"{last_error}"
        )

    @abc.abstractmethod
    def _request(
        self,
        *,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        """Panggilan HTTP sesungguhnya (diimplementasikan tiap provider)."""

    # --------------------------------------------------------------- helper
    def describe(self) -> str:
        return f"{self.name}:{self.model}"
