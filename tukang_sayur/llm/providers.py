"""Implementasi provider: Anthropic, OpenAI (& yang kompatibel), Gemini, Ollama, Mock.

Semua lewat HTTP langsung dengan ``requests`` supaya tidak perlu memasang SDK
tiap vendor. Ganti provider cukup dari ``config.yaml``::

    llm:
      provider: ollama    # anthropic | openai | gemini | ollama | mock
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..errors import LLMError
from ..logging_utils import get_logger
from .base import LLMProvider, LLMResult, RetryableLLMError

log = get_logger("llm.provider")

ANTHROPIC_VERSION = "2023-06-01"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _requests():
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise LLMError(
            "Paket 'requests' belum terpasang. Install: pip install requests"
        ) from exc
    return requests


def _post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
    *,
    provider: str,
) -> Dict[str, Any]:
    """POST JSON dengan pemetaan error yang jelas (retryable vs fatal)."""

    requests = _requests()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise RetryableLLMError(f"{provider}: timeout setelah {timeout:.0f}s") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RetryableLLMError(f"{provider}: gagal menyambung ({exc})") from exc
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"{provider}: permintaan gagal ({exc})") from exc

    if response.status_code in _RETRY_STATUS:
        raise RetryableLLMError(
            f"{provider}: HTTP {response.status_code} - {_short(response.text)}"
        )
    if response.status_code == 401 or response.status_code == 403:
        raise LLMError(
            f"{provider}: kunci API ditolak (HTTP {response.status_code}). "
            "Periksa isi file .env."
        )
    if response.status_code == 404:
        raise LLMError(
            f"{provider}: endpoint/model tidak ditemukan (HTTP 404) di {url}. "
            "Periksa nama model dan base_url di config.yaml."
        )
    if response.status_code >= 400:
        raise LLMError(
            f"{provider}: HTTP {response.status_code} - {_short(response.text)}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise LLMError(f"{provider}: balasan bukan JSON - {_short(response.text)}") from exc


def _short(text: Optional[str], limit: int = 400) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:limit]


# --------------------------------------------------------------------------
class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def _request(self, *, prompt, system, temperature, max_tokens) -> LLMResult:
        if not self.api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY belum diisi. Salin .env.example jadi .env lalu "
                "isi kuncinya."
            )
        base = self.base_url or "https://api.anthropic.com"
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        data = _post_json(
            f"{base}/v1/messages",
            payload,
            {
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            self.timeout_seconds,
            provider=self.name,
        )
        parts = [
            block.get("text", "")
            for block in data.get("content", []) or []
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        usage = data.get("usage") or {}
        return LLMResult(
            text="".join(parts),
            provider=self.name,
            model=str(data.get("model") or self.model),
            raw=data,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )


# --------------------------------------------------------------------------
class OpenAIProvider(LLMProvider):
    """OpenAI dan semua server yang kompatibel (LM Studio, vLLM, OpenRouter)."""

    name = "openai"

    def _request(self, *, prompt, system, temperature, max_tokens) -> LLMResult:
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY belum diisi (lihat file .env).")
        base = self.base_url or "https://api.openai.com/v1"
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            data = _post_json(
                f"{base}/chat/completions", payload, headers, self.timeout_seconds,
                provider=self.name,
            )
        except LLMError as exc:
            # Model keluarga baru menolak "max_tokens" -> coba parameter penggantinya.
            if "max_tokens" not in str(exc):
                raise
            log.debug("Model %s menolak max_tokens, memakai max_completion_tokens.", self.model)
            payload.pop("max_tokens", None)
            payload["max_completion_tokens"] = max_tokens
            data = _post_json(
                f"{base}/chat/completions", payload, headers, self.timeout_seconds,
                provider=self.name,
            )

        choices = data.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, list):  # sebagian server balas berupa list blok
                text = "".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            else:
                text = str(content or "")
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            provider=self.name,
            model=str(data.get("model") or self.model),
            raw=data,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )


# --------------------------------------------------------------------------
class GeminiProvider(LLMProvider):
    name = "gemini"

    def _request(self, *, prompt, system, temperature, max_tokens) -> LLMResult:
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY belum diisi (lihat file .env).")
        base = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        data = _post_json(
            f"{base}/models/{self.model}:generateContent",
            payload,
            {"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            self.timeout_seconds,
            provider=self.name,
        )
        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(str(part.get("text", "")) for part in parts)
        usage = data.get("usageMetadata") or {}
        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            raw=data,
            input_tokens=int(usage.get("promptTokenCount") or 0),
            output_tokens=int(usage.get("candidatesTokenCount") or 0),
        )


# --------------------------------------------------------------------------
class OllamaProvider(LLMProvider):
    """LLM lokal via Ollama - gratis, tanpa kunci API, jalan offline."""

    name = "ollama"
    requires_api_key = False

    def _request(self, *, prompt, system, temperature, max_tokens) -> LLMResult:
        base = self.base_url or "http://localhost:11434"
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            data = _post_json(
                f"{base}/api/chat", payload, headers, self.timeout_seconds,
                provider=self.name,
            )
        except RetryableLLMError as exc:
            if "menyambung" in str(exc):
                raise LLMError(
                    f"Tidak bisa menyambung ke Ollama di {base}. Jalankan dulu "
                    f"'ollama serve' lalu 'ollama pull {self.model}'."
                ) from exc
            raise
        message = data.get("message") or {}
        return LLMResult(
            text=str(message.get("content") or ""),
            provider=self.name,
            model=str(data.get("model") or self.model),
            raw=data,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )


# --------------------------------------------------------------------------
_LINE_RE = re.compile(r"\[(\d+(?:\.\d+)?)\s*->\s*(\d+(?:\.\d+)?)\]\s*(.+)")


class MockProvider(LLMProvider):
    """Provider tiruan: tanpa jaringan, balasan bisa diprediksi.

    Dia membaca baris transkrip berformat ``[12.0 -> 18.5] teks`` dari prompt,
    lalu menyusun JSON yang bentuknya sama seperti balasan LLM sungguhan.
    Dipakai untuk test end-to-end dan demo offline.
    """

    name = "mock"
    requires_api_key = False

    def __init__(self, model: str = "mock", **kwargs: Any) -> None:
        super().__init__(model or "mock", **kwargs)

    # -- pembantu ---------------------------------------------------------
    @staticmethod
    def _lines(prompt: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for match in _LINE_RE.finditer(prompt):
            start, end, text = match.groups()
            out.append({"start": float(start), "end": float(end), "text": text.strip()})
        return out

    @staticmethod
    def _task(prompt: str) -> str:
        match = re.search(r"TASK:\s*([a-z_]+)", prompt)
        if match:
            return match.group(1)
        low = prompt.lower()
        if "skor" in low or "score" in low:
            return "score_candidates"
        if "metadata" in low or "caption" in low:
            return "write_metadata"
        return "find_candidates"

    def _candidates(self, prompt: str) -> Dict[str, Any]:
        lines = self._lines(prompt)
        target_min = 20.0
        target_max = 90.0
        match = re.search(r"durasi\s+(\d+)\s*-\s*(\d+)\s*detik", prompt)
        if match:
            target_min, target_max = float(match.group(1)), float(match.group(2))
        wanted = 5
        match = re.search(r"(\d+)\s*(?:-\s*\d+\s*)?kandidat", prompt)
        if match:
            wanted = max(1, int(match.group(1)))

        candidates: List[Dict[str, Any]] = []
        index = 0
        while index < len(lines) and len(candidates) < wanted:
            start = lines[index]["start"]
            end = lines[index]["end"]
            texts = [lines[index]["text"]]
            cursor = index + 1
            while cursor < len(lines) and (end - start) < target_min:
                end = lines[cursor]["end"]
                texts.append(lines[cursor]["text"])
                cursor += 1
            if (end - start) > target_max:
                end = start + target_max
            if (end - start) >= min(target_min, 1.0):
                hook = " ".join(" ".join(texts).split()[:6]) or "Simak ini"
                candidates.append(
                    {
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "hook_text": hook,
                        "reason": "Bagian ini paling padat informasi (mock).",
                    }
                )
            index = max(cursor, index + 1)

        if not candidates:
            candidates.append(
                {
                    "start": 0.0,
                    "end": target_min,
                    "hook_text": "Klip contoh dari mock",
                    "reason": "Tidak ada baris transkrip di prompt (mock).",
                }
            )
        return {"candidates": candidates}

    @staticmethod
    def _scores(prompt: str) -> Dict[str, Any]:
        indices = [int(m) for m in re.findall(r"^\s*(?:#|KANDIDAT\s*)(\d+)", prompt, re.M)]
        if not indices:
            indices = list(range(1, len(MockProvider._lines(prompt)) + 1)) or [1]
        scores = []
        for position, index in enumerate(sorted(set(indices))):
            base = 88 - (position * 3)
            scores.append(
                {
                    "index": index,
                    "hook": max(30, base),
                    "emotion": max(30, base - 4),
                    "clarity": max(30, base - 2),
                    "standalone": max(30, base - 6),
                    "reason": "Penilaian tiruan untuk test.",
                }
            )
        return {"scores": scores}

    @staticmethod
    def _metadata(prompt: str) -> Dict[str, Any]:
        hooks = re.findall(r"HOOK:\s*(.+)", prompt)
        items = []
        for position, hook in enumerate(hooks or ["Klip menarik"], start=1):
            hook = hook.strip()
            items.append(
                {
                    "index": position,
                    "title": hook[:60],
                    "caption": f"{hook} Tonton sampai habis!",
                    "hashtags": ["fyp", "viral", "shorts", "reels", "tiktok"],
                }
            )
        return {"items": items}

    def _request(self, *, prompt, system, temperature, max_tokens) -> LLMResult:
        task = self._task(prompt)
        if task == "score_candidates":
            payload: Dict[str, Any] = self._scores(prompt)
        elif task == "write_metadata":
            payload = self._metadata(prompt)
        else:
            payload = self._candidates(prompt)
        return LLMResult(
            text=json.dumps(payload, ensure_ascii=False),
            provider=self.name,
            model=self.model,
            raw={"task": task},
        )
