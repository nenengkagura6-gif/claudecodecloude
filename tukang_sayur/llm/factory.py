"""Pemilih provider berdasarkan config."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from ..config import Config
from ..errors import ConfigError
from ..logging_utils import get_logger
from .base import LLMProvider
from .providers import (
    AnthropicProvider,
    GeminiProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
)

log = get_logger("llm.factory")

PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "ollama": OllamaProvider,
    "local": OllamaProvider,
    "mock": MockProvider,
}


def available_providers() -> List[str]:
    return ["anthropic", "openai", "gemini", "ollama", "mock"]


def build_provider(cfg: Config, name: Optional[str] = None) -> LLMProvider:
    """Bikin provider sesuai ``llm.provider`` (atau paksa lewat ``name``)."""

    provider_name = (name or cfg.provider_name()).strip().lower()
    provider_cls = PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ConfigError(
            f"Provider LLM '{provider_name}' tidak dikenal. "
            f"Pilihan: {', '.join(available_providers())}"
        )

    conf = cfg.provider_config(provider_name)
    api_key = cfg.api_key(provider_name)
    if provider_cls.requires_api_key and not api_key:
        env_name = conf.get("api_key_env") or "API_KEY"
        raise ConfigError(
            f"Provider '{provider_name}' butuh kunci API, tapi {env_name} kosong.\n"
            "Buat file .env (contoh ada di .env.example) lalu isi kuncinya, "
            "atau pakai provider lokal: llm.provider: ollama"
        )

    known = {
        "model", "api_key_env", "base_url", "temperature", "max_tokens",
        "timeout_seconds", "max_retries", "name",
    }
    provider = provider_cls(
        model=str(conf.get("model") or ""),
        api_key=api_key,
        base_url=conf.get("base_url"),
        temperature=float(conf.get("temperature", 0.4)),
        max_tokens=int(conf.get("max_tokens", 4000)),
        timeout_seconds=float(conf.get("timeout_seconds", 120)),
        max_retries=int(conf.get("max_retries", 3)),
        extra={k: v for k, v in conf.items() if k not in known},
    )
    log.info("Provider LLM: %s", provider.describe())
    return provider
