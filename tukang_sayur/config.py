"""Pemuat konfigurasi: default bawaan + config.yaml + .env + override CLI.

Prioritas (yang belakangan menang)::

    DEFAULT_CONFIG  <  config.yaml  <  override dari CLI/UI

Kunci API **tidak pernah** ditulis di config.yaml. Dia dibaca dari file ``.env``
atau environment variable, lewat :func:`load_dotenv` dan :meth:`Config.api_key`.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml

from .errors import ConfigError
from .logging_utils import get_logger

log = get_logger("config")

#: Semua default. config.yaml boleh hanya menimpa sebagian.
DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "work_dir": "work",
        "output_dir": "output",
        "cache_dir": ".cache",
        "keep_intermediate": False,
    },
    "ingest": {
        "ytdlp_format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "cookies_from_browser": None,
        "max_duration_minutes": 180,
        "normalize": {
            "fps": 30,
            "max_height": 1080,
            "audio_sample_rate": 48000,
            "audio_channels": 2,
        },
    },
    "transcription": {
        "engine": "faster-whisper",
        "model": "small",
        "compute_type": "int8",
        "device": "cpu",
        "cpu_threads": 4,
        "beam_size": 5,
        "language": "id",
        "vad_filter": True,
        "condition_on_previous_text": False,
        "cache": True,
    },
    "llm": {
        "provider": "anthropic",
        "temperature": 0.4,
        "max_tokens": 4000,
        "timeout_seconds": 120,
        "max_retries": 3,
        "providers": {
            "anthropic": {
                "model": "claude-sonnet-5",
                "api_key_env": "ANTHROPIC_API_KEY",
                "base_url": "https://api.anthropic.com",
            },
            "openai": {
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
            },
            "gemini": {
                "model": "gemini-2.0-flash",
                "api_key_env": "GEMINI_API_KEY",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
            },
            "ollama": {
                "model": "llama3.1:8b",
                "base_url": "http://localhost:11434",
                "api_key_env": None,
            },
            "mock": {"model": "mock", "api_key_env": None, "base_url": None},
        },
    },
    "candidates": {
        "min_clips": 5,
        "max_clips": 15,
        "min_duration": 20,
        "max_duration": 90,
        "snap_to_sentence": True,
        "max_hook_words": 8,
        "padding_start": 0.25,
        "padding_end": 0.35,
        "max_overlap_ratio": 0.35,
    },
    "scoring": {
        "weights": {"hook": 0.35, "emotion": 0.25, "clarity": 0.20, "standalone": 0.20},
        "min_score": 55,
    },
    "reframe": {
        "enabled": True,
        "detector": "auto",
        "sample_fps": 5,
        "smoothing_window": 9,
        "ema_alpha": 0.25,
        "deadzone_px": 24,
        "keyframe_interval": 0.4,
        "fallback": "blur",
        "blur_strength": 40,
        "min_face_confidence": 0.5,
    },
    "subtitle": {
        "enabled": True,
        "font_name": "Montserrat",
        "font_size": 64,
        "primary_color": "&H00FFFFFF",
        "highlight_color": "&H0000E5FF",
        "outline_color": "&H00000000",
        "back_color": "&H80000000",
        "outline": 4,
        "shadow": 1,
        "bold": True,
        "uppercase": True,
        "max_chars_per_line": 22,
        "max_words_per_line": 4,
        "margin_bottom": 420,
        "margin_horizontal": 90,
        "karaoke": True,
        "border_style": 1,
        "alignment": 2,
    },
    "hook": {
        "enabled": True,
        "duration": 3.0,
        "font_name": "Montserrat",
        "font_size": 78,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&HB0000000",
        "outline": 5,
        "shadow": 0,
        "bold": True,
        "uppercase": True,
        "max_chars_per_line": 18,
        "border_style": 3,
        "alignment": 8,
        "margin_top": 240,
        "margin_horizontal": 80,
        "fade_in": 0.25,
        "fade_out": 0.35,
    },
    "audio": {
        "remove_silence": False,
        "silence_threshold_db": -34,
        "min_silence_duration": 0.45,
        "keep_padding": 0.08,
        "loudnorm": True,
        "target_lufs": -14,
        "true_peak": -1.5,
    },
    "render": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "video_codec": "libx264",
        "preset": "veryfast",
        "crf": 20,
        "audio_codec": "aac",
        "audio_bitrate": "160k",
        "threads": 0,
        "thumbnail": True,
        "thumbnail_offset_ratio": 0.25,
    },
    "watermark": {
        "enabled": False,
        "path": "assets/logo.png",
        "position": "top-right",
        "opacity": 0.75,
        "scale_width": 160,
        "margin": 40,
    },
    "metadata": {
        "use_llm": True,
        "hashtag_count": 8,
        "language": "id",
        "json_filename": "metadata.json",
        "csv_filename": "metadata.csv",
    },
    "batch": {"state_file": "batch_state.json", "continue_on_error": True},
    "server": {"host": "0.0.0.0", "port": 8000, "api_token_env": "TUKANG_SAYUR_TOKEN"},
    "ui": {"host": "127.0.0.1", "port": 7860, "share": False},
    "logging": {"level": "INFO", "file": "logs/tukang_sayur.log"},
}

_TRUE = {"1", "true", "yes", "y", "on", "ya"}
_FALSE = {"0", "false", "no", "n", "off", "tidak"}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Gabung dua dict secara rekursif tanpa mengubah input."""

    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_env_file(text: str) -> Dict[str, str]:
    """Parser .env sederhana (KEY=VALUE, komentar ``#``, kutip opsional)."""

    values: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value[:1] in ("\"", "'"):
            # nilai dikutip: ambil sampai kutip penutup, sisanya (komentar) dibuang
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            # buang komentar di akhir baris yang tidak dikutip
            hash_pos = value.find(" #")
            if hash_pos != -1:
                value = value[:hash_pos].rstrip()
        if key:
            values[key] = value
    return values


def load_dotenv(
    path: Optional[str | os.PathLike[str]] = None, *, override: bool = False
) -> Dict[str, str]:
    """Baca file .env ke ``os.environ``.

    Mencari ``.env`` di direktori kerja lalu naik ke induknya (maks. 4 level),
    supaya jalan baik dari root proyek maupun dari subfolder.
    """

    candidates: Iterable[Path]
    if path is not None:
        candidates = [Path(path)]
    else:
        cwd = Path.cwd()
        candidates = [cwd, *list(cwd.parents)[:4]]
        candidates = [p / ".env" for p in candidates]

    for candidate in candidates:
        if candidate.is_file():
            values = parse_env_file(candidate.read_text(encoding="utf-8"))
            applied = {}
            for key, value in values.items():
                if override or key not in os.environ:
                    os.environ[key] = value
                    applied[key] = value
            log.debug("Memuat %d nilai dari %s", len(values), candidate)
            return values
    log.debug("File .env tidak ditemukan, pakai environment yang ada.")
    return {}


def _coerce_scalar(text: str) -> Any:
    """Ubah string override CLI jadi tipe yang wajar."""

    low = text.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    if low in {"none", "null", ""}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


@dataclass
class Config:
    """Wadah konfigurasi dengan akses bertitik: ``cfg.get("render.width")``."""

    data: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))
    path: Optional[Path] = None

    # ---------------------------------------------------------------- load
    @classmethod
    def load(
        cls,
        path: Optional[str | os.PathLike[str]] = None,
        *,
        overrides: Optional[Mapping[str, Any]] = None,
        load_env: bool = True,
    ) -> "Config":
        """Muat config.yaml (kalau ada) di atas :data:`DEFAULT_CONFIG`."""

        if load_env:
            load_dotenv()

        data = copy.deepcopy(DEFAULT_CONFIG)
        used_path: Optional[Path] = None

        candidate = Path(path) if path else Path("config.yaml")
        if candidate.is_file():
            try:
                loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"config.yaml tidak valid: {exc}") from exc
            if not isinstance(loaded, Mapping):
                raise ConfigError("Isi config.yaml harus berupa mapping (key: value).")
            data = deep_merge(data, loaded)
            used_path = candidate
            log.debug("Konfigurasi dimuat dari %s", candidate)
        elif path:
            raise ConfigError(f"File konfigurasi tidak ditemukan: {candidate}")
        else:
            log.debug("config.yaml tidak ada, memakai default bawaan.")

        cfg = cls(data=data, path=used_path)
        if overrides:
            cfg.apply_overrides(overrides)
        cfg.validate()
        return cfg

    # ------------------------------------------------------------- access
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, MutableMapping):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def section(self, name: str) -> Dict[str, Any]:
        value = self.get(name, {})
        return dict(value) if isinstance(value, Mapping) else {}

    def apply_overrides(self, overrides: Mapping[str, Any]) -> None:
        """Terapkan override ``{"render.crf": 18}``; nilai ``None`` diabaikan."""

        for dotted, value in overrides.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = _coerce_scalar(value)
            self.set(dotted, value)

    @staticmethod
    def parse_cli_set(items: Iterable[str]) -> Dict[str, Any]:
        """Ubah ``["render.crf=18", "audio.loudnorm=false"]`` jadi dict."""

        out: Dict[str, Any] = {}
        for item in items:
            if "=" not in item:
                raise ConfigError(f"--set harus berbentuk kunci=nilai, dapat: {item!r}")
            key, _, value = item.partition("=")
            out[key.strip()] = _coerce_scalar(value)
        return out

    # -------------------------------------------------------------- paths
    @property
    def work_dir(self) -> Path:
        return Path(self.get("project.work_dir", "work"))

    @property
    def output_dir(self) -> Path:
        return Path(self.get("project.output_dir", "output"))

    @property
    def cache_dir(self) -> Path:
        return Path(self.get("project.cache_dir", ".cache"))

    def ensure_dirs(self) -> None:
        for directory in (self.work_dir, self.output_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- provider
    def provider_name(self) -> str:
        return str(self.get("llm.provider", "anthropic")).strip().lower()

    def provider_config(self, name: Optional[str] = None) -> Dict[str, Any]:
        name = (name or self.provider_name()).lower()
        providers = self.get("llm.providers", {}) or {}
        conf = providers.get(name)
        if conf is None:
            raise ConfigError(
                f"Provider LLM '{name}' belum diatur di llm.providers. "
                f"Pilihan yang ada: {', '.join(sorted(providers))}"
            )
        merged = {
            "temperature": self.get("llm.temperature", 0.4),
            "max_tokens": self.get("llm.max_tokens", 4000),
            "timeout_seconds": self.get("llm.timeout_seconds", 120),
            "max_retries": self.get("llm.max_retries", 3),
        }
        merged.update(dict(conf))
        merged["name"] = name
        return merged

    def api_key(self, name: Optional[str] = None) -> Optional[str]:
        """Ambil kunci API dari environment (sudah termasuk isi ``.env``)."""

        conf = self.provider_config(name)
        env_name = conf.get("api_key_env")
        if not env_name:
            return None
        value = os.environ.get(str(env_name))
        return value.strip() if value else None

    # --------------------------------------------------------- validation
    def validate(self) -> None:
        """Cek nilai-nilai yang kalau salah bikin gagal jauh di belakang."""

        min_dur = float(self.get("candidates.min_duration", 20))
        max_dur = float(self.get("candidates.max_duration", 90))
        if min_dur <= 0:
            raise ConfigError("candidates.min_duration harus > 0")
        if max_dur <= min_dur:
            raise ConfigError(
                "candidates.max_duration harus lebih besar dari candidates.min_duration"
            )

        min_clips = int(self.get("candidates.min_clips", 5))
        max_clips = int(self.get("candidates.max_clips", 15))
        if min_clips < 1:
            raise ConfigError("candidates.min_clips harus >= 1")
        if max_clips < min_clips:
            raise ConfigError("candidates.max_clips tidak boleh < candidates.min_clips")

        width = int(self.get("render.width", 1080))
        height = int(self.get("render.height", 1920))
        if width <= 0 or height <= 0:
            raise ConfigError("render.width dan render.height harus > 0")
        if height <= width:
            log.warning(
                "render %dx%d bukan vertikal; TikTok/Reels/Shorts mau 9:16 "
                "(mis. 1080x1920).",
                width,
                height,
            )

        weights = self.get("scoring.weights", {}) or {}
        if not isinstance(weights, Mapping) or not weights:
            raise ConfigError("scoring.weights tidak boleh kosong")
        total = sum(float(v) for v in weights.values())
        if total <= 0:
            raise ConfigError("Jumlah scoring.weights harus > 0")
        if abs(total - 1.0) > 0.01:
            log.warning(
                "Jumlah scoring.weights = %.3f (bukan 1.0); skor akan dinormalisasi.",
                total,
            )

        fallback = str(self.get("reframe.fallback", "blur")).lower()
        if fallback not in {"blur", "center"}:
            raise ConfigError("reframe.fallback hanya boleh 'blur' atau 'center'")

        position = str(self.get("watermark.position", "top-right")).lower()
        if position not in {"top-left", "top-right", "bottom-left", "bottom-right"}:
            raise ConfigError(
                "watermark.position harus salah satu dari: top-left, top-right, "
                "bottom-left, bottom-right"
            )

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)
