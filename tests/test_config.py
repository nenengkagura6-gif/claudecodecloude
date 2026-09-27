"""Test pemuat konfigurasi dan pembaca .env."""

from __future__ import annotations

import pytest

from tukang_sayur.config import Config, deep_merge, load_dotenv, parse_env_file
from tukang_sayur.errors import ConfigError


def test_deep_merge_tidak_mengubah_asli():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    hasil = deep_merge(base, {"a": {"c": 99}, "e": 5})
    assert hasil == {"a": {"b": 1, "c": 99}, "d": 3, "e": 5}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


@pytest.mark.parametrize(
    "baris,harapan",
    [
        ('KEY=nilai', {"KEY": "nilai"}),
        ('KEY="ada spasi" # komentar', {"KEY": "ada spasi"}),
        ("KEY=rahasia # komentar", {"KEY": "rahasia"}),
        ("KEY='pass#word'", {"KEY": "pass#word"}),
        ("export KEY=nilai", {"KEY": "nilai"}),
        ("# cuma komentar", {}),
        ("tanpa-sama-dengan", {}),
    ],
)
def test_parse_env_file(baris, harapan):
    assert parse_env_file(baris) == harapan


def test_load_dotenv_tidak_menimpa_env_yang_ada(tmp_path, monkeypatch):
    berkas = tmp_path / ".env"
    berkas.write_text("ANTHROPIC_API_KEY=dari-file\nLAIN=baru\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dari-shell")
    monkeypatch.delenv("LAIN", raising=False)

    load_dotenv(berkas)

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "dari-shell"  # env asli menang
    assert os.environ["LAIN"] == "baru"


def test_kunci_api_dibaca_dari_environment(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "kunci-rahasia")
    assert cfg.api_key("anthropic") == "kunci-rahasia"
    # provider lokal tidak butuh kunci
    assert cfg.api_key("ollama") is None


def test_akses_bertitik_dan_override(cfg):
    assert cfg.get("render.width") == 1080
    assert cfg.get("tidak.ada.kunci", "bawaan") == "bawaan"
    cfg.apply_overrides({"render.crf": "18", "audio.loudnorm": "false", "abaikan": None})
    assert cfg.get("render.crf") == 18
    assert cfg.get("audio.loudnorm") is False


def test_parse_cli_set():
    hasil = Config.parse_cli_set(["render.crf=18", "hook.duration=2.5", "hook.enabled=no"])
    assert hasil == {"render.crf": 18, "hook.duration": 2.5, "hook.enabled": False}
    with pytest.raises(ConfigError):
        Config.parse_cli_set(["tanpa-sama-dengan"])


@pytest.mark.parametrize(
    "kunci,nilai",
    [
        ("candidates.max_duration", 10),   # lebih kecil dari min_duration
        ("candidates.min_duration", 0),
        ("scoring.weights", {}),
        ("reframe.fallback", "pelangi"),
        ("watermark.position", "tengah"),
    ],
)
def test_validasi_menolak_nilai_ngawur(cfg, kunci, nilai):
    cfg.set(kunci, nilai)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_provider_config_menggabungkan_setelan_umum(cfg):
    conf = cfg.provider_config("openai")
    assert conf["model"] == "gpt-4o-mini"
    assert conf["max_retries"] == cfg.get("llm.max_retries")
    with pytest.raises(ConfigError):
        cfg.provider_config("provider-khayalan")
