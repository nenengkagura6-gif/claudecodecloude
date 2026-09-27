"""Test tahap pencarian kandidat (memakai MockProvider, tanpa jaringan)."""

from __future__ import annotations

import pytest

from tukang_sayur.candidates import find_candidates
from tukang_sayur.errors import CandidateError
from tukang_sayur.llm.base import LLMProvider, LLMResult
from tukang_sayur.llm.factory import build_provider
from tukang_sayur.llm.providers import MockProvider
from tukang_sayur.models import Transcript
from tukang_sayur.prompts import candidate_prompt, chunk_segments, judge_prompt


def test_mock_provider_membaca_baris_transkrip():
    prompt = (
        "TASK: find_candidates\nCari 5 kandidat berdurasi 20-90 detik.\n"
        "[0.00 -> 8.00] Kalimat pertama.\n[8.00 -> 19.00] Kalimat kedua.\n"
        "[19.00 -> 31.00] Kalimat ketiga.\n"
    )
    hasil = MockProvider().complete(prompt)
    assert '"candidates"' in hasil.text


def test_find_candidates_menghasilkan_potongan_valid(cfg, transcript):
    cfg.set("candidates.min_duration", 15)
    cfg.set("candidates.max_duration", 40)
    kandidat = find_candidates(transcript, cfg, MockProvider(), wanted=5)

    assert kandidat, "harus ada kandidat"
    for satu in kandidat:
        assert 0 <= satu.start < satu.end <= transcript.duration
        assert satu.duration >= 9  # minimal 60% dari min_duration
        assert satu.transcript_text
        assert len(satu.hook_text.split()) <= cfg.get("candidates.max_hook_words")


def test_find_candidates_potongan_jatuh_di_batas_kalimat(cfg, transcript):
    from tukang_sayur.candidates import sentence_boundaries

    cfg.set("candidates.padding_start", 0)
    cfg.set("candidates.padding_end", 0)
    cfg.set("candidates.min_duration", 15)
    cfg.set("candidates.max_duration", 40)
    mulai, selesai = sentence_boundaries(transcript)

    for satu in find_candidates(transcript, cfg, MockProvider(), wanted=4):
        assert any(abs(satu.start - b) < 0.01 for b in mulai)
        assert any(abs(satu.end - b) < 0.01 for b in selesai)


def test_find_candidates_tidak_tumpang_tindih(cfg, transcript):
    cfg.set("candidates.min_duration", 12)
    cfg.set("candidates.max_duration", 30)
    kandidat = find_candidates(transcript, cfg, MockProvider(), wanted=6)
    for satu, berikutnya in zip(kandidat, kandidat[1:]):
        assert satu.overlap_ratio(berikutnya) <= cfg.get("candidates.max_overlap_ratio")


def test_find_candidates_transkrip_kosong(cfg):
    with pytest.raises(CandidateError):
        find_candidates(Transcript("id", 0, []), cfg, MockProvider())


class ProviderKosong(LLMProvider):
    name = "kosong"
    requires_api_key = False

    def _request(self, *, prompt, system, temperature, max_tokens):
        return LLMResult(text='{"candidates": []}', provider=self.name)


def test_find_candidates_balasan_kosong_melempar_error(cfg, transcript):
    with pytest.raises(CandidateError):
        find_candidates(transcript, cfg, ProviderKosong(model="x"))


def test_transkrip_panjang_dipecah_jadi_beberapa_prompt(cfg, transcript):
    potongan = chunk_segments(transcript.segments, max_chars=120)
    assert len(potongan) > 1
    assert sum(len(bagian) for bagian in potongan) == len(transcript.segments)


def test_prompt_memuat_aturan_penting(transcript):
    prompt = candidate_prompt(
        transcript.segments, wanted=5, min_duration=20, max_duration=90,
        video_duration=transcript.duration,
    )
    assert "TASK: find_candidates" in prompt
    assert "awal kalimat" in prompt and "akhir kalimat" in prompt
    assert "[0.00 -> 6.00]" in prompt


def test_judge_prompt_memberi_nomor_kandidat(cfg, transcript):
    kandidat = find_candidates(transcript, cfg, MockProvider(), wanted=3)
    prompt = judge_prompt(kandidat)
    assert "TASK: score_candidates" in prompt
    assert "#1" in prompt and "HOOK:" in prompt


def test_build_provider_menolak_provider_tak_dikenal(cfg):
    from tukang_sayur.errors import ConfigError

    cfg.set("llm.provider", "khayalan")
    with pytest.raises(ConfigError):
        build_provider(cfg)


def test_build_provider_minta_kunci_api(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from tukang_sayur.errors import ConfigError

    cfg.set("llm.provider", "anthropic")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_provider(cfg)


def test_build_provider_lokal_tanpa_kunci(cfg):
    cfg.set("llm.provider", "ollama")
    provider = build_provider(cfg)
    assert provider.name == "ollama" and provider.requires_api_key is False
