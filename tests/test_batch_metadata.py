"""Test mode batch (resume) dan penulisan metadata."""

from __future__ import annotations

import json

import pytest

from tukang_sayur.batch import (
    STATUS_DONE,
    STATUS_FAILED,
    BatchState,
    read_sources,
    state_path_for,
)
from tukang_sayur.errors import TukangSayurError
from tukang_sayur.metadata import (
    CSV_COLUMNS,
    clip_row,
    fallback_metadata,
    generate_metadata,
    write_metadata_files,
)
from tukang_sayur.models import Candidate, ClipResult, JobResult, ScoreBreakdown, ScoredCandidate


# --------------------------------------------------------------------- batch
def test_read_sources_melewati_komentar_dan_baris_kosong(tmp_path):
    berkas = tmp_path / "daftar.txt"
    berkas.write_text(
        "# daftar video\n\nhttps://a.test/1\n  https://a.test/2  \n# lagi\nvideo.mp4\n",
        encoding="utf-8",
    )
    assert read_sources(berkas) == ["https://a.test/1", "https://a.test/2", "video.mp4"]


def test_read_sources_berkas_kosong(tmp_path):
    berkas = tmp_path / "kosong.txt"
    berkas.write_text("# cuma komentar\n", encoding="utf-8")
    with pytest.raises(TukangSayurError):
        read_sources(berkas)


def test_read_sources_berkas_tidak_ada(tmp_path):
    with pytest.raises(TukangSayurError):
        read_sources(tmp_path / "hilang.txt")


def test_state_menyimpan_dan_memuat_ulang(tmp_path):
    berkas = tmp_path / "state.json"
    state = BatchState.load(berkas)
    state.update("https://a.test/1", status=STATUS_DONE, clips=3, best_score=88.5)
    state.update("https://a.test/2", status=STATUS_FAILED, error="ffmpeg mati")

    lagi = BatchState.load(berkas)
    assert lagi.is_done("https://a.test/1")
    assert not lagi.is_done("https://a.test/2")
    assert lagi.status("https://a.test/3") == "pending"
    assert lagi.summary()[STATUS_DONE] == 1


def test_state_rusak_dimulai_dari_awal(tmp_path):
    berkas = tmp_path / "state.json"
    berkas.write_text("{bukan json", encoding="utf-8")
    assert BatchState.load(berkas).items == {}


def test_state_path_mengikuti_folder_output(cfg):
    assert state_path_for(cfg).parent == cfg.output_dir
    cfg.set("batch.state_file", "/tmp/khusus.json")
    assert str(state_path_for(cfg)) == "/tmp/khusus.json"


def test_resume_melewati_sumber_yang_sudah_selesai(cfg, tmp_path, monkeypatch):
    daftar = tmp_path / "daftar.txt"
    daftar.write_text("sumber-a\nsumber-b\n", encoding="utf-8")

    diproses = []

    def palsu(source, config, **kwargs):
        diproses.append(source)
        return JobResult(source=source, video_path="x.mp4", output_dir="out", best_score=70)

    import tukang_sayur.batch as batch_module

    monkeypatch.setattr(batch_module, "run_clip_job", palsu)

    hasil = batch_module.run_batch(daftar, cfg)
    assert hasil.done == 2 and diproses == ["sumber-a", "sumber-b"]

    diproses.clear()
    lanjutan = batch_module.run_batch(daftar, cfg)
    assert lanjutan.skipped == 2 and diproses == []

    diproses.clear()
    ulang = batch_module.run_batch(daftar, cfg, resume=False)
    assert ulang.done == 2 and len(diproses) == 2


def test_batch_lanjut_meski_satu_sumber_gagal(cfg, tmp_path, monkeypatch):
    daftar = tmp_path / "daftar.txt"
    daftar.write_text("gagal\nsukses\n", encoding="utf-8")

    def palsu(source, config, **kwargs):
        if source == "gagal":
            raise TukangSayurError("sengaja gagal")
        return JobResult(source=source, video_path="x.mp4", output_dir="out")

    import tukang_sayur.batch as batch_module

    monkeypatch.setattr(batch_module, "run_clip_job", palsu)
    hasil = batch_module.run_batch(daftar, cfg)
    assert hasil.done == 1 and hasil.failed == 1
    assert "gagal" in hasil.errors

    # sumber yang gagal tidak ditandai selesai, jadi bisa dicoba lagi
    state = BatchState.load(state_path_for(cfg))
    assert state.status("gagal") == STATUS_FAILED
    assert state.is_done("sukses")


# ------------------------------------------------------------------ metadata
def contoh_item(skor: float = 74.0) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=Candidate(
            10, 45, "Rahasia dapur warteg",
            transcript_text="Ternyata rahasianya sederhana sekali. Kuncinya ada di kaldu.",
        ),
        breakdown=ScoreBreakdown(80, 70, 75, 65),
        score=skor,
    )


def test_fallback_metadata_selalu_terisi(cfg):
    meta = fallback_metadata(contoh_item(), cfg)
    assert meta["title"] and meta["caption"] and meta["hashtags"]
    assert len(meta["title"]) <= 60
    assert meta["caption"].rstrip().endswith("?")


def test_fallback_metadata_saat_hook_kosong(cfg):
    kosong = ScoredCandidate(candidate=Candidate(0, 30, "", transcript_text=""))
    meta = fallback_metadata(kosong, cfg)
    assert meta["title"] and meta["hashtags"]


def test_generate_metadata_dengan_mock_llm(cfg):
    from tukang_sayur.llm.providers import MockProvider

    hasil = generate_metadata([contoh_item(), contoh_item(60)], cfg, MockProvider())
    assert len(hasil) == 2
    assert all(item["title"] and item["hashtags"] for item in hasil)


def test_generate_metadata_tanpa_llm(cfg):
    cfg.set("metadata.use_llm", False)
    hasil = generate_metadata([contoh_item()], cfg, None)
    assert hasil[0]["hashtags"]


def test_generate_metadata_llm_gagal_pakai_cadangan(cfg):
    from tukang_sayur.errors import LLMError
    from tukang_sayur.llm.base import LLMProvider

    class Gagal(LLMProvider):
        name = "gagal"
        requires_api_key = False

        def _request(self, **kwargs):
            raise LLMError("mati")

        def complete(self, prompt, **kwargs):
            raise LLMError("mati")

    hasil = generate_metadata([contoh_item()], cfg, Gagal(model="x"))
    assert hasil[0]["title"]  # tetap ada isinya


def test_menulis_metadata_json_dan_csv(cfg, tmp_path):
    job = JobResult(
        source="video.mp4", video_path="video.mp4", output_dir=str(tmp_path),
        best_score=88.5, candidate_count=7, language="id", duration=600.0,
    )
    job.clips = [
        ClipResult(
            index=1, video_path=str(tmp_path / "01-klip.mp4"),
            source_start=12.5, source_end=48.0, duration=35.5, score=88.5,
            hook_text="Hook uji", title="Judul uji", caption="Caption uji.",
            hashtags=["fyp", "sayur"], thumbnail_path=str(tmp_path / "01-klip.jpg"),
            score_breakdown={"hook": 90, "emotion": 85, "clarity": 88, "standalone": 84},
            reframe_mode="blur", width=1080, height=1920,
        )
    ]
    jalur = write_metadata_files(job, cfg, tmp_path)

    data = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert data["best_score"] == pytest.approx(88.5)
    assert data["clips"][0]["hook_text"] == "Hook uji"
    assert data["clip_count"] == 1

    baris = (tmp_path / "metadata.csv").read_text(encoding="utf-8").splitlines()
    assert baris[0].split(",") == CSV_COLUMNS
    assert "#fyp #sayur" in baris[1]
    assert "00:00:12.500" in baris[1]  # timestamp sumber ikut tertulis
    assert jalur["json"].endswith("metadata.json")


def test_clip_row_lengkap():
    baris = clip_row(
        ClipResult(index=2, video_path="/x/02.mp4", source_start=1, source_end=2,
                   duration=1, score=50, hook_text="h")
    )
    assert set(baris) <= set(CSV_COLUMNS)
    assert baris["file"] == "02.mp4"
