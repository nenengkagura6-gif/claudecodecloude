"""Test end-to-end: video contoh -> klip jadi, tanpa jaringan.

Memakai video dummy (ffmpeg lavfi), transcriber tiruan, dan MockProvider, jadi
tidak butuh internet, kunci API, maupun model Whisper. Yang diuji di sini
justru bagian yang tidak bisa dites dengan mock: ffmpeg beneran menghasilkan
MP4 9:16, subtitle terbakar, dan hook tampil di atas video yang tetap berjalan.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tukang_sayur.demo import DEMO_SCRIPT, make_dummy_video
from tukang_sayur.llm.providers import MockProvider
from tukang_sayur.pipeline import ClipJobOptions, run_clip_job
from tukang_sayur.transcribe import DummyTranscriber
from tukang_sayur.utils import ffmpeg_bin, probe_media

from conftest import needs_ffmpeg  # noqa: E402  (conftest satu folder)

pytestmark = [needs_ffmpeg, pytest.mark.slow]


@pytest.fixture(scope="module")
def video_contoh(tmp_path_factory) -> Path:
    """Video 45 detik, 640x360 - cukup untuk menguji crop 9:16."""

    folder = tmp_path_factory.mktemp("contoh")
    target = folder / "contoh.mp4"
    make_dummy_video(target, duration=45, width=640, height=360, fps=24, script=DEMO_SCRIPT)
    return target


@pytest.fixture
def cfg_cepat(cfg):
    """Config yang dibuat kecil & cepat supaya test tidak lama."""

    cfg.set("render.width", 360)
    cfg.set("render.height", 640)
    cfg.set("render.fps", 24)
    cfg.set("render.preset", "ultrafast")
    cfg.set("render.crf", 34)
    cfg.set("candidates.min_duration", 10)
    cfg.set("candidates.max_duration", 20)
    cfg.set("candidates.min_clips", 1)
    cfg.set("candidates.max_clips", 2)
    cfg.set("scoring.min_score", 0)
    cfg.set("audio.loudnorm", False)   # loudnorm dua tahap lambat untuk test
    cfg.set("reframe.enabled", False)  # OpenCV opsional; paksa jalur fallback
    return cfg


@pytest.fixture(scope="module")
def _hasil(tmp_path_factory, video_contoh):
    """Jalankan pipeline sekali, dipakai banyak assertion."""

    from tukang_sayur.config import Config

    folder = tmp_path_factory.mktemp("kerja")
    config = Config.load(None, load_env=False)
    config.set("project.work_dir", str(folder / "work"))
    config.set("project.output_dir", str(folder / "output"))
    config.set("project.cache_dir", str(folder / "cache"))
    for kunci, nilai in {
        "render.width": 360, "render.height": 640, "render.fps": 24,
        "render.preset": "ultrafast", "render.crf": 34,
        "candidates.min_duration": 10, "candidates.max_duration": 20,
        "candidates.min_clips": 1, "candidates.max_clips": 2,
        "scoring.min_score": 0, "audio.loudnorm": False, "reframe.enabled": False,
        "project.keep_intermediate": True,  # supaya file .ass bisa diperiksa
    }.items():
        config.set(kunci, nilai)

    job = run_clip_job(
        str(video_contoh),
        config,
        options=ClipJobOptions(max_clips=2),
        provider=MockProvider(),
        transcriber=DummyTranscriber(config, script=DEMO_SCRIPT),
    )
    return job, config


def test_pipeline_menghasilkan_klip(_hasil):
    job, _ = _hasil
    assert job.clips, "harus ada minimal satu klip"
    for klip in job.clips:
        berkas = Path(klip.video_path)
        assert berkas.is_file() and berkas.stat().st_size > 10_000


def test_klip_berbentuk_vertikal_dan_punya_audio(_hasil):
    job, config = _hasil
    for klip in job.clips:
        info = probe_media(klip.video_path)
        assert (info.width, info.height) == (
            config.get("render.width"), config.get("render.height")
        )
        assert info.height > info.width  # 9:16
        assert info.has_audio, "audio harus ikut sejak awal"
        assert info.duration > 5


def test_best_score_tersimpan_dan_bukan_nol(_hasil):
    job, _ = _hasil
    assert job.best_score > 0
    assert job.best_score >= max(klip.score for klip in job.clips) - 0.01
    assert all(klip.score > 0 for klip in job.clips)


def test_metadata_json_dan_csv_ditulis(_hasil):
    import json

    job, _ = _hasil
    data = json.loads(Path(job.metadata_json).read_text(encoding="utf-8"))
    assert data["best_score"] == pytest.approx(job.best_score)
    assert data["clip_count"] == len(job.clips)

    baris = Path(job.metadata_csv).read_text(encoding="utf-8").splitlines()
    assert len(baris) == len(job.clips) + 1  # header + isi
    assert "title" in baris[0] and "score" in baris[0]


def test_thumbnail_dibuat(_hasil):
    job, _ = _hasil
    for klip in job.clips:
        assert klip.thumbnail_path and Path(klip.thumbnail_path).is_file()


def test_transkrip_disimpan(_hasil):
    job, _ = _hasil
    assert Path(job.transcript_path).is_file()


def test_berkas_ass_berisi_hook_dari_detik_nol(_hasil):
    job, _ = _hasil
    klip = job.clips[0]
    assert klip.subtitle_path, "jalur berkas ASS harus tercatat"
    isi = Path(klip.subtitle_path).read_text(encoding="utf-8")
    hook = [b for b in isi.splitlines() if ",Hook," in b]
    assert hook and ",0:00:00.00," in hook[0]
    assert any(",Sub," in b for b in isi.splitlines()), "subtitle kata ikut ada"


def _ambil_frame(video: str, detik: float, tujuan: Path) -> bytes:
    subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{detik}", "-i", str(video), "-frames:v", "1", str(tujuan)],
        check=True,
    )
    return tujuan.read_bytes()


def test_hook_tampil_di_atas_video_hidup_bukan_freeze(_hasil, tmp_path):
    """Selama hook tampil (0-3 detik), gambarnya harus tetap bergerak."""

    job, _ = _hasil
    video = job.clips[0].video_path
    awal = _ambil_frame(video, 0.2, tmp_path / "a.png")
    tengah = _ambil_frame(video, 1.5, tmp_path / "b.png")
    akhir = _ambil_frame(video, 2.8, tmp_path / "c.png")

    assert awal != tengah, "frame membeku selama hook (freeze-frame terdeteksi)"
    assert tengah != akhir, "frame membeku selama hook (freeze-frame terdeteksi)"


def test_audio_mulai_dari_detik_nol(_hasil):
    """Audio harus ada sejak awal, bukan baru masuk setelah hook."""

    job, _ = _hasil
    proses = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-nostats", "-t", "2.0", "-i", job.clips[0].video_path,
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    keluaran = proses.stderr
    assert "mean_volume" in keluaran
    rata = float(
        [b for b in keluaran.splitlines() if "mean_volume" in b][0].split(":")[-1].replace("dB", "")
    )
    assert rata > -90, "dua detik pertama ternyata senyap"


def test_dry_run_tidak_merender_tapi_tetap_menulis_metadata(cfg_cepat, video_contoh):
    job = run_clip_job(
        str(video_contoh),
        cfg_cepat,
        options=ClipJobOptions(max_clips=2, dry_run=True),
        provider=MockProvider(),
        transcriber=DummyTranscriber(cfg_cepat, script=DEMO_SCRIPT),
    )
    assert job.clips and all(not klip.video_path for klip in job.clips)
    assert Path(job.metadata_json).is_file()
    assert job.best_score > 0


def test_mode_potong_hening_menghasilkan_klip_lebih_pendek(cfg_cepat, video_contoh):
    """Nada sinus tidak punya hening, jadi pipeline harus melewati pemotongan
    dengan aman (bukan menghasilkan klip kosong)."""

    cfg_cepat.set("audio.remove_silence", True)
    cfg_cepat.set("candidates.max_clips", 1)
    job = run_clip_job(
        str(video_contoh),
        cfg_cepat,
        options=ClipJobOptions(max_clips=1),
        provider=MockProvider(),
        transcriber=DummyTranscriber(cfg_cepat, script=DEMO_SCRIPT),
    )
    assert job.clips and probe_media(job.clips[0].video_path).duration > 3


def test_klip_bisa_dibuat_tanpa_subtitle_dan_hook(cfg_cepat, video_contoh):
    job = run_clip_job(
        str(video_contoh),
        cfg_cepat,
        options=ClipJobOptions(max_clips=1, no_subtitles=True, no_hook=True),
        provider=MockProvider(),
        transcriber=DummyTranscriber(cfg_cepat, script=DEMO_SCRIPT),
    )
    assert job.clips and Path(job.clips[0].video_path).is_file()
