"""Fixture bersama untuk semua test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tukang_sayur.config import Config  # noqa: E402
from tukang_sayur.models import Segment, Transcript  # noqa: E402
from tukang_sayur.transcribe import distribute_words  # noqa: E402
from tukang_sayur.utils import has_ffmpeg  # noqa: E402

#: Kalimat contoh berbahasa Indonesia untuk transkrip uji.
SAMPLE_SENTENCES = [
    "Selamat datang di dapur Tukang Sayur.",
    "Hari ini kita bahas cara memilih sayur yang segar.",
    "Pertama, lihat batangnya, kalau kaku berarti baru dipetik.",
    "Kedua, cium aromanya, sayur segar masih wangi tanah.",
    "Ketiga, jangan pilih yang pangkal daunnya menguning.",
    "Ternyata rahasianya sederhana, tapi jarang ada yang tahu.",
    "Banyak orang salah karena hanya melihat harga termurah.",
    "Padahal sayur lemas justru jadi lebih mahal di akhir.",
    "Karena setengahnya berakhir di tempat sampah, bukan di meja.",
    "Coba praktikkan besok pagi waktu belanja ke pasar.",
    "Kalau menurut kalian berguna, simpan dulu videonya.",
    "Minggu depan kita bahas cara menyimpan sayur supaya awet.",
]

needs_ffmpeg = pytest.mark.skipif(
    not has_ffmpeg(), reason="butuh ffmpeg + ffprobe di PATH"
)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Config default dengan folder kerja di direktori sementara."""

    config = Config.load(None, load_env=False)
    config.set("project.work_dir", str(tmp_path / "work"))
    config.set("project.output_dir", str(tmp_path / "output"))
    config.set("project.cache_dir", str(tmp_path / "cache"))
    config.set("llm.provider", "mock")
    config.set("transcription.engine", "dummy")
    return config


@pytest.fixture
def transcript() -> Transcript:
    """Transkrip contoh: 12 kalimat, masing-masing 6 detik (total 72 detik)."""

    segments = []
    cursor = 0.0
    for sentence in SAMPLE_SENTENCES:
        end = cursor + 6.0
        segments.append(
            Segment(
                start=cursor,
                end=end,
                text=sentence,
                words=distribute_words(sentence, cursor, end),
            )
        )
        cursor = end
    return Transcript(
        language="id", duration=cursor, segments=segments, model="dummy", source="uji.mp4"
    )


@pytest.fixture
def weights() -> dict:
    return {"hook": 0.35, "emotion": 0.25, "clarity": 0.20, "standalone": 0.20}
