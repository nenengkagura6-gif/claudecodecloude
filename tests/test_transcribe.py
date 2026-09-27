"""Test transkripsi tiruan, pembagian waktu kata, dan cache."""

from __future__ import annotations

import pytest

from tukang_sayur.models import Segment, Transcript
from tukang_sayur.transcribe import (
    DummyTranscriber,
    cache_path,
    distribute_words,
    ensure_word_timings,
    load_cached,
    transcribe_media,
    transcript_stats,
)
from tukang_sayur.utils import write_json


def test_distribute_words_menutupi_seluruh_rentang():
    kata = distribute_words("Halo dunia yang panjang", 10, 14)
    assert kata[0].start == pytest.approx(10)
    assert kata[-1].end == pytest.approx(14)
    for satu, berikutnya in zip(kata, kata[1:]):
        assert satu.end <= berikutnya.start + 1e-9


def test_distribute_words_berbobot_panjang_kata():
    kata = distribute_words("a panjangsekali", 0, 10)
    assert kata[1].duration > kata[0].duration


@pytest.mark.parametrize("teks", ["", "   "])
def test_distribute_words_teks_kosong(teks):
    assert distribute_words(teks, 0, 5) == []


def test_distribute_words_rentang_nol():
    kata = distribute_words("dua kata", 5, 5)
    assert all(k.start == k.end == 5 for k in kata)


def test_ensure_word_timings_mengisi_segmen_tanpa_kata():
    hasil = ensure_word_timings([Segment(0, 3, "dua kata")])
    assert len(hasil[0].words) == 2


def test_dummy_transcriber_dari_naskah(tmp_path):
    berkas = tmp_path / "video.mp4"
    berkas.write_bytes(b"bukan video sungguhan")
    transkrip = DummyTranscriber(script="Satu dua. Tiga empat!").transcribe(berkas)
    assert len(transkrip.segments) == 2
    assert transkrip.words and transkrip.language == "id"


def test_dummy_transcriber_membaca_sidecar_txt(tmp_path):
    berkas = tmp_path / "video.mp4"
    berkas.write_bytes(b"x")
    (tmp_path / "video.txt").write_text("Dari berkas pendamping.", encoding="utf-8")
    transkrip = DummyTranscriber().transcribe(berkas)
    assert "pendamping" in transkrip.text


def test_dummy_transcriber_membaca_sidecar_json(tmp_path):
    berkas = tmp_path / "video.mp4"
    berkas.write_bytes(b"x")
    isi = Transcript(
        "su", 5.0, [Segment(0, 5, "Kumaha damang", distribute_words("Kumaha damang", 0, 5))]
    )
    write_json(tmp_path / "video.mp4.words.json", isi.to_dict())
    transkrip = DummyTranscriber().transcribe(berkas)
    assert transkrip.language == "su" and transkrip.text == "Kumaha damang"


def test_cache_menyimpan_dan_memakai_ulang(cfg, tmp_path):
    berkas = tmp_path / "video.mp4"
    berkas.write_bytes(b"isi video palsu")

    hitungan = {"panggil": 0}

    class Penghitung(DummyTranscriber):
        name = "dummy"

        def transcribe(self, media_path):
            hitungan["panggil"] += 1
            return super().transcribe(media_path)

    pertama = transcribe_media(berkas, cfg, transcriber=Penghitung(cfg, script="Satu dua."))
    kedua = transcribe_media(berkas, cfg, transcriber=Penghitung(cfg, script="Satu dua."))

    assert hitungan["panggil"] == 1, "panggilan kedua harus dilayani cache"
    assert pertama.text == kedua.text
    assert cache_path(berkas, cfg, "dummy").is_file()


def test_cache_bisa_dimatikan(cfg, tmp_path):
    berkas = tmp_path / "video.mp4"
    berkas.write_bytes(b"isi")
    transcribe_media(berkas, cfg, transcriber=DummyTranscriber(cfg, script="A."), use_cache=False)
    assert not cache_path(berkas, cfg, "dummy").is_file()


def test_cache_kunci_berubah_saat_model_berganti(cfg, tmp_path):
    berkas = tmp_path / "video.mp4"
    berkas.write_bytes(b"isi")
    kunci_kecil = cache_path(berkas, cfg, "faster-whisper")
    cfg.set("transcription.model", "medium")
    assert cache_path(berkas, cfg, "faster-whisper") != kunci_kecil


def test_cache_rusak_diabaikan(cfg, tmp_path):
    rusak = tmp_path / "rusak.json"
    rusak.write_text("{bukan json", encoding="utf-8")
    assert load_cached(rusak) is None
    assert load_cached(tmp_path / "tidak-ada.json") is None


def test_transcript_stats(transcript):
    statistik = transcript_stats(transcript)
    assert statistik["segments"] == 12
    assert statistik["words"] > 0
    assert statistik["words_per_minute"] > 0


def test_words_between_dan_text_between(transcript):
    kata = transcript.words_between(6.0, 12.0)
    assert kata and all(6.0 - 0.01 <= w.start <= 12.0 for w in kata)
    assert transcript.text_between(6.0, 12.0)


def test_transcript_round_trip(transcript):
    kembali = Transcript.from_dict(transcript.to_dict())
    assert kembali.text == transcript.text
    assert len(kembali.words) == len(transcript.words)
