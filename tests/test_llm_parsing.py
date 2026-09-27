"""Test pembaca balasan LLM - bagian yang paling sering bikin pipeline mati."""

from __future__ import annotations

import json

import pytest

from tukang_sayur.errors import LLMParseError
from tukang_sayur.llm.parsing import (
    extract_json,
    find_items,
    parse_candidates,
    parse_metadata_items,
    parse_scores,
    pick_time,
    score_from_entry,
    strip_fences,
)
from tukang_sayur.models import Candidate


# --------------------------------------------------------------- extract_json
def test_json_polos():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_json_dalam_blok_kode():
    teks = 'Tentu, ini hasilnya:\n```json\n{"candidates": []}\n```\nSemoga membantu!'
    assert extract_json(teks) == {"candidates": []}


def test_json_dengan_basa_basi_di_depan_dan_belakang():
    teks = 'Berikut hasil analisis saya.\n[{"start": 1, "end": 2}]\nSilakan dicek.'
    assert extract_json(teks) == [{"start": 1, "end": 2}]


def test_json_dengan_koma_berlebih_dan_komentar():
    teks = '{"items": [{"a": 1,}, {"b": 2}],  // catatan\n}'
    assert extract_json(teks) == {"items": [{"a": 1}, {"b": 2}]}


def test_json_dengan_kutip_miring_dan_nilai_python():
    teks = '{“a”: None, "b": True, "c": NaN}'
    assert extract_json(teks) == {"a": None, "b": True, "c": 0}


def test_ambil_blok_terpanjang_saat_ada_dua_blok():
    teks = "```json\n{\"a\": 1}\n```\nlalu yang benar:\n```json\n{\"candidates\": [1,2,3]}\n```"
    assert extract_json(teks) == {"candidates": [1, 2, 3]}


def test_strip_fences_tanpa_blok():
    assert strip_fences("cuma teks") == "cuma teks"


@pytest.mark.parametrize("teks", ["", "   ", "sama sekali bukan json"])
def test_json_tidak_ada_melempar_error(teks):
    with pytest.raises(LLMParseError):
        extract_json(teks)


def test_find_items_menelusuri_struktur_bersarang():
    payload = {"hasil": {"data": {"candidates": [{"start": 1, "end": 5}]}}}
    assert find_items(payload) == [{"start": 1, "end": 5}]


def test_find_items_menerima_objek_tunggal():
    assert find_items({"start": 1, "end": 5, "hook": "x"})[0]["hook"] == "x"


# ----------------------------------------------------------- parse_candidates
def test_parse_candidates_alias_kunci_dan_timecode():
    balasan = json.dumps(
        {
            "kandidat": [
                {"mulai": "00:00:12.5", "selesai": "00:01:05", "hook": "Satu dua tiga",
                 "alasan": "menarik"},
                {"start_time": 120, "duration": 45, "hook_text": "Kedua"},
                {"begin": 200, "to": "03:40", "judul": "Judul jadi hook"},
            ]
        }
    )
    hasil = parse_candidates(balasan, max_hook_words=8)
    assert [(c.start, c.end) for c in hasil] == [(12.5, 65.0), (120.0, 165.0), (200.0, 220.0)]
    assert hasil[0].reason == "menarik"
    assert hasil[2].hook_text == "Judul jadi hook"


def test_parse_candidates_memotong_hook_maksimal_delapan_kata():
    balasan = '[{"start": 1, "end": 30, "hook_text": "satu dua tiga empat lima enam tujuh delapan sembilan sepuluh"}]'
    (kandidat,) = parse_candidates(balasan, max_hook_words=8)
    assert len(kandidat.hook_text.split()) == 8


def test_parse_candidates_melewati_entri_rusak_bukan_gagal_total():
    balasan = json.dumps(
        {
            "candidates": [
                {"start": 10},                      # tanpa end/duration
                {"start": 30, "end": 20},           # end <= start
                {"keterangan": "tidak ada waktu"},  # tanpa waktu sama sekali
                {"start": 50, "end": 80, "hook_text": "Yang ini benar"},
            ]
        }
    )
    hasil = parse_candidates(balasan, max_hook_words=8)
    assert len(hasil) == 1 and hasil[0].hook_text == "Yang ini benar"


def test_parse_candidates_semua_rusak_melempar_error():
    with pytest.raises(LLMParseError):
        parse_candidates('{"candidates": [{"catatan": "kosong"}]}')


def test_parse_candidates_menghormati_durasi_sumber():
    balasan = '[{"start": 500, "end": 540, "hook_text": "Di luar durasi"}]'
    with pytest.raises(LLMParseError):
        parse_candidates(balasan, source_duration=120.0)


@pytest.mark.parametrize(
    "nilai,harapan",
    [(12.5, 12.5), ("12.5", 12.5), ("00:01:05", 65.0), ("1:05", 65.0), ("90s", 90.0)],
)
def test_pick_time_berbagai_format(nilai, harapan):
    assert pick_time({"start": nilai}, ("start",)) == harapan


# --------------------------------------------------------------- parse_scores
@pytest.fixture
def kandidat():
    return [
        Candidate(0, 30, "Pertama", transcript_text="isi satu"),
        Candidate(40, 75, "Kedua", transcript_text="isi dua"),
        Candidate(90, 130, "Ketiga", transcript_text="isi tiga"),
    ]


def test_parse_scores_dimensi_bersarang(kandidat, weights):
    balasan = json.dumps(
        {
            "scores": [
                {"index": 1, "scores": {"hook": 90, "emotion": 80, "clarity": 70, "standalone": 60}},
                {"index": 2, "scores": {"hook": 50, "emotion": 50, "clarity": 50, "standalone": 50}},
                {"index": 3, "scores": {"hook": 40, "emotion": 30, "clarity": 60, "standalone": 20}},
            ]
        }
    )
    hasil = parse_scores(balasan, kandidat, weights)
    assert len(hasil) == 3
    assert hasil[0].score == pytest.approx(77.5)
    assert hasil[1].score == pytest.approx(50.0)
    assert all(item.score > 0 for item in hasil)


def test_parse_scores_penomoran_nol_based(kandidat, weights):
    balasan = json.dumps(
        [{"id": 0, "hook": 80, "emotion": 80, "clarity": 80, "standalone": 80},
         {"id": 1, "hook": 60, "emotion": 60, "clarity": 60, "standalone": 60}]
    )
    hasil = parse_scores(balasan, kandidat, weights)
    assert hasil[0].candidate.hook_text == "Pertama"
    assert hasil[1].candidate.hook_text == "Kedua"


def test_parse_scores_tanpa_nomor_pakai_urutan(kandidat, weights):
    balasan = json.dumps(
        [{"hook": 90, "emotion": 90, "clarity": 90, "standalone": 90},
         {"hook": 40, "emotion": 40, "clarity": 40, "standalone": 40}]
    )
    hasil = parse_scores(balasan, kandidat, weights)
    assert hasil[0].candidate.hook_text == "Pertama"
    assert hasil[0].score == pytest.approx(90.0)


def test_parse_scores_nomor_ngawur_tidak_bikin_gagal(kandidat, weights):
    balasan = json.dumps([{"index": 999, "hook": 70, "emotion": 70, "clarity": 70, "standalone": 70}])
    hasil = parse_scores(balasan, kandidat, weights)
    assert len(hasil) == 1 and hasil[0].score == pytest.approx(70.0)


def test_parse_scores_hanya_total_eksplisit(kandidat, weights):
    balasan = json.dumps([{"index": 1, "total_score": "82 / 100"}])
    (hasil,) = parse_scores(balasan, kandidat, weights)
    assert hasil.score == pytest.approx(82.0)


def test_parse_scores_tanpa_angka_melempar_error(kandidat, weights):
    with pytest.raises(LLMParseError):
        parse_scores('[{"index": 1, "reason": "bagus"}]', kandidat, weights)


def test_score_from_entry_urutan_sumber_nilai(weights):
    # dimensi menang atas total eksplisit
    total, rincian = score_from_entry(
        {"hook": 100, "emotion": 100, "clarity": 100, "standalone": 100, "score": 10}, weights
    )
    assert total == pytest.approx(100.0) and rincian.hook == 100

    # dimensi semuanya 0 tapi ada total -> pakai total (bukan 0)
    total, _ = score_from_entry(
        {"hook": 0, "emotion": 0, "clarity": 0, "standalone": 0, "overall": 81}, weights
    )
    assert total == pytest.approx(81.0)

    # tidak ada angka apa pun -> None, bukan 0
    total, _ = score_from_entry({"reason": "lumayan"}, weights)
    assert total is None


def test_score_from_entry_alias_bahasa_indonesia(weights):
    total, rincian = score_from_entry(
        {"rincian": {"hook": 80, "emosi": 60, "kejelasan": 70, "mandiri": 50}}, weights
    )
    assert total > 0 and rincian.emotion == 60


# ------------------------------------------------------------- metadata items
def test_parse_metadata_items():
    balasan = json.dumps(
        {"items": [
            {"index": 1, "title": "Judul", "caption": "Isi caption.", "hashtags": ["#fyp", "sayur"]},
            {"index": 2, "judul": "Kedua", "deskripsi": "Caption dua", "tags": "a, b c"},
        ]}
    )
    hasil = parse_metadata_items(balasan, 2)
    assert hasil[0]["title"] == "Judul"
    assert hasil[0]["hashtags"] == ["fyp", "sayur"]
    assert hasil[1]["hashtags"] == ["a", "b", "c"]
