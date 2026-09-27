"""Test penilaian viral.

Fokus utama: **best_score tidak boleh 0 gara-gara salah path properti.**
Itu bug klasik di tool sejenis, jadi di sini diuji dari beberapa arah:

* nama atribut skor pada model dikunci lewat ``SCORE_ATTR``;
* ``get_score`` melempar error (bukan mengembalikan 0) kalau path-nya salah;
* hasil ``parse_scores`` dari balasan LLM bersarang tetap menghasilkan > 0;
* ada jaring pengaman ``repair_zero_scores``;
* kegagalan LLM jatuh ke heuristik yang selalu > 0;
* nilai yang tersimpan di ``JobResult`` sama dengan yang dihitung.
"""

from __future__ import annotations

import json

import pytest

from tukang_sayur.errors import ScoringError
from tukang_sayur.llm.base import LLMProvider, LLMResult
from tukang_sayur.llm.parsing import parse_scores
from tukang_sayur.models import (
    SCORE_ATTR,
    SCORE_DIMENSIONS,
    Candidate,
    JobResult,
    ScoreBreakdown,
    ScoredCandidate,
)
from tukang_sayur.scoring import (
    average_score,
    best_score,
    get_score,
    heuristic_breakdown,
    heuristic_scores,
    repair_zero_scores,
    score_candidates,
    scoring_summary,
    select_clips,
)


def buat_kandidat(jumlah: int = 3):
    return [
        Candidate(
            start=i * 100.0,
            end=i * 100.0 + 40.0,
            hook_text=f"Hook nomor {i}",
            transcript_text="Ternyata rahasianya sederhana sekali. " * 8,
            index=i + 1,
        )
        for i in range(jumlah)
    ]


def skor(nilai: float, kandidat=None) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=kandidat or buat_kandidat(1)[0],
        breakdown=ScoreBreakdown(nilai, nilai, nilai, nilai),
        score=nilai,
    )


# ----------------------------------------------------- kontrak nama properti
def test_nama_atribut_skor_terkunci():
    """Kalau atribut ini diganti nama, test ini gagal lebih dulu.

    Dengan begitu tidak ada lagi kode yang membaca properti lain dan diam-diam
    mendapat 0.
    """

    item = skor(70)
    assert SCORE_ATTR == "score"
    assert hasattr(item, SCORE_ATTR)
    assert getattr(item, SCORE_ATTR) == pytest.approx(70)
    assert item.to_dict()["score"] == pytest.approx(70)
    assert set(SCORE_DIMENSIONS) == {"hook", "emotion", "clarity", "standalone"}


def test_get_score_dari_berbagai_bentuk():
    assert get_score(skor(88)) == pytest.approx(88)
    assert get_score({"score": 77}) == pytest.approx(77)
    assert get_score({"best_score": 66}) == pytest.approx(66)
    assert get_score({"total_score": 55}) == pytest.approx(55)
    assert get_score(42.0) == pytest.approx(42)
    # hanya ada rincian dimensi -> tetap dapat angka, bukan 0
    assert get_score({"score_breakdown": {"hook": 80, "emotion": 60, "clarity": 70, "standalone": 50}}) == pytest.approx(65)


def test_get_score_melempar_error_bukan_mengembalikan_nol():
    class Kosong:
        judul = "tanpa skor"

    with pytest.raises(ScoringError):
        get_score(Kosong())
    with pytest.raises(ScoringError):
        get_score({"judul": "tanpa skor"})


# ------------------------------------------------------------- best_score
def test_best_score_mengambil_yang_tertinggi():
    items = [skor(41), skor(93.4), skor(77)]
    assert best_score(items) == pytest.approx(93.4)
    assert average_score(items) == pytest.approx((41 + 93.4 + 77) / 3, abs=0.01)


def test_best_score_daftar_kosong_nol():
    assert best_score([]) == 0.0


def test_best_score_tidak_nol_untuk_payload_llm_bersarang(weights):
    """Regresi: skor hanya ada di path bersarang ``scores.hook`` dsb.

    Kode lama yang membaca ``item["score"]`` akan menghasilkan 0 di sini.
    """

    kandidat = buat_kandidat(3)
    balasan = json.dumps(
        {
            "result": {
                "scores": [
                    {"index": 1, "scores": {"hook": 91, "emotion": 84, "clarity": 88, "standalone": 79}},
                    {"index": 2, "scores": {"hook": 62, "emotion": 55, "clarity": 70, "standalone": 58}},
                    {"index": 3, "scores": {"hook": 45, "emotion": 40, "clarity": 52, "standalone": 38}},
                ]
            }
        }
    )
    dinilai = parse_scores(balasan, kandidat, weights)
    nilai_terbaik = best_score(dinilai)
    assert nilai_terbaik > 0
    assert nilai_terbaik == pytest.approx(86.5, abs=0.5)


def test_best_score_untuk_total_eksplisit_saja(weights):
    kandidat = buat_kandidat(2)
    balasan = '[{"index": 1, "overall_score": 73}, {"index": 2, "overall_score": 64}]'
    dinilai = parse_scores(balasan, kandidat, weights)
    assert best_score(dinilai) == pytest.approx(73.0)


def test_repair_zero_scores_menyelamatkan_skor_nol(weights):
    rusak = ScoredCandidate(
        candidate=buat_kandidat(1)[0],
        breakdown=ScoreBreakdown(90, 80, 70, 60),
        score=0,  # seolah-olah dibaca dari properti yang salah
    )
    (diperbaiki,) = repair_zero_scores([rusak], weights)
    assert diperbaiki.score == pytest.approx(77.5)
    assert best_score([diperbaiki]) > 0


def test_repair_zero_scores_membiarkan_nol_yang_memang_nol(weights):
    benar_benar_nol = ScoredCandidate(
        candidate=buat_kandidat(1)[0], breakdown=ScoreBreakdown(0, 0, 0, 0), score=0
    )
    (hasil,) = repair_zero_scores([benar_benar_nol], weights)
    assert hasil.score == 0


def test_job_result_menyimpan_best_score():
    job = JobResult(source="a.mp4", video_path="a.mp4", output_dir="out")
    job.best_score = best_score([skor(55), skor(81.25)])
    assert job.best_score == pytest.approx(81.25)
    assert job.to_dict()["best_score"] == pytest.approx(81.25)


# -------------------------------------------------------------- heuristik
def test_heuristik_tidak_pernah_nol(cfg):
    for kandidat in buat_kandidat(4):
        rincian = heuristic_breakdown(kandidat, cfg)
        for nama in SCORE_DIMENSIONS:
            assert getattr(rincian, nama) > 0
        assert rincian.weighted_total(cfg.get("scoring.weights")) > 0


def test_heuristik_kandidat_kosong_pun_tetap_positif(cfg):
    kosong = Candidate(0, 25, "", transcript_text="")
    assert heuristic_breakdown(kosong, cfg).weighted_total(cfg.get("scoring.weights")) > 0


def test_heuristik_membedakan_hook_kuat_dan_lemah(cfg):
    kuat = Candidate(0, 55, "Kenapa 90 persen orang salah?",
                     transcript_text="Ternyata banyak yang kaget waktu tahu ini. " * 6)
    lemah = Candidate(0, 55, "dan", transcript_text="dan begitu saja terus menerus. " * 6)
    bobot = cfg.get("scoring.weights")
    assert heuristic_breakdown(kuat, cfg).weighted_total(bobot) > heuristic_breakdown(lemah, cfg).weighted_total(bobot)


# ------------------------------------------------ score_candidates end-to-end
class ProviderGagal(LLMProvider):
    name = "gagal"
    requires_api_key = False

    def _request(self, **kwargs):
        raise RuntimeError("seharusnya tidak dipanggil langsung")

    def complete(self, prompt, **kwargs):  # noqa: D401 - sengaja selalu gagal
        from tukang_sayur.errors import LLMError

        raise LLMError("provider sedang mati")


class ProviderNgaco(LLMProvider):
    name = "ngaco"
    requires_api_key = False

    def _request(self, *, prompt, system, temperature, max_tokens):
        return LLMResult(text="maaf saya tidak mengerti", provider=self.name)


def test_score_candidates_jatuh_ke_heuristik_saat_llm_mati(cfg):
    kandidat = buat_kandidat(3)
    hasil = score_candidates(kandidat, cfg, ProviderGagal(model="x"))
    assert len(hasil) == 3
    assert best_score(hasil) > 0


def test_score_candidates_jatuh_ke_heuristik_saat_balasan_ngaco(cfg):
    kandidat = buat_kandidat(2)
    hasil = score_candidates(kandidat, cfg, ProviderNgaco(model="x"))
    assert best_score(hasil) > 0


def test_score_candidates_tanpa_provider(cfg):
    hasil = score_candidates(buat_kandidat(2), cfg, None)
    assert best_score(hasil) > 0


def test_score_candidates_dengan_mock_provider(cfg):
    from tukang_sayur.llm.providers import MockProvider

    kandidat = buat_kandidat(3)
    hasil = score_candidates(kandidat, cfg, MockProvider())
    assert len(hasil) == 3
    assert best_score(hasil) > 0
    assert hasil[0].score >= hasil[-1].score  # sudah terurut


# ------------------------------------------------------------- select_clips
def test_select_clips_urut_skor_dan_batas_jumlah(cfg):
    cfg.set("scoring.min_score", 0)
    items = [skor(40), skor(95), skor(70), skor(60)]
    terpilih = select_clips(items, cfg, max_clips=2)
    assert [round(i.score) for i in terpilih] == [95, 70]


def test_select_clips_menyaring_min_score(cfg):
    cfg.set("scoring.min_score", 65)
    terpilih = select_clips([skor(40), skor(95), skor(70)], cfg, max_clips=5)
    assert [round(i.score) for i in terpilih] == [95, 70]


def test_select_clips_tetap_memberi_hasil_walau_semua_di_bawah_ambang(cfg):
    cfg.set("scoring.min_score", 99)
    terpilih = select_clips([skor(40), skor(55)], cfg, max_clips=1)
    assert len(terpilih) == 1 and terpilih[0].score == pytest.approx(55)


def test_scoring_summary(cfg):
    ringkasan = scoring_summary(heuristic_scores(buat_kandidat(3), cfg))
    assert ringkasan["count"] == 3
    assert ringkasan["best_score"] > 0
    assert set(ringkasan["dimensions"]) == set(SCORE_DIMENSIONS)
