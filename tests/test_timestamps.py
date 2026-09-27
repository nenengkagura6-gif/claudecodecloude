"""Test semua perhitungan waktu: batas kalimat, padding, potong hening, keyframe.

Kalau angka-angka di sini salah, hasilnya kelihatan jelas di video: kalimat
terpotong di tengah, subtitle meleset, atau kamera bergeser tidak pada waktunya.
"""

from __future__ import annotations

import pytest

from tukang_sayur.audio import (
    keep_ranges_from_silences,
    parse_silencedetect,
    remap_time,
    remap_words,
    total_kept_duration,
)
from tukang_sayur.candidates import (
    apply_padding,
    drop_overlaps,
    nearest_boundary,
    normalize_candidates,
    sentence_boundaries,
    snap_range,
)
from tukang_sayur.ffexpr import (
    evaluate_keyframes,
    make_keyframes,
    piecewise_expression,
    simplify_keyframes,
)
from tukang_sayur.models import Candidate, Word
from tukang_sayur.utils import format_timestamp, human_duration, parse_timecode


# ------------------------------------------------------------ parse_timecode
@pytest.mark.parametrize(
    "masukan,harapan",
    [
        (83, 83.0), (83.5, 83.5), ("83", 83.0), ("83.5s", 83.5),
        ("1:23", 83.0), ("00:01:23", 83.0), ("00:01:23.5", 83.5),
        ("1:01:01", 3661.0), ("1m23s", 83.0), ("2h", 7200.0),
        ("bukan waktu", None), ("", None), (None, None), (True, None),
    ],
)
def test_parse_timecode(masukan, harapan):
    assert parse_timecode(masukan) == harapan


def test_format_waktu():
    assert format_timestamp(3661.2345) == "01:01:01.235"
    assert human_duration(95) == "01:35"
    assert human_duration(3723) == "1:02:03"


# ---------------------------------------------------------- batas kalimat
def test_sentence_boundaries_dari_segmen(transcript):
    mulai, selesai = sentence_boundaries(transcript)
    assert mulai[0] == 0.0
    assert selesai[-1] == pytest.approx(transcript.duration)
    assert len(mulai) >= len(transcript.segments)


def test_sentence_boundaries_memakai_tanda_baca_di_tengah_segmen():
    from tukang_sayur.models import Segment, Transcript
    from tukang_sayur.transcribe import distribute_words

    teks = "Kalimat pertama. Kalimat kedua menyusul."
    segmen = Segment(0, 8, teks, distribute_words(teks, 0, 8))
    mulai, selesai = sentence_boundaries(Transcript("id", 8, [segmen]))
    # ada batas di tengah segmen (setelah "pertama.")
    assert any(0 < b < 8 for b in selesai)
    assert any(0 < b < 8 for b in mulai)


@pytest.mark.parametrize(
    "nilai,prefer,harapan",
    [
        (10.2, "nearest", 10.0),
        (11.4, "after", 12.0),
        (11.4, "before", 10.0),   # prefer="before" memilih batas sebelumnya bila masih dalam toleransi
        (30.0, "nearest", None),  # jauh dari semua batas
    ],
)
def test_nearest_boundary(nilai, prefer, harapan):
    batas = [0.0, 6.0, 10.0, 12.0, 18.0]
    hasil = nearest_boundary(nilai, batas, tolerance=2.5, prefer=prefer)
    if harapan is None:
        assert hasil is None
    else:
        assert hasil == pytest.approx(harapan)


def test_nearest_boundary_daftar_kosong():
    assert nearest_boundary(5, []) is None


# ----------------------------------------------------------- snap_range
@pytest.fixture
def batas(transcript):
    return sentence_boundaries(transcript)


def test_snap_menggeser_ke_batas_kalimat(batas, transcript):
    mulai, selesai = batas
    hasil = snap_range(7.3, 25.4, mulai, selesai, min_duration=20, max_duration=90,
                       video_duration=transcript.duration)
    assert hasil[0] in mulai and hasil[1] in selesai


def test_snap_memperpanjang_yang_terlalu_pendek(batas, transcript):
    mulai, selesai = batas
    awal, akhir = snap_range(10, 14, mulai, selesai, min_duration=20, max_duration=90,
                             video_duration=transcript.duration)
    assert akhir - awal >= 20


def test_snap_memendekkan_yang_terlalu_panjang(batas, transcript):
    mulai, selesai = batas
    awal, akhir = snap_range(0, 200, mulai, selesai, min_duration=20, max_duration=40,
                             video_duration=transcript.duration)
    assert 20 <= akhir - awal <= 40


def test_snap_tidak_melewati_akhir_video(batas, transcript):
    mulai, selesai = batas
    awal, akhir = snap_range(65, 120, mulai, selesai, min_duration=20, max_duration=90,
                             video_duration=transcript.duration)
    assert akhir <= transcript.duration
    assert akhir - awal >= 20  # ditarik mundur, bukan dipaksa lewat ujung


def test_snap_tanpa_batas_kalimat_tetap_aman():
    awal, akhir = snap_range(5, 9, [], [], min_duration=20, max_duration=90, video_duration=60)
    assert akhir - awal >= 20 and akhir <= 60


# ------------------------------------------------------------- padding
def test_apply_padding_dijepit_di_tepi_video():
    assert apply_padding(0.1, 59.9, pad_start=0.5, pad_end=0.5, video_duration=60) == (0.0, 60.0)
    awal, akhir = apply_padding(10, 40, pad_start=0.25, pad_end=0.35, video_duration=60)
    assert (awal, akhir) == (9.75, 40.35)


# --------------------------------------------------------- tumpang tindih
def test_drop_overlaps_membuang_yang_mirip():
    kandidat = [
        Candidate(0, 40, "a"),
        Candidate(5, 42, "b"),    # tumpang tindih besar -> dibuang
        Candidate(60, 100, "c"),
    ]
    hasil = drop_overlaps(kandidat, max_overlap_ratio=0.35)
    assert [c.hook_text for c in hasil] == ["a", "c"]


def test_overlap_ratio_dihitung_atas_kandidat_terpendek():
    panjang = Candidate(0, 100, "panjang")
    pendek = Candidate(90, 100, "pendek")
    assert pendek.overlap_ratio(panjang) == pytest.approx(1.0)


def test_normalize_candidates_merapikan_semuanya(cfg, transcript):
    cfg.set("candidates.min_duration", 20)
    cfg.set("candidates.max_duration", 40)
    mentah = [
        Candidate(7.3, 25.4, "Hook satu yang panjang sekali sampai lebih delapan kata"),
        Candidate(8.0, 26.0, "Duplikat"),          # tumpang tindih -> dibuang
        Candidate(200.0, 260.0, "Di luar video"),  # di luar durasi
    ]
    hasil = normalize_candidates(mentah, transcript, cfg)
    assert len(hasil) >= 1
    for kandidat in hasil:
        assert kandidat.end <= transcript.duration
        assert kandidat.duration >= 12  # minimal 60% dari min_duration
        assert len(kandidat.hook_text.split()) <= 8
        assert kandidat.transcript_text  # teks transkrip ikut terisi
    assert [c.index for c in hasil] == list(range(1, len(hasil) + 1))


# -------------------------------------------------------- potong hening
def test_parse_silencedetect():
    keluaran = (
        "[silencedetect @ 0x1] silence_start: 3.5\n"
        "[silencedetect @ 0x1] silence_end: 5.2 | silence_duration: 1.7\n"
        "[silencedetect @ 0x1] silence_start: 9.0\n"
    )
    assert parse_silencedetect(keluaran, duration=12.0) == [(3.5, 5.2), (9.0, 12.0)]


def test_keep_ranges_dan_durasi_baru():
    keeps = keep_ranges_from_silences([(3.5, 5.2), (9.0, 12.0)], 12.0, keep_padding=0.1)
    assert keeps == [(0.0, 3.6), (5.1, 9.1)]
    assert total_kept_duration(keeps) == pytest.approx(7.6)


def test_keep_ranges_tanpa_hening():
    assert keep_ranges_from_silences([], 30.0) == [(0.0, 30.0)]


def test_remap_time_monoton_dan_tepat():
    keeps = [(0.0, 3.6), (5.1, 9.1)]
    nilai = [remap_time(t, keeps) for t in (0, 1.5, 3.6, 4.5, 5.1, 7.0, 9.1, 20)]
    assert nilai == [0.0, 1.5, 3.6, 3.6, 3.6, 5.5, 7.6, 7.6]
    assert nilai == sorted(nilai)  # tidak pernah mundur


def test_remap_time_tanpa_rentang():
    assert remap_time(5, []) is None


def test_remap_words_membuang_kata_di_bagian_yang_dipotong():
    keeps = [(0.0, 3.6), (5.1, 9.1)]
    kata = [
        Word(0.0, 1.0, "tetap"),
        Word(3.9, 4.9, "hilang"),   # seluruhnya di bagian hening
        Word(5.4, 6.0, "geser"),
    ]
    hasil = remap_words(kata, keeps)
    assert [w.text for w in hasil] == ["tetap", "geser"]
    assert hasil[1].start == pytest.approx(3.9)  # 3.6 + (5.4 - 5.1)
    assert all(w.end > w.start for w in hasil)


def test_remap_words_tanpa_potongan_tidak_mengubah_apa_pun():
    kata = [Word(1, 2, "a")]
    assert remap_words(kata, [])[0].start == 1


# ------------------------------------------------------ keyframe & ekspresi
def test_evaluate_keyframes_interpolasi_linear():
    kf = make_keyframes([(0, 100), (2, 300)])
    assert evaluate_keyframes(kf, 1) == pytest.approx(200)
    assert evaluate_keyframes(kf, -5) == pytest.approx(100)   # sebelum awal: datar
    assert evaluate_keyframes(kf, 99) == pytest.approx(300)   # sesudah akhir: datar


def test_ekspresi_ffmpeg_cocok_dengan_evaluasi_python():
    """Ekspresi untuk ffmpeg dan hitungan Python harus memberi angka yang sama."""

    kf = make_keyframes([(0, 540), (1.2, 480), (2.4, 620), (4.0, 300)])
    ekspresi = piecewise_expression(kf)

    # evaluator kecil untuk subset ekspresi yang kita hasilkan: if(lt(t,X),A,B)
    def hitung(teks: str, t: float) -> float:
        teks = teks.strip()
        if teks.startswith("if(lt(t,"):
            tutup = _cari_argumen(teks)
            ambang = float(tutup[0].split(",")[-1].rstrip(")"))
            return hitung(tutup[1], t) if t < ambang else hitung(tutup[2], t)
        return float(eval(teks.replace("t", repr(t)), {"__builtins__": {}}))  # noqa: S307

    def _cari_argumen(teks: str):
        isi = teks[len("if(") : -1]
        bagian, dalam, mulai = [], 0, 0
        for indeks, huruf in enumerate(isi):
            if huruf == "(":
                dalam += 1
            elif huruf == ")":
                dalam -= 1
            elif huruf == "," and dalam == 0:
                bagian.append(isi[mulai:indeks])
                mulai = indeks + 1
        bagian.append(isi[mulai:])
        return bagian

    for t in (0.0, 0.5, 1.2, 1.9, 2.4, 3.1, 4.0, 5.0):
        assert hitung(ekspresi, t) == pytest.approx(evaluate_keyframes(kf, t), abs=0.05)


def test_ekspresi_konstan_disederhanakan():
    assert piecewise_expression(make_keyframes([(0, 540), (5, 540)])) == "540"


def test_ekspresi_tidak_memakai_tanda_ganda():
    kf = make_keyframes([(0, 300), (1, 100)])
    assert "+-" not in piecewise_expression(kf)


def test_simplify_membuang_titik_di_garis_lurus():
    kf = make_keyframes([(0, 0), (1, 10), (2, 20), (3, 30)])
    assert len(simplify_keyframes(kf, tolerance=0.5)) == 2


def test_keyframe_waktu_ganda_diambil_yang_terakhir():
    kf = make_keyframes([(1.0, 10), (1.0, 20)])
    assert len(kf) == 1 and kf[0].value == 20


def test_evaluate_keyframes_kosong_melempar_error():
    with pytest.raises(ValueError):
        evaluate_keyframes([], 1.0)
