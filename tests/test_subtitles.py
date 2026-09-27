"""Test pembuatan subtitle ASS dan teks hook."""

from __future__ import annotations

import re

import pytest

from tukang_sayur.models import Word
from tukang_sayur.subtitles import (
    HookStyle,
    SubtitleStyle,
    build_ass,
    build_hook_events,
    build_word_events,
    escape_ass_text,
    format_ass_time,
    group_words_into_lines,
    inline_color,
    relativize_words,
    to_ass_color,
    write_ass,
)

DIALOGUE_RE = re.compile(
    r"^Dialogue: (\d+),(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),(\w+),"
)


def kata(mulai, selesai, teks):
    return Word(start=mulai, end=selesai, text=teks)


@pytest.fixture
def kalimat():
    return [
        kata(0.00, 0.40, "Ini"),
        kata(0.45, 0.90, "contoh"),
        kata(0.95, 1.50, "subtitle"),
        kata(1.60, 2.10, "karaoke."),
        kata(3.00, 3.40, "Baris"),
        kata(3.45, 3.90, "kedua"),
    ]


# ------------------------------------------------------------- format waktu
@pytest.mark.parametrize(
    "detik,harapan",
    [
        (0, "0:00:00.00"),
        (1.5, "0:00:01.50"),
        (59.999, "0:01:00.00"),   # pembulatan naik ikut membawa ke menit
        (61.23, "0:01:01.23"),
        (3661.234, "1:01:01.23"),
        (-5, "0:00:00.00"),       # negatif dijepit ke nol
        (36000, "10:00:00.00"),
    ],
)
def test_format_ass_time(detik, harapan):
    assert format_ass_time(detik) == harapan


def test_waktu_event_selalu_naik(kalimat):
    isi = build_ass(kalimat, style=SubtitleStyle(), clip_duration=5.0)
    waktu = [
        (m.group(2), m.group(3))
        for m in (DIALOGUE_RE.match(b) for b in isi.splitlines())
        if m
    ]
    assert waktu, "harus ada event"
    for mulai, selesai in waktu:
        assert mulai < selesai


# --------------------------------------------------------------- escaping
def test_escape_ass_text():
    assert escape_ass_text("halo {dunia}") == r"halo \{dunia\}"
    assert escape_ass_text("baris\nbaru") == r"baris\Nbaru"
    assert escape_ass_text("  spasi  ") == "spasi"


def test_teks_pengguna_tidak_bisa_menyuntik_tag(kalimat):
    isi = build_ass(
        [kata(0, 1, "{\\an8}bahaya")],
        style=SubtitleStyle(uppercase=False),
        clip_duration=2,
    )
    assert r"\{\\an8\}" in isi or r"\{" in isi
    assert "{\\an8}bahaya" not in isi


# ------------------------------------------------------------------ warna
@pytest.mark.parametrize(
    "masukan,harapan",
    [
        ("&H00FFFFFF", "&H00FFFFFF"),
        ("#FFCC00", "&H0000CCFF"),   # RGB -> BGR
        ("kuning", "&H0000FFFF"),
        ("white", "&H00FFFFFF"),
        ("ngawur", "&H00FFFFFF"),    # jatuh ke default, tidak error
    ],
)
def test_to_ass_color(masukan, harapan):
    assert to_ass_color(masukan) == harapan


def test_inline_color_format_tag():
    # tag override butuh 6 digit + diakhiri '&'
    assert inline_color("&H0000E5FF") == "&H00E5FF&"
    assert inline_color("&H00FFFFFF") == "&HFFFFFF&"


# --------------------------------------------------------- pengelompokan
def test_group_words_batas_jumlah_kata(kalimat):
    baris = group_words_into_lines(kalimat, max_words=2, max_chars=99)
    assert all(len(b) <= 2 for b in baris)


def test_group_words_batas_karakter():
    kata_kata = [kata(i * 0.3, i * 0.3 + 0.25, "duabelas") for i in range(6)]
    baris = group_words_into_lines(kata_kata, max_words=9, max_chars=20)
    assert all(len(" ".join(w.text for w in b)) <= 20 for b in baris)


def test_group_words_memotong_saat_jeda_panjang():
    kata_kata = [kata(0, 0.4, "satu"), kata(0.45, 0.9, "dua"), kata(3.0, 3.4, "tiga")]
    baris = group_words_into_lines(kata_kata, max_words=9, max_chars=99)
    assert len(baris) == 2 and baris[1][0].text == "tiga"


def test_group_words_memotong_setelah_akhir_kalimat():
    kata_kata = [kata(0, 0.4, "selesai."), kata(0.45, 0.9, "mulai")]
    baris = group_words_into_lines(kata_kata, max_words=9, max_chars=99)
    assert len(baris) == 2


def test_kata_kosong_diabaikan():
    assert group_words_into_lines([kata(0, 1, "   ")]) == []


# ------------------------------------------------------------- highlight
def test_setiap_kata_mendapat_giliran_highlight(kalimat):
    gaya = SubtitleStyle(karaoke=True, max_words_per_line=4, max_chars_per_line=40)
    baris = group_words_into_lines(kalimat, max_words=4, max_chars=40)
    event = build_word_events(baris, gaya, clip_duration=5)
    # satu event per kata
    assert len(event) == len(kalimat)
    sorotan = inline_color(gaya.highlight_color)
    for teks_kata, baris_event in zip([w.text.upper() for w in kalimat], event):
        assert f"{{\\c{sorotan}" in baris_event
        assert teks_kata in baris_event


def test_mode_tanpa_karaoke_satu_event_per_baris(kalimat):
    gaya = SubtitleStyle(karaoke=False, max_words_per_line=4, max_chars_per_line=40)
    baris = group_words_into_lines(kalimat, max_words=4, max_chars=40)
    event = build_word_events(baris, gaya, clip_duration=5)
    assert len(event) == len(baris) < len(kalimat)


def test_event_tidak_melewati_durasi_klip(kalimat):
    isi = build_ass(kalimat, style=SubtitleStyle(), clip_duration=2.0)
    for baris in isi.splitlines():
        cocok = DIALOGUE_RE.match(baris)
        if cocok:
            assert cocok.group(3) <= "0:00:02.00"


def test_huruf_besar_bisa_dimatikan(kalimat):
    isi = build_ass(kalimat, style=SubtitleStyle(uppercase=False), clip_duration=5)
    assert "contoh" in isi and "CONTOH" not in isi


# ------------------------------------------------------------------- hook
def test_hook_mulai_dari_detik_nol():
    (event,) = build_hook_events("Jangan sampai kelewatan", HookStyle(), clip_duration=30)
    cocok = DIALOGUE_RE.match(event)
    assert cocok and cocok.group(2) == "0:00:00.00"
    assert cocok.group(4) == "Hook"


def test_hook_dipendekkan_kalau_klip_pendek():
    (event,) = build_hook_events("Hook", HookStyle(duration=3.0), clip_duration=1.0)
    assert DIALOGUE_RE.match(event).group(3) == "0:00:00.90"


def test_hook_dibungkus_beberapa_baris():
    (event,) = build_hook_events(
        "Hook yang sangat panjang sekali sampai harus dibungkus",
        HookStyle(max_chars_per_line=12),
        clip_duration=30,
    )
    assert event.count("\\N") >= 2


def test_hook_kosong_atau_nonaktif_tidak_menghasilkan_event():
    assert build_hook_events("", HookStyle()) == []
    assert build_hook_events("ada teks", HookStyle(enabled=False)) == []


def test_hook_ada_di_layer_atas_subtitle(kalimat):
    isi = build_ass(
        kalimat, style=SubtitleStyle(), hook_style=HookStyle(), hook_text="Hook",
        clip_duration=5,
    )
    hook_baris = [b for b in isi.splitlines() if ",Hook," in b][0]
    sub_baris = [b for b in isi.splitlines() if ",Sub," in b][0]
    assert hook_baris.startswith("Dialogue: 1,")
    assert sub_baris.startswith("Dialogue: 0,")


# --------------------------------------------------------------- file ASS
def test_struktur_file_ass_lengkap(kalimat):
    isi = build_ass(
        kalimat,
        style=SubtitleStyle(),
        hook_style=HookStyle(),
        hook_text="Hook uji",
        width=1080,
        height=1920,
        clip_duration=5,
    )
    assert "[Script Info]" in isi
    assert "PlayResX: 1080" in isi and "PlayResY: 1920" in isi
    assert "[V4+ Styles]" in isi and "[Events]" in isi
    # hitung baris gaya saja (jangan ikut "WrapStyle: 2" di header)
    assert len([b for b in isi.splitlines() if b.startswith("Style: ")]) == 2
    assert "Format: Layer, Start, End" in isi


def test_gaya_dari_config_terpakai(cfg, kalimat):
    cfg.set("subtitle.font_name", "Impact")
    cfg.set("subtitle.font_size", 90)
    cfg.set("subtitle.highlight_color", "#FF0000")
    gaya = SubtitleStyle.from_config(cfg)
    isi = build_ass(kalimat, style=gaya, clip_duration=5)
    assert "Impact,90" in isi
    assert inline_color("&H000000FF") in isi


def test_write_ass_menulis_berkas(tmp_path, kalimat):
    target = write_ass(
        tmp_path / "klip.ass", kalimat, style=SubtitleStyle(), clip_duration=5
    )
    assert target.is_file() and target.read_text(encoding="utf-8").startswith("[Script Info]")


def test_relativize_words_menggeser_dan_membuang_yang_di_luar():
    kata_kata = [kata(10, 11, "a"), kata(12, 13, "b"), kata(5, 6, "sebelum")]
    hasil = relativize_words(kata_kata, 10.0)
    assert [(w.start, w.text) for w in hasil] == [(0.0, "a"), (2.0, "b")]


def test_tanpa_kata_tetap_menghasilkan_file_valid():
    isi = build_ass([], style=SubtitleStyle(), hook_style=HookStyle(), hook_text="Cuma hook")
    assert "[Events]" in isi and ",Hook," in isi


# ------------------------------------------------- penskalaan resolusi
def test_ukuran_gaya_diskalakan_mengikuti_tinggi_render():
    """Angka px di config dianggap untuk 1080x1920; resolusi lain diskalakan."""

    dari_1920 = build_ass([kata(0, 1, "uji")], style=SubtitleStyle(font_size=64),
                          width=1080, height=1920, clip_duration=2)
    dari_1280 = build_ass([kata(0, 1, "uji")], style=SubtitleStyle(font_size=64),
                          width=720, height=1280, clip_duration=2)

    def ukuran(isi: str) -> int:
        baris = [b for b in isi.splitlines() if b.startswith("Style: Sub")][0]
        return int(baris.split(",")[2])

    assert ukuran(dari_1920) == 64
    assert ukuran(dari_1280) == pytest.approx(43, abs=1)  # 64 * 1280/1920


def test_margin_ikut_diskalakan():
    isi = build_ass([kata(0, 1, "uji")], style=SubtitleStyle(margin_bottom=420),
                    width=540, height=960, clip_duration=2)
    baris = [b for b in isi.splitlines() if b.startswith("Style: Sub")][0]
    assert int(baris.split(",")[-2]) == pytest.approx(210, abs=1)


def test_hook_juga_diskalakan():
    isi = build_ass([], style=SubtitleStyle(), hook_style=HookStyle(font_size=78),
                    hook_text="Hook", width=540, height=960, clip_duration=2)
    baris = [b for b in isi.splitlines() if b.startswith("Style: Hook")][0]
    assert int(baris.split(",")[2]) == pytest.approx(39, abs=1)


def test_penskalaan_tidak_pernah_nol():
    gaya = SubtitleStyle(font_size=10).scaled(0.01)
    assert gaya.font_size >= 8
