"""Test penyusun perintah ffmpeg (tanpa benar-benar menjalankan ffmpeg)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tukang_sayur.audio import Range
from tukang_sayur.reframe import MODE_BLUR, MODE_CENTER, MODE_TRACK, ReframePlan, build_keyframes
from tukang_sayur.render import (
    RenderRequest,
    build_audio_chain,
    build_ffmpeg_command,
    build_video_chain,
    escape_filter_path,
    resolve_watermark,
)


@pytest.fixture
def rencana_track():
    return ReframePlan(
        mode=MODE_TRACK, crop_width=608, crop_height=1080,
        frame_width=1920, frame_height=1080,
        keyframes=build_keyframes([(0, 100), (2, 400)], interval=0.4),
    )


def gabung(bagian):
    return ";".join(bagian)


# ------------------------------------------------------------------ escaping
@pytest.mark.parametrize(
    "masukan,harus_ada",
    [
        (r"C:\kerja\klip.ass", "C\\:/kerja/klip.ass"),
        ("/home/user/a b.ass", "/home/user/a b.ass"),
        ("/tmp/klip:1.ass", "/tmp/klip\\:1.ass"),
    ],
)
def test_escape_filter_path(masukan, harus_ada):
    assert escape_filter_path(masukan) == harus_ada


# ------------------------------------------------------------- rantai video
def test_rantai_track_memotong_setelah_crop(cfg, rencana_track):
    """Ekspresi crop memakai waktu `t`, jadi crop harus sebelum select/setpts."""

    bagian, keluaran = build_video_chain(
        rencana_track, cfg, keep_ranges=[(0.0, 3.0), (4.0, 8.0)]
    )
    teks = gabung(bagian)
    assert teks.index("crop=w=608") < teks.index("select=")
    assert "setpts=N/FRAME_RATE/TB" in teks
    assert keluaran == "vout"


def test_rantai_blur_memotong_lebih_dulu(cfg):
    bagian = build_video_chain(ReframePlan(mode=MODE_BLUR), cfg, keep_ranges=[(0.0, 3.0)])[0]
    teks = gabung(bagian)
    assert teks.index("select=") < teks.index("boxblur")
    assert "split=2" in teks and "overlay=" in teks


def test_rantai_blur_menghasilkan_ukuran_target(cfg):
    teks = gabung(build_video_chain(ReframePlan(mode=MODE_BLUR), cfg)[0])
    assert f"scale={cfg.get('render.width')}:{cfg.get('render.height')}" in teks
    assert "format=yuv420p[vout]" in teks


def test_rantai_center(cfg):
    teks = gabung(build_video_chain(ReframePlan(mode=MODE_CENTER), cfg)[0])
    assert "force_original_aspect_ratio=increase" in teks and "boxblur" not in teks


def test_rantai_dengan_subtitle_dan_watermark(cfg, rencana_track, tmp_path):
    ass = tmp_path / "klip.ass"
    teks = gabung(
        build_video_chain(rencana_track, cfg, ass_path=ass, watermark=True)[0]
    )
    assert f"ass=filename='{escape_filter_path(ass)}'" in teks
    assert "colorchannelmixer=aa=" in teks
    assert teks.index("ass=filename") < teks.index("[wm]")  # watermark di atas subtitle


def test_posisi_watermark_mengikuti_config(cfg, rencana_track):
    cfg.set("watermark.position", "bottom-left")
    cfg.set("watermark.margin", 55)
    teks = gabung(build_video_chain(rencana_track, cfg, watermark=True)[0])
    assert "overlay=x=55:y=main_h-overlay_h-55" in teks


def test_tanpa_pemotongan_tidak_ada_select(cfg, rencana_track):
    assert "select=" not in gabung(build_video_chain(rencana_track, cfg)[0])


# ------------------------------------------------------------- rantai audio
def test_rantai_audio_lengkap(cfg):
    bagian, label = build_audio_chain(
        cfg,
        keep_ranges=[(0.0, 3.0)],
        loudness={"input_i": -22.1, "input_tp": -3.0, "input_lra": 7.1,
                  "input_thresh": -32.2, "target_offset": 0.3},
    )
    teks = gabung(bagian)
    assert label == "aout"
    assert "aselect=" in teks and "asetpts=N/SR/TB" in teks
    assert "loudnorm=I=-14" in teks and "measured_i=-22.1" in teks and "linear=true" in teks


def test_rantai_audio_tanpa_loudnorm(cfg):
    cfg.set("audio.loudnorm", False)
    teks = gabung(build_audio_chain(cfg)[0])
    assert "loudnorm" not in teks and "aresample=" in teks


def test_rantai_audio_saat_sumber_tanpa_audio(cfg):
    bagian, label = build_audio_chain(cfg, has_audio=False)
    assert bagian == [] and label is None


# ----------------------------------------------------------- perintah utuh
def test_perintah_ffmpeg_lengkap(cfg, rencana_track, tmp_path):
    permintaan = RenderRequest(
        clip_path=tmp_path / "raw.mp4",
        output_path=tmp_path / "final.mp4",
        cfg=cfg,
        plan=rencana_track,
        ass_path=tmp_path / "klip.ass",
        keep_ranges=[(0.0, 5.0)],
    )
    perintah = build_ffmpeg_command(permintaan)
    assert "-filter_complex" in perintah
    assert perintah[perintah.index("-map") + 1] == "[vout]"
    assert "[aout]" in perintah
    assert perintah[-1] == str(tmp_path / "final.mp4")
    assert "+faststart" in perintah


def test_perintah_tanpa_audio_memakai_an(cfg, rencana_track, tmp_path):
    permintaan = RenderRequest(
        clip_path=tmp_path / "raw.mp4", output_path=tmp_path / "out.mp4",
        cfg=cfg, plan=rencana_track,
    )
    perintah = build_ffmpeg_command(permintaan, has_audio=False)
    assert "-an" in perintah and "[aout]" not in perintah


def test_perintah_dengan_watermark_menambah_input(cfg, rencana_track, tmp_path):
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"palsu")
    permintaan = RenderRequest(
        clip_path=tmp_path / "raw.mp4", output_path=tmp_path / "out.mp4",
        cfg=cfg, plan=rencana_track, watermark_path=logo,
    )
    perintah = build_ffmpeg_command(permintaan)
    assert perintah.count("-i") == 2


# ------------------------------------------------------------- watermark
def test_resolve_watermark_nonaktif(cfg):
    assert resolve_watermark(cfg) is None


def test_resolve_watermark_file_tidak_ada(cfg, tmp_path):
    cfg.set("watermark.enabled", True)
    cfg.set("watermark.path", str(tmp_path / "hilang.png"))
    assert resolve_watermark(cfg) is None  # diperingatkan, tapi tidak bikin gagal


def test_resolve_watermark_file_ada(cfg, tmp_path):
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"palsu")
    cfg.set("watermark.enabled", True)
    cfg.set("watermark.path", str(logo))
    assert resolve_watermark(cfg) == logo
