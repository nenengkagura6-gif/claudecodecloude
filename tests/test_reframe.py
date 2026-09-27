"""Test pemulusan gerak kamera dan pemilihan mode reframe."""

from __future__ import annotations

import pytest

from tukang_sayur.reframe import (
    MODE_BLUR,
    MODE_CENTER,
    Detection,
    ReframePlan,
    apply_deadzone,
    build_keyframes,
    clamp_positions,
    crop_size_for,
    exponential_smoothing,
    fallback_plan,
    interpolate_missing,
    median_filter,
    plan_reframe,
    smooth_centers,
)


def test_median_filter_membuang_lonjakan():
    hasil = median_filter([10, 500, 12, 11, 13], 3)
    assert 500 not in hasil
    assert all(9 <= nilai <= 14 for nilai in hasil)


def test_median_filter_menjaga_tepi():
    # lonjakan di frame pertama tidak boleh bocor ke hasil
    assert median_filter([999, 10, 11, 12], 3)[0] < 999


@pytest.mark.parametrize("data", [[], [5.0]])
def test_median_filter_data_minim(data):
    assert median_filter(data, 5) == data


def test_median_filter_tidak_menghapus_gerakan_sungguhan():
    """Perpindahan yang bertahan beberapa sampel harus tetap lolos."""

    hasil = median_filter([100, 100, 100, 400, 400, 400, 400], 3)
    assert hasil[0] == 100 and hasil[-1] == 400


def test_exponential_smoothing_mengikuti_pelan():
    hasil = exponential_smoothing([0, 100, 100, 100], 0.5)
    assert hasil == [0, 50, 75, 87.5]
    assert hasil[-1] < 100  # sengaja tidak langsung menempel


def test_exponential_smoothing_alpha_satu_sama_dengan_aslinya():
    assert exponential_smoothing([0, 100, 50], 1.0) == [0, 100, 50]


def test_deadzone_mengabaikan_gerakan_kecil():
    assert apply_deadzone([100, 110, 140, 145], 25) == [100, 100, 140, 140]


def test_deadzone_nol_tidak_mengubah():
    assert apply_deadzone([1, 2, 3], 0) == [1, 2, 3]


def test_clamp_positions_selalu_di_dalam_frame():
    posisi = clamp_positions([0, 960, 1920], crop_width=1080, frame_width=1920)
    assert posisi == [0.0, 420.0, 840.0]
    assert all(0 <= x <= 1920 - 1080 for x in posisi)


def test_crop_size_untuk_sumber_lanskap():
    lebar, tinggi = crop_size_for(1920, 1080, 1080 / 1920)
    assert tinggi == 1080 and 600 <= lebar <= 612
    assert lebar % 2 == 0 and tinggi % 2 == 0


def test_crop_size_untuk_sumber_sudah_vertikal():
    lebar, tinggi = crop_size_for(1080, 1920, 1080 / 1920)
    assert lebar == 1080 and tinggi <= 1920


def test_crop_size_frame_tidak_valid():
    assert crop_size_for(0, 0, 0.5625) == (0, 0)


def test_interpolate_missing_mengisi_celah():
    deteksi = [Detection(0, 100), Detection(2, 300)]
    hasil = interpolate_missing(deteksi, [0, 0.5, 1, 1.5, 2, 2.5], fallback_center=960)
    assert [round(d.center_x) for d in hasil] == [100, 150, 200, 250, 300, 300]


def test_interpolate_missing_tanpa_deteksi_pakai_tengah():
    hasil = interpolate_missing([], [0, 1], fallback_center=960)
    assert all(d.center_x == 960 and d.confidence == 0 for d in hasil)


def test_smooth_centers_menghasilkan_lintasan_halus():
    deteksi = [Detection(i * 0.2, 500 if i % 2 == 0 else 700) for i in range(12)]
    hasil = smooth_centers(deteksi, window=5, alpha=0.3, deadzone=10)
    loncatan = [abs(b[1] - a[1]) for a, b in zip(hasil, hasil[1:])]
    assert max(loncatan) < 200  # aslinya bolak-balik 200 px tiap frame


def test_smooth_centers_kosong():
    assert smooth_centers([], window=5, alpha=0.3, deadzone=10) == []


def test_build_keyframes_mengurangi_jumlah_titik():
    sampel = [(i * 0.2, 500 + i * 4) for i in range(30)]
    kf = build_keyframes(sampel, interval=0.4)
    assert 2 <= len(kf) < len(sampel)
    assert kf[0].time == pytest.approx(0.0)
    assert kf[-1].time == pytest.approx(sampel[-1][0])


def test_build_keyframes_sampel_kosong():
    assert build_keyframes([], interval=0.4) == []


# ------------------------------------------------------------------ rencana
def test_fallback_blur_dan_center(cfg):
    cfg.set("reframe.fallback", "blur")
    assert fallback_plan(cfg, 1920, 1080, "uji").mode == MODE_BLUR
    cfg.set("reframe.fallback", "center")
    assert fallback_plan(cfg, 1920, 1080, "uji").mode == MODE_CENTER


def test_plan_reframe_nonaktif_langsung_fallback(cfg, tmp_path):
    cfg.set("reframe.enabled", False)
    rencana = plan_reframe(tmp_path / "tidak-ada.mp4", cfg)
    assert rencana.mode in {MODE_BLUR, MODE_CENTER}
    assert not rencana.is_tracking


def test_plan_reframe_detector_none(cfg, tmp_path):
    cfg.set("reframe.detector", "none")
    assert not plan_reframe(tmp_path / "x.mp4", cfg).is_tracking


def test_rencana_fallback_memberi_ekspresi_aman():
    rencana = ReframePlan(mode=MODE_BLUR)
    assert rencana.crop_x_expression() == "0"
    assert "blur" in rencana.describe()


def test_rencana_tracking_menghasilkan_ekspresi_waktu():
    rencana = ReframePlan(
        mode="track", crop_width=608, crop_height=1080,
        keyframes=build_keyframes([(0, 100), (1, 300)], interval=0.4),
        detection_ratio=0.9,
    )
    ekspresi = rencana.crop_x_expression()
    assert rencana.is_tracking
    assert "if(lt(t," in ekspresi
    assert "keyframe" in rencana.describe()
