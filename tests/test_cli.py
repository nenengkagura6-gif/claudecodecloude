"""Test antarmuka terminal (tanpa merender apa pun)."""

from __future__ import annotations

import pytest

from tukang_sayur import cli
from tukang_sayur.errors import TukangSayurError
from tukang_sayur.models import JobResult


def parse(args):
    return cli.build_parser().parse_args(args)


def test_perintah_clip_argumen_dasar():
    args = parse(["clip", "https://a.test/v", "--max-clips", "7"])
    assert args.command == "clip" and args.source == "https://a.test/v"
    assert args.max_clips == 7


def test_perintah_tanpa_subcommand_gagal():
    with pytest.raises(SystemExit):
        parse([])


def test_override_provider_dan_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse(["clip", "v.mp4", "--provider", "ollama", "--model", "qwen2.5:7b"])
    cfg = cli.config_from_args(args)
    assert cfg.provider_name() == "ollama"
    assert cfg.provider_config()["model"] == "qwen2.5:7b"


def test_override_lewat_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse(["clip", "v.mp4", "--set", "render.crf=18", "--set", "hook.enabled=false"])
    cfg = cli.config_from_args(args)
    assert cfg.get("render.crf") == 18 and cfg.get("hook.enabled") is False


def test_opsi_watermark_mengaktifkan_watermark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse(["clip", "v.mp4", "--watermark", "logo.png"])
    cfg = cli.config_from_args(args)
    assert cfg.get("watermark.enabled") is True
    assert cfg.get("watermark.path") == "logo.png"


def test_job_options_dari_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse(["clip", "v.mp4", "--max-clips", "3", "--no-hook", "--dry-run",
                  "--keep-intermediate", "-o", str(tmp_path / "keluaran")])
    opsi = cli.job_options_from_args(args)
    assert opsi.max_clips == 3 and opsi.no_hook and opsi.dry_run
    assert opsi.keep_intermediate is True
    assert opsi.output_dir == tmp_path / "keluaran"


def test_main_clip_memanggil_pipeline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    dipanggil = {}

    def palsu(source, config, **kwargs):
        dipanggil["source"] = source
        job = JobResult(source=source, video_path="v.mp4", output_dir=str(tmp_path))
        job.best_score = 81.0
        return job

    import tukang_sayur.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "run_clip_job", palsu)
    kode = cli.main(["clip", "video.mp4", "--provider", "mock"])
    assert kode == 0 and dipanggil["source"] == "video.mp4"
    assert "81.0" in capsys.readouterr().out


def test_main_mengubah_error_jadi_kode_keluar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def gagal(source, config, **kwargs):
        raise TukangSayurError("ffmpeg tidak ada")

    import tukang_sayur.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "run_clip_job", gagal)
    assert cli.main(["clip", "video.mp4", "--provider", "mock"]) == 1


def test_perintah_doctor_jalan(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    kode = cli.main(["doctor"])
    keluaran = capsys.readouterr().out
    assert "ffmpeg" in keluaran and kode in {0, 1}


def test_perintah_batch_argumen():
    args = parse(["batch", "daftar.txt", "--no-resume", "--max-clips", "4"])
    assert args.list_file == "daftar.txt" and args.no_resume is True


def test_perintah_ui_dan_serve_argumen():
    assert parse(["ui", "--port", "7000"]).port == 7000
    assert parse(["serve", "--host", "0.0.0.0"]).host == "0.0.0.0"
