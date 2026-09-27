"""Antarmuka web sederhana (Gradio) untuk yang tidak mau pakai terminal.

Jalankan::

    python -m tukang_sayur ui

Lalu buka http://127.0.0.1:7860 di browser. Semua pengaturan penting ada di
halaman itu: sumber video, jumlah klip, provider LLM, model Whisper, gaya
subtitle, dan tombol proses. Hasilnya langsung bisa diputar dan diunduh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import Config
from .errors import TukangSayurError
from .logging_utils import get_logger
from .models import JobResult

log = get_logger("webapp")

_INTRO = """
# Tukang Sayur

Potong video panjang jadi klip vertikal 9:16 siap unggah ke TikTok, Reels, dan Shorts.

**Cara pakai:** tempel URL video **atau** unggah file, atur jumlah klip, lalu klik **Proses**.
Transkripsi di CPU memang lama (video 10 menit bisa 3-8 menit dengan model `small`),
jadi tunggu sampai bar progres selesai.
"""

_TIPS = """
### Tips singkat

- Belum punya kunci API? Pilih provider **ollama** (jalan lokal, gratis) atau **mock** (untuk coba-coba).
- Video tanpa wajah otomatis memakai latar blur, jadi tetap jadi 9:16.
- Hasil, thumbnail, `metadata.json`, dan `metadata.csv` ada di folder keluaran yang tertera di ringkasan.
"""


def _import_gradio():
    try:
        import gradio as gr  # type: ignore[import-not-found]

        return gr
    except ImportError as exc:  # pragma: no cover
        raise TukangSayurError(
            "Gradio belum terpasang. Install dulu: pip install gradio"
        ) from exc


def _summary_markdown(job: JobResult) -> str:
    lines = [
        f"### Selesai: {len(job.clips)} klip",
        "",
        f"- Skor terbaik: **{job.best_score:.1f}**",
        f"- Kandidat diperiksa: {job.candidate_count}",
        f"- Bahasa terdeteksi: `{job.language or '-'}`",
        f"- Folder keluaran: `{job.output_dir}`",
        f"- Metadata: `{Path(job.metadata_csv).name if job.metadata_csv else '-'}`",
    ]
    if job.warnings:
        lines.append("")
        lines.append("**Catatan:**")
        lines.extend(f"- {warning}" for warning in job.warnings)
    return "\n".join(lines)


def _table_rows(job: JobResult) -> List[List[Any]]:
    return [
        [
            clip.index,
            round(clip.score, 1),
            f"{clip.duration:.0f}s",
            clip.hook_text,
            clip.title,
            " ".join(f"#{tag.lstrip('#')}" for tag in clip.hashtags[:6]),
            Path(clip.video_path).name if clip.video_path else "",
        ]
        for clip in job.clips
    ]


def build_interface(cfg: Config):
    """Susun halaman Gradio."""

    gr = _import_gradio()
    from .pipeline import ClipJobOptions, run_clip_job

    def process(
        url: str,
        uploaded: Optional[str],
        max_clips: int,
        provider: str,
        whisper_model: str,
        language: str,
        min_duration: float,
        max_duration: float,
        remove_silence: bool,
        with_hook: bool,
        with_subtitles: bool,
        font_size: int,
        highlight_color: str,
        progress=gr.Progress(track_tqdm=False),
    ) -> Tuple[str, List[List[Any]], List[str]]:
        source = (url or "").strip() or (uploaded or "")
        if not source:
            return "**Isi dulu URL video atau unggah file.**", [], []

        job_cfg = Config.load(
            cfg.path,
            overrides={
                "llm.provider": provider,
                "transcription.model": whisper_model,
                "transcription.language": language,
                "candidates.min_duration": float(min_duration),
                "candidates.max_duration": float(max_duration),
                "audio.remove_silence": bool(remove_silence),
                "hook.enabled": bool(with_hook),
                "subtitle.enabled": bool(with_subtitles),
                "subtitle.font_size": int(font_size),
                "subtitle.highlight_color": highlight_color,
            },
        )

        def report(stage: str, message: str, fraction: float) -> None:
            progress(min(1.0, max(0.0, fraction)), desc=f"{stage}: {message}")

        try:
            job = run_clip_job(
                source,
                job_cfg,
                options=ClipJobOptions(max_clips=int(max_clips)),
                progress=report,
            )
        except TukangSayurError as exc:
            log.error("UI gagal: %s", exc)
            return f"### Gagal\n\n```\n{exc}\n```", [], []

        videos = [clip.video_path for clip in job.clips if clip.video_path]
        return _summary_markdown(job), _table_rows(job), videos

    with gr.Blocks(title="Tukang Sayur", theme=gr.themes.Soft()) as demo:
        gr.Markdown(_INTRO)

        with gr.Row():
            with gr.Column(scale=3):
                url = gr.Textbox(
                    label="URL video",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                uploaded = gr.Video(label="atau unggah file video", sources=["upload"])
                with gr.Row():
                    max_clips = gr.Slider(
                        1, 15, value=int(cfg.get("candidates.max_clips", 10)),
                        step=1, label="Jumlah klip",
                    )
                    provider = gr.Dropdown(
                        ["anthropic", "openai", "gemini", "ollama", "mock"],
                        value=cfg.provider_name(),
                        label="Provider LLM",
                    )
                with gr.Row():
                    whisper_model = gr.Dropdown(
                        ["tiny", "base", "small", "medium"],
                        value=str(cfg.get("transcription.model", "small")),
                        label="Model Whisper",
                    )
                    language = gr.Dropdown(
                        ["id", "su", "auto"],
                        value=str(cfg.get("transcription.language", "id")),
                        label="Bahasa",
                    )
                with gr.Accordion("Pengaturan lanjutan", open=False):
                    with gr.Row():
                        min_duration = gr.Slider(
                            5, 120, value=float(cfg.get("candidates.min_duration", 20)),
                            step=1, label="Durasi minimum (detik)",
                        )
                        max_duration = gr.Slider(
                            10, 180, value=float(cfg.get("candidates.max_duration", 90)),
                            step=1, label="Durasi maksimum (detik)",
                        )
                    with gr.Row():
                        remove_silence = gr.Checkbox(
                            value=bool(cfg.get("audio.remove_silence", False)),
                            label="Potong jeda hening",
                        )
                        with_hook = gr.Checkbox(
                            value=bool(cfg.get("hook.enabled", True)), label="Pakai hook"
                        )
                        with_subtitles = gr.Checkbox(
                            value=bool(cfg.get("subtitle.enabled", True)),
                            label="Pakai subtitle",
                        )
                    with gr.Row():
                        font_size = gr.Slider(
                            32, 110, value=int(cfg.get("subtitle.font_size", 64)),
                            step=2, label="Ukuran font subtitle",
                        )
                        highlight_color = gr.Textbox(
                            value=str(cfg.get("subtitle.highlight_color", "&H0000E5FF")),
                            label="Warna kata aktif (&HAABBGGRR atau #RRGGBB)",
                        )
                run_button = gr.Button("Proses", variant="primary", size="lg")

            with gr.Column(scale=2):
                summary = gr.Markdown("Hasil akan muncul di sini.")
                gr.Markdown(_TIPS)

        table = gr.Dataframe(
            headers=["#", "Skor", "Durasi", "Hook", "Judul", "Hashtag", "File"],
            label="Daftar klip",
            wrap=True,
        )
        gallery = gr.Files(label="Unduh klip")

        run_button.click(
            fn=process,
            inputs=[
                url, uploaded, max_clips, provider, whisper_model, language,
                min_duration, max_duration, remove_silence, with_hook,
                with_subtitles, font_size, highlight_color,
            ],
            outputs=[summary, table, gallery],
        )

    return demo


def launch_ui(
    cfg: Config,
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    share: bool = False,
) -> int:
    """Jalankan server Gradio (dipanggil dari ``python -m tukang_sayur ui``)."""

    demo = build_interface(cfg)
    host = host or str(cfg.get("ui.host", "127.0.0.1"))
    port = int(port or cfg.get("ui.port", 7860))
    share = bool(share or cfg.get("ui.share", False))
    log.info("Menjalankan UI di http://%s:%d", host, port)
    demo.launch(server_name=host, server_port=port, share=share, show_api=False)
    return 0
