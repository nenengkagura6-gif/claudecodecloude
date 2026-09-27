"""Pembuat video contoh (dummy) untuk test dan uji coba tanpa mengunduh apa pun.

Video dibuat murni dengan ffmpeg lavfi: gambar bergerak (``testsrc2``) plus nada
sinus. Di sampingnya ditulis file ``.txt`` berisi naskah, yang akan dibaca
:class:`~tukang_sayur.transcribe.DummyTranscriber` sebagai "hasil transkripsi".
Jadi pipeline penuh bisa dijalankan offline: ingest -> transkrip -> kandidat
(mock LLM) -> render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .errors import RenderError
from .logging_utils import get_logger
from .utils import ffmpeg_bin, run_command

log = get_logger("demo")

DEMO_SCRIPT = (
    "Selamat datang di dapur Tukang Sayur. "
    "Hari ini kita bahas cara memilih sayur yang masih segar. "
    "Pertama, lihat batangnya; kalau masih kaku berarti baru dipetik. "
    "Kedua, cium aromanya, sayur segar itu wangi tanahnya masih terasa. "
    "Ketiga, jangan pilih yang daunnya menguning di pangkal. "
    "Ternyata rahasianya sederhana sekali, tapi jarang ada yang tahu. "
    "Banyak orang salah karena cuma melihat harga yang paling murah. "
    "Padahal sayur murah yang sudah lemas justru jadi lebih mahal. "
    "Karena setengahnya berakhir di tempat sampah, bukan di meja makan. "
    "Coba praktikkan besok pagi waktu ke pasar, lalu rasakan bedanya. "
    "Kalau menurut kalian ini berguna, simpan dulu videonya. "
    "Minggu depan kita bahas cara menyimpan sayur supaya tahan seminggu."
)


def make_dummy_video(
    output: str | Path,
    *,
    duration: float = 90.0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    script: Optional[str] = DEMO_SCRIPT,
) -> Path:
    """Bikin video contoh lanskap (default 90 detik, 1280x720)."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={duration:g}",
        "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={duration:g}",
        "-shortest",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-ar", "48000",
        "-movflags", "+faststart",
        str(output),
    ]
    run_command(cmd, timeout=600, error_cls=RenderError, label="dummy-video")
    if not output.is_file() or output.stat().st_size == 0:
        raise RenderError(f"Video contoh gagal dibuat: {output}")

    if script:
        sidecar = output.with_suffix(".txt")
        sidecar.write_text(script.strip() + "\n", encoding="utf-8")
        log.debug("Naskah contoh ditulis: %s", sidecar)

    log.info("Video contoh siap: %s (%.0f detik, %dx%d)", output, duration, width, height)
    return output
