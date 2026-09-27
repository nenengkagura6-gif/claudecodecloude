"""Pembuat subtitle ASS: per kata, dengan highlight kata yang sedang diucapkan.

Kenapa ASS dan bukan SRT? Karena kita butuh kendali gaya (font besar, outline
tebal, posisi, warna kata aktif) yang tidak ada di SRT, dan ffmpeg bisa
membakarnya langsung lewat filter ``ass``.

Cara highlight-nya: untuk setiap baris subtitle, dibuat satu event per kata.
Seluruh baris tetap tampil, tapi kata yang sedang diucapkan diberi warna
berbeda (dan sedikit diperbesar). Hasilnya mirip gaya "karaoke" yang biasa
dipakai klip TikTok, tanpa bergantung pada dukungan ``\\k`` di renderer.

Hook juga ditulis sebagai event ASS dengan gaya sendiri, ditampilkan sejak
detik 0 **di atas video yang tetap berjalan** - tidak ada freeze-frame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import Config
from .logging_utils import get_logger
from .models import Word
from .utils import atomic_write_text, wrap_text

log = get_logger("subtitles")

#: Jeda maksimum (detik) sebelum baris subtitle dipotong jadi baris baru.
LINE_BREAK_GAP = 0.65

#: Tinggi acuan gaya. Semua angka px di config (font_size, margin, outline)
#: dianggap untuk kanvas 1080x1920. Kalau render.height diubah - misalnya ke
#: 720x1280 supaya file lebih kecil - ukurannya diskalakan otomatis, jadi
#: tampilannya tetap sama proporsinya.
REFERENCE_HEIGHT = 1920

_NAMED_COLORS = {
    "white": "&H00FFFFFF", "putih": "&H00FFFFFF",
    "black": "&H00000000", "hitam": "&H00000000",
    "yellow": "&H0000FFFF", "kuning": "&H0000FFFF",
    "green": "&H0000FF00", "hijau": "&H0000FF00",
    "red": "&H000000FF", "merah": "&H000000FF",
    "cyan": "&H00FFFF00", "biru": "&H00FF0000", "blue": "&H00FF0000",
    "orange": "&H0000A5FF", "oranye": "&H0000A5FF",
    "pink": "&H00CB7FFF", "magenta": "&H00FF00FF",
}


# --------------------------------------------------------------------------
# Warna & waktu
# --------------------------------------------------------------------------
def to_ass_color(value: str, default: str = "&H00FFFFFF") -> str:
    """Terima ``&H00BBGGRR``, ``#RRGGBB``, ``#RRGGBBAA``, atau nama warna.

    ASS memakai urutan BGR (bukan RGB) dan alpha terbalik (00 = solid).
    """

    text = str(value or "").strip()
    if not text:
        return default
    if text.lower().startswith("&h"):
        return "&H" + text[2:].upper().rstrip("&") + ""
    named = _NAMED_COLORS.get(text.lower())
    if named:
        return named
    hexed = text.lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{6}", hexed):
        red, green, blue = hexed[0:2], hexed[2:4], hexed[4:6]
        return f"&H00{blue}{green}{red}".upper()
    if re.fullmatch(r"[0-9a-fA-F]{8}", hexed):
        red, green, blue, alpha = hexed[0:2], hexed[2:4], hexed[4:6], hexed[6:8]
        inverted = f"{255 - int(alpha, 16):02X}"
        return f"&H{inverted}{blue}{green}{red}".upper()
    log.warning("Warna '%s' tidak dikenali, memakai default %s", value, default)
    return default


def inline_color(ass_color: str) -> str:
    """Ubah warna gaya ``&HAABBGGRR`` jadi bentuk tag inline ``&HBBGGRR&``.

    Tag override ASS (``\\c``) memakai 6 digit tanpa alpha dan diakhiri ``&``;
    kalau formatnya salah, libass bisa mengabaikan warnanya.
    """

    text = str(ass_color or "").strip().upper().rstrip("&")
    if text.startswith("&H"):
        text = text[2:]
    digits = re.sub(r"[^0-9A-F]", "", text)
    if len(digits) >= 8:
        digits = digits[-6:]
    elif len(digits) < 6:
        digits = digits.rjust(6, "0")
    return f"&H{digits}&"


def format_ass_time(seconds: float) -> str:
    """Format waktu ASS ``H:MM:SS.cc`` (centisecond, bukan milidetik)."""

    total_cs = int(round(max(0.0, float(seconds)) * 100))
    hours, remainder = divmod(total_cs, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def escape_ass_text(text: str) -> str:
    """Amankan teks agar tidak dianggap tag override ASS."""

    cleaned = str(text or "")
    cleaned = cleaned.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    cleaned = cleaned.replace("\r\n", "\\N").replace("\n", "\\N").replace("\r", "\\N")
    return cleaned.strip()


# --------------------------------------------------------------------------
# Gaya
# --------------------------------------------------------------------------
@dataclass
class SubtitleStyle:
    """Gaya subtitle utama (semua bisa diatur dari ``config.yaml``)."""

    font_name: str = "Montserrat"
    font_size: int = 64
    primary_color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000E5FF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    outline: float = 4
    shadow: float = 1
    bold: bool = True
    italic: bool = False
    uppercase: bool = True
    max_chars_per_line: int = 22
    max_words_per_line: int = 4
    margin_bottom: int = 420
    margin_horizontal: int = 90
    alignment: int = 2
    karaoke: bool = True
    highlight_scale: int = 108
    border_style: int = 1  # 1 = teks ber-outline, 3 = kotak penuh

    @classmethod
    def from_config(cls, cfg: Config) -> "SubtitleStyle":
        return cls(
            font_name=str(cfg.get("subtitle.font_name", "Montserrat")),
            font_size=int(cfg.get("subtitle.font_size", 64)),
            primary_color=to_ass_color(cfg.get("subtitle.primary_color", "&H00FFFFFF")),
            highlight_color=to_ass_color(
                cfg.get("subtitle.highlight_color", "&H0000E5FF"), "&H0000E5FF"
            ),
            outline_color=to_ass_color(cfg.get("subtitle.outline_color", "&H00000000"), "&H00000000"),
            back_color=to_ass_color(cfg.get("subtitle.back_color", "&H80000000"), "&H80000000"),
            outline=float(cfg.get("subtitle.outline", 4)),
            shadow=float(cfg.get("subtitle.shadow", 1)),
            bold=bool(cfg.get("subtitle.bold", True)),
            uppercase=bool(cfg.get("subtitle.uppercase", True)),
            max_chars_per_line=int(cfg.get("subtitle.max_chars_per_line", 22)),
            max_words_per_line=int(cfg.get("subtitle.max_words_per_line", 4)),
            margin_bottom=int(cfg.get("subtitle.margin_bottom", 420)),
            margin_horizontal=int(cfg.get("subtitle.margin_horizontal", 90)),
            alignment=int(cfg.get("subtitle.alignment", 2)),
            karaoke=bool(cfg.get("subtitle.karaoke", True)),
            border_style=int(cfg.get("subtitle.border_style", 1)),
        )

    def scaled(self, factor: float) -> "SubtitleStyle":
        """Salinan gaya dengan ukuran px diskalakan (lihat REFERENCE_HEIGHT)."""

        if abs(factor - 1.0) < 1e-6:
            return self
        return replace(
            self,
            font_size=max(8, int(round(self.font_size * factor))),
            outline=round(self.outline * factor, 2),
            shadow=round(self.shadow * factor, 2),
            margin_bottom=max(0, int(round(self.margin_bottom * factor))),
            margin_horizontal=max(0, int(round(self.margin_horizontal * factor))),
        )

    def style_line(self, name: str = "Sub") -> str:
        return (
            f"Style: {name},{self.font_name},{self.font_size},{self.primary_color},"
            f"{self.highlight_color},{self.outline_color},{self.back_color},"
            f"{-1 if self.bold else 0},{-1 if self.italic else 0},0,0,100,100,0,0,"
            f"{self.border_style},"
            f"{self.outline:g},{self.shadow:g},{self.alignment},"
            f"{self.margin_horizontal},{self.margin_horizontal},{self.margin_bottom},1"
        )


@dataclass
class HookStyle:
    """Gaya teks hook di bagian atas video."""

    font_name: str = "Montserrat"
    font_size: int = 78
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&HB0000000"
    outline: float = 5
    shadow: float = 0
    bold: bool = True
    uppercase: bool = True
    max_chars_per_line: int = 18
    margin_top: int = 240
    margin_horizontal: int = 80
    alignment: int = 8
    border_style: int = 3  # kotak penuh: hook lebih terbaca di atas video apa pun
    duration: float = 3.0
    fade_in: float = 0.25
    fade_out: float = 0.35
    enabled: bool = True

    @classmethod
    def from_config(cls, cfg: Config) -> "HookStyle":
        return cls(
            font_name=str(cfg.get("hook.font_name", "Montserrat")),
            font_size=int(cfg.get("hook.font_size", 78)),
            primary_color=to_ass_color(cfg.get("hook.primary_color", "&H00FFFFFF")),
            outline_color=to_ass_color(cfg.get("hook.outline_color", "&H00000000"), "&H00000000"),
            back_color=to_ass_color(cfg.get("hook.back_color", "&HB0000000"), "&HB0000000"),
            outline=float(cfg.get("hook.outline", 5)),
            shadow=float(cfg.get("hook.shadow", 0)),
            bold=bool(cfg.get("hook.bold", True)),
            uppercase=bool(cfg.get("hook.uppercase", True)),
            max_chars_per_line=int(cfg.get("hook.max_chars_per_line", 18)),
            margin_top=int(cfg.get("hook.margin_top", 240)),
            margin_horizontal=int(cfg.get("hook.margin_horizontal", 80)),
            alignment=int(cfg.get("hook.alignment", 8)),
            duration=float(cfg.get("hook.duration", 3.0)),
            fade_in=float(cfg.get("hook.fade_in", 0.25)),
            fade_out=float(cfg.get("hook.fade_out", 0.35)),
            enabled=bool(cfg.get("hook.enabled", True)),
            border_style=int(cfg.get("hook.border_style", 3)),
        )

    def scaled(self, factor: float) -> "HookStyle":
        """Salinan gaya dengan ukuran px diskalakan (lihat REFERENCE_HEIGHT)."""

        if abs(factor - 1.0) < 1e-6:
            return self
        return replace(
            self,
            font_size=max(8, int(round(self.font_size * factor))),
            outline=round(self.outline * factor, 2),
            shadow=round(self.shadow * factor, 2),
            margin_top=max(0, int(round(self.margin_top * factor))),
            margin_horizontal=max(0, int(round(self.margin_horizontal * factor))),
        )

    def style_line(self, name: str = "Hook") -> str:
        return (
            f"Style: {name},{self.font_name},{self.font_size},{self.primary_color},"
            f"{self.primary_color},{self.outline_color},{self.back_color},"
            f"{-1 if self.bold else 0},0,0,0,100,100,0,0,"
            f"{self.border_style},"
            f"{self.outline:g},{self.shadow:g},{self.alignment},"
            f"{self.margin_horizontal},{self.margin_horizontal},{self.margin_top},1"
        )


# --------------------------------------------------------------------------
# Pengelompokan kata
# --------------------------------------------------------------------------
def relativize_words(words: Sequence[Word], offset: float) -> List[Word]:
    """Geser waktu kata agar relatif terhadap awal klip, buang yang negatif."""

    out: List[Word] = []
    for word in words:
        shifted = Word(
            start=max(0.0, word.start - offset),
            end=max(0.0, word.end - offset),
            text=word.text,
            probability=word.probability,
        )
        if shifted.end > 0:
            out.append(shifted)
    return out


def group_words_into_lines(
    words: Sequence[Word],
    *,
    max_words: int = 4,
    max_chars: int = 22,
    max_gap: float = LINE_BREAK_GAP,
) -> List[List[Word]]:
    """Pecah aliran kata menjadi baris-baris subtitle.

    Baris baru dimulai kalau: jumlah kata/karakter sudah penuh, ada jeda bicara
    yang cukup panjang, atau kata sebelumnya mengakhiri kalimat.
    """

    lines: List[List[Word]] = []
    current: List[Word] = []
    length = 0

    for word in words:
        text = word.text.strip()
        if not text:
            continue
        gap = (word.start - current[-1].end) if current else 0.0
        sentence_ended = bool(current) and current[-1].text.strip().endswith((".", "!", "?", "…"))
        too_many = len(current) >= max(1, max_words)
        too_long = current and (length + 1 + len(text)) > max(6, max_chars)

        if current and (too_many or too_long or gap > max_gap or sentence_ended):
            lines.append(current)
            current, length = [], 0

        current.append(word)
        length += len(text) + (1 if length else 0)

    if current:
        lines.append(current)
    return lines


# --------------------------------------------------------------------------
# Penyusun file ASS
# --------------------------------------------------------------------------
def _header(width: int, height: int, styles: Iterable[str]) -> List[str]:
    return [
        "[Script Info]",
        "; Dibuat otomatis oleh Tukang Sayur",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        *styles,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]


def _dialogue(start: float, end: float, style: str, text: str, layer: int = 0) -> str:
    return (
        f"Dialogue: {layer},{format_ass_time(start)},{format_ass_time(end)},"
        f"{style},,0,0,0,,{text}"
    )


def build_word_events(
    lines: Sequence[Sequence[Word]],
    style: SubtitleStyle,
    *,
    clip_duration: float = 0.0,
    style_name: str = "Sub",
) -> List[str]:
    """Bikin event Dialogue. Mode karaoke = satu event per kata."""

    events: List[str] = []
    for line_index, line in enumerate(lines):
        if not line:
            continue
        words = [w for w in line if w.text.strip()]
        if not words:
            continue
        line_start = words[0].start
        line_end = max(words[-1].end, line_start + 0.2)
        if clip_duration:
            line_end = min(line_end, clip_duration)
        next_line_start = None
        if line_index + 1 < len(lines) and lines[line_index + 1]:
            next_line_start = lines[line_index + 1][0].start
        # Tahan baris sedikit lebih lama kalau baris berikutnya masih jauh.
        hold = min(0.4, max(0.0, (next_line_start - line_end))) if next_line_start else 0.25
        line_end = line_end + hold
        if clip_duration:
            line_end = min(line_end, clip_duration)

        rendered = [
            escape_ass_text(w.text.upper() if style.uppercase else w.text) for w in words
        ]

        if not style.karaoke:
            events.append(
                _dialogue(line_start, line_end, style_name, " ".join(rendered))
            )
            continue

        for position, word in enumerate(words):
            segment_start = word.start if position else line_start
            if position + 1 < len(words):
                segment_end = max(words[position + 1].start, segment_start + 0.03)
            else:
                segment_end = max(line_end, segment_start + 0.05)
            if clip_duration:
                segment_end = min(segment_end, clip_duration)
            if segment_end <= segment_start:
                continue

            highlight = inline_color(style.highlight_color)
            normal = inline_color(style.primary_color)
            pieces: List[str] = []
            for other, text in enumerate(rendered):
                if other == position:
                    pieces.append(
                        f"{{\\c{highlight}\\fscx{style.highlight_scale}"
                        f"\\fscy{style.highlight_scale}}}{text}"
                        f"{{\\c{normal}\\fscx100\\fscy100}}"
                    )
                else:
                    pieces.append(text)
            events.append(_dialogue(segment_start, segment_end, style_name, " ".join(pieces)))
    return events


def build_hook_events(
    hook_text: str,
    hook_style: HookStyle,
    *,
    clip_duration: float = 0.0,
    style_name: str = "Hook",
) -> List[str]:
    """Event hook: tampil sejak detik 0 di atas video yang tetap berjalan."""

    text = str(hook_text or "").strip()
    if not text or not hook_style.enabled or hook_style.duration <= 0:
        return []

    duration = hook_style.duration
    if clip_duration:
        duration = min(duration, max(0.5, clip_duration - 0.1))

    lines = wrap_text(
        text.upper() if hook_style.uppercase else text, hook_style.max_chars_per_line
    )
    body = "\\N".join(escape_ass_text(line) for line in lines)
    fade = ""
    if hook_style.fade_in > 0 or hook_style.fade_out > 0:
        fade = (
            f"{{\\fad({int(max(0.0, hook_style.fade_in) * 1000)},"
            f"{int(max(0.0, hook_style.fade_out) * 1000)})}}"
        )
    return [_dialogue(0.0, duration, style_name, f"{fade}{body}", layer=1)]


def build_ass(
    words: Sequence[Word],
    *,
    style: SubtitleStyle,
    hook_style: Optional[HookStyle] = None,
    hook_text: str = "",
    width: int = 1080,
    height: int = 1920,
    clip_duration: float = 0.0,
    subtitles_enabled: bool = True,
) -> str:
    """Rangkai seluruh isi file ``.ass``."""

    factor = (float(height) / REFERENCE_HEIGHT) if height else 1.0
    style = style.scaled(factor)
    if hook_style is not None:
        hook_style = hook_style.scaled(factor)

    styles = [style.style_line("Sub")]
    if hook_style is not None:
        styles.append(hook_style.style_line("Hook"))

    events: List[str] = []
    if hook_style is not None and hook_text:
        events.extend(build_hook_events(hook_text, hook_style, clip_duration=clip_duration))
    if subtitles_enabled and words:
        lines = group_words_into_lines(
            words,
            max_words=style.max_words_per_line,
            max_chars=style.max_chars_per_line,
        )
        events.extend(build_word_events(lines, style, clip_duration=clip_duration))

    return "\n".join([*_header(width, height, styles), *events, ""])


def write_ass(
    path: str | Path,
    words: Sequence[Word],
    **kwargs,
) -> Path:
    """Tulis file ASS ke disk (dipakai tahap render)."""

    content = build_ass(words, **kwargs)
    target = atomic_write_text(path, content)
    log.debug("File subtitle ASS ditulis: %s (%d baris)", target, content.count("\n"))
    return target
