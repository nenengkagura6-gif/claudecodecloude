"""Semua prompt LLM dikumpulkan di sini supaya mudah disetel ulang.

Format baris transkrip yang dipakai di semua prompt::

    [12.00 -> 18.50] teks kalimatnya

Angkanya detik dari awal video. Format ini juga yang dibaca
:class:`~tukang_sayur.llm.providers.MockProvider` saat test.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .models import Candidate, Segment

#: Batas aman jumlah karakter transkrip per permintaan LLM.
DEFAULT_CHUNK_CHARS = 14000

SYSTEM_CANDIDATE = (
    "Kamu editor video pendek berpengalaman untuk TikTok, Reels, dan Shorts. "
    "Kamu jago mencari bagian video yang bikin orang berhenti scroll. "
    "Kamu menjawab HANYA dengan JSON valid, tanpa penjelasan tambahan."
)

SYSTEM_JUDGE = (
    "Kamu juri konten video pendek yang tegas dan konsisten. "
    "Kamu menilai potongan video secara objektif dan menjawab HANYA JSON valid."
)

SYSTEM_METADATA = (
    "Kamu penulis caption media sosial berbahasa Indonesia yang natural, "
    "tidak lebay, dan tahu cara memancing komentar. "
    "Kamu menjawab HANYA dengan JSON valid."
)


def format_transcript_lines(segments: Sequence[Segment]) -> List[str]:
    """Ubah segmen jadi baris ``[mulai -> selesai] teks``."""

    lines: List[str] = []
    for segment in segments:
        text = " ".join(str(segment.text).split())
        if not text:
            continue
        lines.append(f"[{segment.start:.2f} -> {segment.end:.2f}] {text}")
    return lines


def chunk_segments(
    segments: Sequence[Segment], max_chars: int = DEFAULT_CHUNK_CHARS
) -> List[List[Segment]]:
    """Pecah transkrip panjang jadi beberapa bagian agar muat di satu prompt."""

    chunks: List[List[Segment]] = []
    current: List[Segment] = []
    size = 0
    for segment in segments:
        length = len(segment.text) + 24  # + perkiraan panjang label waktu
        if current and size + length > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(segment)
        size += length
    if current:
        chunks.append(current)
    return chunks or [[]]


def candidate_prompt(
    segments: Sequence[Segment],
    *,
    wanted: int,
    min_duration: float,
    max_duration: float,
    language: str = "id",
    video_duration: float = 0.0,
    max_hook_words: int = 8,
    part: int = 1,
    total_parts: int = 1,
) -> str:
    """Prompt pencari kandidat klip."""

    lines = "\n".join(format_transcript_lines(segments))
    part_note = (
        f"\nCatatan: ini bagian {part} dari {total_parts} potongan transkrip. "
        "Nilai waktunya sudah absolut terhadap video penuh, jangan diubah."
        if total_parts > 1
        else ""
    )
    return f"""TASK: find_candidates

Kamu diberi transkrip sebuah video berbahasa {language} dengan timestamp.
Durasi video: {video_duration:.1f} detik.{part_note}

Tugasmu: pilih {wanted} kandidat potongan paling menarik untuk dijadikan klip
vertikal (TikTok/Reels/Shorts), masing-masing berdurasi {min_duration:.0f}-{max_duration:.0f} detik.

Aturan wajib:
1. start dan end HARUS berupa angka detik yang ada di rentang transkrip di bawah.
2. Potongan harus dimulai di awal kalimat dan berakhir di akhir kalimat.
   Jangan memotong di tengah kalimat.
3. Setiap kandidat harus bisa dimengerti tanpa menonton bagian lain video.
4. Prioritaskan: pernyataan mengejutkan, cerita personal, tips konkret, angka
   atau fakta spesifik, konflik/opini tajam, momen lucu, dan kalimat yang
   memancing rasa penasaran.
5. Hindari bagian yang isinya cuma sapaan, basa-basi, atau promosi.
6. hook_text maksimal {max_hook_words} kata, bahasa {language}, ditulis seperti
   teks besar di detik pertama video. Tanpa tanda kutip, tanpa emoji.
7. Jangan ada kandidat yang saling tumpang tindih.

Jawab HANYA dengan JSON berbentuk:
{{
  "candidates": [
    {{
      "start": 123.4,
      "end": 178.9,
      "hook_text": "Kalimat hook singkat",
      "reason": "kenapa bagian ini menarik (1 kalimat)"
    }}
  ]
}}

TRANSKRIP:
{lines}
"""


def judge_prompt(
    candidates: Sequence[Candidate],
    *,
    language: str = "id",
    max_transcript_chars: int = 900,
) -> str:
    """Prompt penilaian viral (0-100 per dimensi)."""

    blocks: List[str] = []
    for position, candidate in enumerate(candidates, start=1):
        text = " ".join(str(candidate.transcript_text).split())
        if len(text) > max_transcript_chars:
            text = text[:max_transcript_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(
            f"#{position} | durasi {candidate.duration:.1f}s | "
            f"sumber {candidate.start:.2f}-{candidate.end:.2f}\n"
            f"HOOK: {candidate.hook_text or '(belum ada)'}\n"
            f"ISI: {text or '(transkrip tidak tersedia)'}"
        )
    body = "\n\n".join(blocks)

    return f"""TASK: score_candidates

Nilai setiap kandidat klip di bawah untuk media sosial vertikal berbahasa {language}.
Beri angka 0-100 pada empat dimensi:

- hook: seberapa kuat 3 detik pertama menahan orang supaya tidak scroll.
- emotion: seberapa kuat dorongan emosinya (kaget, lucu, haru, marah, penasaran).
- clarity: seberapa jelas dan enak diikuti kalimatnya.
- standalone: seberapa utuh isinya tanpa perlu konteks video penuh.

Pakai seluruh rentang nilai. Klip biasa-biasa saja layak 40-60; hanya klip yang
benar-benar kuat boleh di atas 85. Jangan memberi nilai sama untuk semua.

Jawab HANYA dengan JSON berbentuk:
{{
  "scores": [
    {{
      "index": 1,
      "hook": 82,
      "emotion": 74,
      "clarity": 88,
      "standalone": 69,
      "reason": "alasan singkat"
    }}
  ]
}}

Nomor "index" harus sama dengan nomor kandidat (#1, #2, ...).

KANDIDAT:
{body}
"""


def metadata_prompt(
    candidates: Sequence[Candidate],
    *,
    language: str = "id",
    hashtag_count: int = 8,
) -> str:
    """Prompt pembuat judul, caption, dan hashtag."""

    blocks: List[str] = []
    for position, candidate in enumerate(candidates, start=1):
        text = " ".join(str(candidate.transcript_text).split())[:600]
        blocks.append(
            f"#{position}\nHOOK: {candidate.hook_text}\nISI: {text or '(tidak ada)'}"
        )
    body = "\n\n".join(blocks)

    return f"""TASK: write_metadata

Untuk setiap klip di bawah, tulis metadata unggahan dalam bahasa {language}:

- title: maksimal 60 karakter, bikin penasaran, tanpa clickbait bohong.
- caption: 1-2 kalimat, diakhiri satu pertanyaan pemancing komentar.
  Tanpa hashtag di dalam caption.
- hashtags: {hashtag_count} hashtag relevan, tanpa tanda "#", huruf kecil,
  campuran umum (fyp, viral) dan spesifik sesuai topik klip.

Jawab HANYA dengan JSON berbentuk:
{{
  "items": [
    {{
      "index": 1,
      "title": "Judul singkat",
      "caption": "Caption dua kalimat. Pertanyaan pemancing komentar?",
      "hashtags": ["fyp", "contoh"]
    }}
  ]
}}

KLIP:
{body}
"""


def join_lines(lines: Iterable[str]) -> str:
    return "\n".join(lines)
