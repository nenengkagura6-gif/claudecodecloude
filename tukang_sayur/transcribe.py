"""Tahap 2 - Transkripsi dengan faster-whisper (timestamp per kata) + cache.

Default: model ``small``, ``compute_type=int8``, ``device=cpu``. Kombinasi ini
sengaja dipilih supaya jalan di PC tanpa GPU dengan RAM 12 GB (pemakaian RAM
kira-kira 1-2 GB saat transkripsi).

Bahasa: ``id`` (Indonesia) dan ``su`` (Sunda) didukung Whisper. Untuk Sunda,
akurasi model kecil biasanya turun, jadi kode ini memberi peringatan dan
menyarankan model yang lebih besar bila hasilnya kacau.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .config import Config
from .errors import TranscriptionError
from .logging_utils import get_logger
from .models import Segment, Transcript, Word
from .utils import file_fingerprint, human_duration, probe_media, read_json, sha1_text, write_json

log = get_logger("transcribe")

#: Bahasa yang kita janjikan didukung penuh oleh CLI ini.
PRIMARY_LANGUAGES = {"id", "su"}

#: Model yang aman untuk CPU 12 GB RAM.
CPU_FRIENDLY_MODELS = {"tiny", "base", "small", "distil-small.en", "medium"}

_SENTENCE_END = ".!?…"


class Transcriber(Protocol):
    """Antarmuka minimal supaya pipeline bisa disuntik transcriber palsu."""

    name: str

    def transcribe(self, media_path: Path) -> Transcript: ...


# --------------------------------------------------------------------------
# Pembagi waktu kata (fallback)
# --------------------------------------------------------------------------
def distribute_words(text: str, start: float, end: float) -> List[Word]:
    """Bagi rata waktu satu segmen ke kata-katanya, berbobot panjang kata.

    Dipakai kalau model tidak mengembalikan timestamp per kata. Tanpa ini,
    subtitle karaoke tidak bisa dibuat.
    """

    words = str(text or "").split()
    if not words:
        return []
    start = max(0.0, float(start))
    end = max(start, float(end))
    span = end - start
    if span <= 0:
        return [Word(start=start, end=start, text=w, probability=0.0) for w in words]

    weights = [max(1, len(w)) for w in words]
    total = float(sum(weights))
    result: List[Word] = []
    cursor = start
    for word, weight in zip(words, weights):
        share = span * (weight / total)
        word_end = min(end, cursor + share)
        result.append(Word(start=cursor, end=word_end, text=word, probability=0.0))
        cursor = word_end
    if result:
        result[-1].end = end
    return result


def ensure_word_timings(segments: Sequence[Segment]) -> List[Segment]:
    """Pastikan setiap segmen punya kata bertimestamp."""

    out: List[Segment] = []
    for segment in segments:
        if segment.words:
            out.append(segment)
            continue
        out.append(
            Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=distribute_words(segment.text, segment.start, segment.end),
            )
        )
    return out


# --------------------------------------------------------------------------
# faster-whisper
# --------------------------------------------------------------------------
class FasterWhisperTranscriber:
    """Pembungkus ``faster_whisper.WhisperModel``."""

    name = "faster-whisper"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model_size = str(cfg.get("transcription.model", "small"))
        self.device = str(cfg.get("transcription.device", "cpu"))
        self.compute_type = str(cfg.get("transcription.compute_type", "int8"))
        self.cpu_threads = int(cfg.get("transcription.cpu_threads", 4) or 0)
        self.beam_size = int(cfg.get("transcription.beam_size", 5) or 1)
        self.language = str(cfg.get("transcription.language", "id") or "auto").lower()
        self.vad_filter = bool(cfg.get("transcription.vad_filter", True))
        self.condition_on_previous_text = bool(
            cfg.get("transcription.condition_on_previous_text", False)
        )
        self._model: Any = None
        self._warn_language()

    def _warn_language(self) -> None:
        if self.language in {"auto", ""}:
            return
        if self.language not in PRIMARY_LANGUAGES:
            log.info(
                "Bahasa '%s' di luar fokus utama (id/su); Whisper tetap mencoba.",
                self.language,
            )
        if self.language == "su" and self.model_size in {"tiny", "base"}:
            log.warning(
                "Bahasa Sunda dengan model '%s' sering kurang akurat. "
                "Kalau hasilnya kacau, coba transcription.model: small atau medium.",
                self.model_size,
            )

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TranscriptionError(
                "Paket faster-whisper belum terpasang. Install: "
                "pip install faster-whisper"
            ) from exc

        if self.model_size not in CPU_FRIENDLY_MODELS and self.device == "cpu":
            log.warning(
                "Model '%s' berat untuk CPU. Untuk RAM 12 GB, 'small' paling pas.",
                self.model_size,
            )
        log.info(
            "Memuat model Whisper '%s' (device=%s, compute_type=%s)...",
            self.model_size, self.device, self.compute_type,
        )
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads or 0,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Gagal memuat model Whisper '{self.model_size}': {exc}\n"
                "Model diunduh otomatis saat pertama kali dipakai, jadi pastikan "
                "ada koneksi internet dan ruang disk (~500 MB untuk 'small')."
            ) from exc
        return self._model

    def transcribe(self, media_path: Path) -> Transcript:
        model = self._load()
        language = None if self.language in {"auto", ""} else self.language
        started = time.time()
        log.info("Transkripsi %s ...", Path(media_path).name)
        try:
            raw_segments, info = model.transcribe(
                str(media_path),
                language=language,
                beam_size=self.beam_size,
                word_timestamps=True,
                vad_filter=self.vad_filter,
                condition_on_previous_text=self.condition_on_previous_text,
            )
            segments: List[Segment] = []
            for raw in raw_segments:  # generator: pekerjaan nyata terjadi di sini
                words = [
                    Word(
                        start=float(getattr(w, "start", raw.start) or 0.0),
                        end=float(getattr(w, "end", raw.end) or 0.0),
                        text=str(getattr(w, "word", "")).strip(),
                        probability=float(getattr(w, "probability", 1.0) or 0.0),
                    )
                    for w in (getattr(raw, "words", None) or [])
                    if str(getattr(w, "word", "")).strip()
                ]
                segment = Segment(
                    start=float(raw.start or 0.0),
                    end=float(raw.end or 0.0),
                    text=str(raw.text or "").strip(),
                    words=words,
                )
                if segment.text:
                    segments.append(segment)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Transkripsi gagal: {exc}") from exc

        segments = ensure_word_timings(segments)
        detected = str(getattr(info, "language", language or "") or "")
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if duration <= 0:
            duration = segments[-1].end if segments else 0.0

        elapsed = time.time() - started
        log.info(
            "Transkripsi selesai: %d segmen, %d kata, bahasa=%s, %s (butuh %s)",
            len(segments),
            sum(len(s.words) for s in segments),
            detected,
            human_duration(duration),
            human_duration(elapsed),
        )
        if not segments:
            raise TranscriptionError(
                "Tidak ada ucapan yang terdeteksi. Kalau videonya memang ada suara, "
                "coba matikan transcription.vad_filter atau ganti bahasa ke 'auto'."
            )
        return Transcript(
            language=detected or (language or ""),
            duration=duration,
            segments=segments,
            model=self.model_size,
            source=str(media_path),
        )


# --------------------------------------------------------------------------
# Transcriber tiruan (test / offline)
# --------------------------------------------------------------------------
class DummyTranscriber:
    """Transcriber tanpa model AI.

    Urutan sumber teks:

    1. file pendamping ``<video>.words.json`` (format Transcript.to_dict()),
    2. file pendamping ``<video>.txt`` (teks biasa, waktunya dibagi rata),
    3. teks yang disuntikkan lewat parameter ``script``.

    Berguna untuk test end-to-end dan untuk mencoba pipeline render tanpa
    menunggu Whisper.
    """

    name = "dummy"

    def __init__(
        self,
        cfg: Optional[Config] = None,
        *,
        script: Optional[str] = None,
        language: str = "id",
    ) -> None:
        self.cfg = cfg
        self.script = script
        self.language = language

    def transcribe(self, media_path: Path) -> Transcript:
        media_path = Path(media_path)
        duration = 0.0
        try:
            duration = probe_media(media_path).duration
        except Exception as exc:  # ffprobe tidak ada / bukan media -> pakai dugaan
            log.debug("DummyTranscriber tidak bisa ffprobe %s: %s", media_path, exc)

        sidecar_json = media_path.with_suffix(media_path.suffix + ".words.json")
        if sidecar_json.is_file():
            data = read_json(sidecar_json)
            transcript = Transcript.from_dict(data)
            transcript.source = str(media_path)
            transcript.model = self.name
            return transcript

        sidecar_txt = media_path.with_suffix(".txt")
        text = self.script
        if text is None and sidecar_txt.is_file():
            text = sidecar_txt.read_text(encoding="utf-8")
        if not text:
            text = (
                "Ini transkrip contoh dari transcriber tiruan. "
                "Dipakai untuk menguji pipeline tanpa model Whisper."
            )

        duration = duration or 30.0
        sentences = _split_sentences(text)
        segments: List[Segment] = []
        cursor = 0.0
        per_sentence = duration / max(1, len(sentences))
        for sentence in sentences:
            end = min(duration, cursor + per_sentence)
            segments.append(
                Segment(
                    start=cursor,
                    end=end,
                    text=sentence,
                    words=distribute_words(sentence, cursor, end),
                )
            )
            cursor = end
        return Transcript(
            language=self.language,
            duration=duration,
            segments=segments,
            model=self.name,
            source=str(media_path),
        )


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    buffer: List[str] = []
    for token in str(text).split():
        buffer.append(token)
        if token[-1] in _SENTENCE_END:
            sentences.append(" ".join(buffer))
            buffer = []
    if buffer:
        sentences.append(" ".join(buffer))
    return sentences or [str(text).strip()]


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def cache_key(media_path: Path, cfg: Config, engine: str) -> str:
    """Kunci cache: sidik jari file + parameter yang mempengaruhi hasil."""

    try:
        fingerprint = file_fingerprint(media_path)
    except OSError:
        fingerprint = str(media_path)
    return sha1_text(
        fingerprint,
        engine,
        cfg.get("transcription.model"),
        cfg.get("transcription.language"),
        cfg.get("transcription.compute_type"),
        cfg.get("transcription.beam_size"),
        cfg.get("transcription.vad_filter"),
    )


def cache_path(media_path: Path, cfg: Config, engine: str) -> Path:
    return cfg.cache_dir / "transcripts" / f"{cache_key(media_path, cfg, engine)}.json"


def load_cached(path: Path) -> Optional[Transcript]:
    if not path.is_file():
        return None
    try:
        transcript = Transcript.from_dict(read_json(path))
    except Exception as exc:
        log.warning("Cache transkrip rusak, diabaikan (%s): %s", path.name, exc)
        return None
    if not transcript.segments:
        return None
    return transcript


def build_transcriber(cfg: Config) -> Transcriber:
    engine = str(cfg.get("transcription.engine", "faster-whisper")).lower()
    if engine in {"faster-whisper", "fasterwhisper", "whisper"}:
        return FasterWhisperTranscriber(cfg)
    if engine in {"dummy", "mock", "fake"}:
        log.warning("transcription.engine='%s': memakai transkrip tiruan.", engine)
        return DummyTranscriber(cfg)
    raise TranscriptionError(
        f"transcription.engine '{engine}' tidak dikenal. Pilihan: faster-whisper, dummy"
    )


def transcribe_media(
    media_path: str | Path,
    cfg: Config,
    *,
    transcriber: Optional[Transcriber] = None,
    use_cache: Optional[bool] = None,
) -> Transcript:
    """Transkripsi dengan cache. Cache hit = pipeline bisa diulang instan."""

    media_path = Path(media_path)
    transcriber = transcriber or build_transcriber(cfg)
    engine = getattr(transcriber, "name", transcriber.__class__.__name__)
    caching = cfg.get("transcription.cache", True) if use_cache is None else use_cache

    target = cache_path(media_path, cfg, engine) if caching else None
    if target is not None:
        cached = load_cached(target)
        if cached is not None:
            log.info(
                "Memakai cache transkrip (%d segmen) dari %s", len(cached.segments), target.name
            )
            return cached

    transcript = transcriber.transcribe(media_path)
    if target is not None:
        try:
            write_json(target, transcript.to_dict())
            log.debug("Transkrip disimpan ke cache: %s", target)
        except OSError as exc:
            log.warning("Gagal menyimpan cache transkrip: %s", exc)
    return transcript


def transcript_stats(transcript: Transcript) -> Dict[str, Any]:
    words = transcript.words
    return {
        "segments": len(transcript.segments),
        "words": len(words),
        "duration": round(transcript.duration, 2),
        "words_per_minute": round(len(words) / (transcript.duration / 60), 1)
        if transcript.duration > 0
        else 0.0,
        "language": transcript.language,
    }
