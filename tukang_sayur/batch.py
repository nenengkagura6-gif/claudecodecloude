"""Mode batch: proses banyak video dari satu file daftar, bisa dilanjutkan.

State disimpan sebagai JSON. Kalau proses terhenti (mati lampu, Ctrl+C, atau
error), menjalankan perintah yang sama akan melanjutkan dari sumber yang belum
selesai - yang sudah ``done`` dilewati.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .errors import TukangSayurError
from .llm.base import LLMProvider
from .logging_utils import get_logger
from .models import JobResult
from .pipeline import ClipJobOptions, ProgressFn, run_clip_job
from .utils import read_json, sha1_text, write_json

log = get_logger("batch")

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def read_sources(path: str | Path) -> List[str]:
    """Baca daftar sumber: satu per baris, ``#`` untuk komentar."""

    target = Path(path)
    if not target.is_file():
        raise TukangSayurError(f"File daftar tidak ditemukan: {target}")
    sources: List[str] = []
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sources.append(line)
    if not sources:
        raise TukangSayurError(f"File daftar kosong: {target}")
    return sources


@dataclass
class BatchState:
    """Isi file state batch."""

    path: Path
    items: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "BatchState":
        target = Path(path)
        if not target.is_file():
            return cls(path=target)
        try:
            data = read_json(target)
        except Exception as exc:
            log.warning("State batch rusak (%s), mulai dari awal: %s", target, exc)
            return cls(path=target)
        items = data.get("items") if isinstance(data, dict) else None
        return cls(path=target, items=dict(items) if isinstance(items, dict) else {})

    @staticmethod
    def key(source: str) -> str:
        return sha1_text(source.strip())[:16]

    def status(self, source: str) -> str:
        entry = self.items.get(self.key(source)) or {}
        return str(entry.get("status", STATUS_PENDING))

    def is_done(self, source: str) -> bool:
        return self.status(source) == STATUS_DONE

    def update(self, source: str, **fields: Any) -> None:
        key = self.key(source)
        entry = self.items.setdefault(key, {"source": source})
        entry.update(fields)
        entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def save(self) -> None:
        try:
            write_json(self.path, {"items": self.items})
        except OSError as exc:  # pragma: no cover
            log.warning("Gagal menyimpan state batch: %s", exc)

    def summary(self) -> Dict[str, int]:
        counts = {STATUS_DONE: 0, STATUS_FAILED: 0, STATUS_PENDING: 0}
        for entry in self.items.values():
            status = str(entry.get("status", STATUS_PENDING))
            counts[status] = counts.get(status, 0) + 1
        return counts


def state_path_for(cfg: Config) -> Path:
    """Lokasi file state: relatif terhadap folder output kalau bukan absolut."""

    raw = str(cfg.get("batch.state_file", "batch_state.json"))
    path = Path(raw)
    return path if path.is_absolute() else cfg.output_dir / path


@dataclass
class BatchResult:
    total: int = 0
    done: int = 0
    skipped: int = 0
    failed: int = 0
    jobs: List[JobResult] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "done": self.done,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": dict(self.errors),
            "clip_count": sum(len(job.clips) for job in self.jobs),
        }


def run_batch(
    list_path: str | Path,
    cfg: Config,
    *,
    options: Optional[ClipJobOptions] = None,
    resume: bool = True,
    provider: Optional[LLMProvider] = None,
    progress: Optional[ProgressFn] = None,
) -> BatchResult:
    """Proses semua sumber di ``list_path``."""

    sources = read_sources(list_path)
    options = options or ClipJobOptions()
    cfg.ensure_dirs()
    state = BatchState.load(state_path_for(cfg))
    continue_on_error = bool(cfg.get("batch.continue_on_error", True))

    result = BatchResult(total=len(sources))
    log.info("Batch: %d sumber, state di %s", len(sources), state.path)

    for position, source in enumerate(sources, start=1):
        if resume and state.is_done(source):
            log.info("[%d/%d] Dilewati (sudah selesai): %s", position, len(sources), source)
            result.skipped += 1
            continue

        log.info("[%d/%d] Memproses: %s", position, len(sources), source)
        state.update(source, status=STATUS_PENDING, error="")
        try:
            job = run_clip_job(
                source, cfg, options=options, provider=provider, progress=progress
            )
        except TukangSayurError as exc:
            message = str(exc)
            log.error("[%d/%d] Gagal: %s", position, len(sources), message)
            state.update(source, status=STATUS_FAILED, error=message)
            result.failed += 1
            result.errors[source] = message
            if not continue_on_error:
                break
            continue
        except KeyboardInterrupt:
            state.update(source, status=STATUS_PENDING, error="dihentikan pengguna")
            log.warning("Dihentikan pengguna. Jalankan lagi perintah yang sama "
                        "untuk melanjutkan dari sini.")
            raise

        state.update(
            source,
            status=STATUS_DONE,
            output_dir=job.output_dir,
            clips=len(job.clips),
            best_score=job.best_score,
            error="",
        )
        result.done += 1
        result.jobs.append(job)

    log.info(
        "Batch selesai: %d sukses, %d dilewati, %d gagal (total %d klip)",
        result.done, result.skipped, result.failed,
        sum(len(job.clips) for job in result.jobs),
    )
    return result
