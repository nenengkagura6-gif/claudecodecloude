"""Endpoint HTTP opsional supaya Tukang Sayur bisa dipanggil dari n8n/Zapier.

Jalankan::

    python -m tukang_sayur serve --port 8000

Contoh panggilan dari n8n (node HTTP Request)::

    POST http://localhost:8000/clip
    Authorization: Bearer <isi TUKANG_SAYUR_TOKEN di .env>
    {"source": "https://youtu.be/xxxx", "max_clips": 5}

Balasannya langsung memuat ``job_id``. Status dan daftar klip dicek di
``GET /jobs/<job_id>``. Kalau ingin menunggu sampai selesai dalam satu
permintaan, kirim ``"wait": true`` (hati-hati timeout di sisi pemanggil).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .config import Config
from .errors import TukangSayurError
from .logging_utils import get_logger

log = get_logger("server")

#: Riwayat job dalam memori (cukup untuk pemakaian satu mesin).
_JOBS: Dict[str, "JobRecord"] = {}
_LOCK = threading.Lock()


@dataclass
class JobRecord:
    job_id: str
    source: str
    status: str = "antre"  # antre | jalan | selesai | gagal
    stage: str = ""
    message: str = ""
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["duration_seconds"] = round(
            (self.finished_at or time.time()) - self.created_at, 1
        )
        return data


def _import_fastapi():
    try:
        from fastapi import FastAPI, Header, HTTPException  # type: ignore
        from pydantic import BaseModel  # type: ignore

        return FastAPI, Header, HTTPException, BaseModel
    except ImportError as exc:  # pragma: no cover
        raise TukangSayurError(
            "FastAPI belum terpasang. Install dulu: pip install fastapi uvicorn"
        ) from exc


def _run_job(record: JobRecord, cfg: Config, payload: Dict[str, Any]) -> None:
    from .pipeline import ClipJobOptions, run_clip_job

    def report(stage: str, message: str, fraction: float) -> None:
        record.stage, record.message, record.progress = stage, message, round(fraction, 3)

    record.status = "jalan"
    try:
        overrides = {
            key: payload[key]
            for key in ("llm.provider", "transcription.model", "transcription.language")
            if key in payload
        }
        if payload.get("provider"):
            overrides["llm.provider"] = payload["provider"]
        job_cfg = Config.load(cfg.path, overrides=overrides) if overrides else cfg

        job = run_clip_job(
            record.source,
            job_cfg,
            options=ClipJobOptions(
                max_clips=payload.get("max_clips"),
                dry_run=bool(payload.get("dry_run", False)),
            ),
            progress=report,
        )
        record.result = job.to_dict()
        record.status = "selesai"
        record.progress = 1.0
    except TukangSayurError as exc:
        record.status, record.error = "gagal", str(exc)
        log.error("Job %s gagal: %s", record.job_id, exc)
    except Exception as exc:  # pragma: no cover - jaring pengaman
        record.status, record.error = "gagal", f"Error tak terduga: {exc}"
        log.exception("Job %s error tak terduga", record.job_id)
    finally:
        record.finished_at = time.time()


def create_app(cfg: Config):
    """Bangun aplikasi FastAPI."""

    FastAPI, Header, HTTPException, BaseModel = _import_fastapi()

    class ClipRequest(BaseModel):  # type: ignore[misc]
        source: str
        max_clips: Optional[int] = None
        provider: Optional[str] = None
        dry_run: bool = False
        wait: bool = False

    app = FastAPI(
        title="Tukang Sayur",
        version="0.1.0",
        description="Pemotong video otomatis jadi klip vertikal 9:16.",
    )

    token_env = str(cfg.get("server.api_token_env", "TUKANG_SAYUR_TOKEN"))

    def check_token(authorization: Optional[str]) -> None:
        expected = os.environ.get(token_env, "").strip()
        if not expected:
            return  # token tidak diset = endpoint terbuka (pakai di jaringan lokal saja)
        given = (authorization or "").removeprefix("Bearer ").strip()
        if given != expected:
            raise HTTPException(status_code=401, detail="Token salah atau tidak ada.")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        from .utils import has_ffmpeg

        return {
            "status": "ok",
            "ffmpeg": has_ffmpeg(),
            "provider": cfg.provider_name(),
            "jobs": len(_JOBS),
        }

    @app.post("/clip")
    def clip(request: ClipRequest, authorization: Optional[str] = Header(default=None)):
        check_token(authorization)
        if not request.source.strip():
            raise HTTPException(status_code=400, detail="Field 'source' wajib diisi.")

        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(job_id=job_id, source=request.source.strip())
        with _LOCK:
            _JOBS[job_id] = record

        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        if request.wait:
            _run_job(record, cfg, payload)
            return record.to_dict()

        thread = threading.Thread(
            target=_run_job, args=(record, cfg, payload), daemon=True, name=f"job-{job_id}"
        )
        thread.start()
        return {"job_id": job_id, "status": record.status, "poll": f"/jobs/{job_id}"}

    @app.get("/jobs")
    def jobs() -> Dict[str, Any]:
        with _LOCK:
            return {"jobs": [record.to_dict() for record in _JOBS.values()]}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        record = _JOBS.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job_id tidak ditemukan.")
        return record.to_dict()

    return app


def run_server(
    cfg: Config, *, host: Optional[str] = None, port: Optional[int] = None
) -> int:
    """Jalankan uvicorn (dipanggil dari ``python -m tukang_sayur serve``)."""

    try:
        import uvicorn  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TukangSayurError(
            "uvicorn belum terpasang. Install dulu: pip install uvicorn fastapi"
        ) from exc

    app = create_app(cfg)
    host = host or str(cfg.get("server.host", "0.0.0.0"))
    port = int(port or cfg.get("server.port", 8000))
    token = os.environ.get(str(cfg.get("server.api_token_env", "TUKANG_SAYUR_TOKEN")), "")
    if not token:
        log.warning(
            "TUKANG_SAYUR_TOKEN kosong: endpoint terbuka tanpa autentikasi. "
            "Jangan dibuka ke internet."
        )
    log.info("API jalan di http://%s:%d (dokumen: /docs)", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0
