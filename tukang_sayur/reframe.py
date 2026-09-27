"""Tahap 5 - Reframe 9:16: ikuti wajah/subjek, dengan pemulusan gerak.

Alur:

1. Ambil sampel frame (default 5 frame/detik) dari klip.
2. Deteksi wajah dengan mediapipe (kalau terpasang) atau OpenCV Haar cascade.
3. Haluskan titik pusat: median filter -> EMA -> deadzone -> clamp ke tepi frame.
4. Ubah jadi keyframe crop, lalu jadi ekspresi ffmpeg.

Kalau tidak ada wajah terdeteksi (atau OpenCV tidak terpasang), rencananya
turun ke mode ``blur`` (video utuh di tengah, latar belakang blur) atau
``center`` sesuai ``reframe.fallback`` di config. Jadi pipeline tidak pernah
berhenti hanya karena tidak ada wajah.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import Config
from .ffexpr import Keyframe, make_keyframes, piecewise_expression, simplify_keyframes
from .logging_utils import get_logger
from .utils import clamp, even

log = get_logger("reframe")

#: Mode hasil perencanaan.
MODE_TRACK = "track"
MODE_CENTER = "center"
MODE_BLUR = "blur"


@dataclass
class Detection:
    """Satu hasil deteksi pada waktu tertentu."""

    time: float
    center_x: float
    center_y: float = 0.0
    confidence: float = 1.0


@dataclass
class ReframePlan:
    """Rencana pemotongan 9:16 untuk satu klip."""

    mode: str
    crop_width: int = 0
    crop_height: int = 0
    frame_width: int = 0
    frame_height: int = 0
    keyframes: List[Keyframe] = field(default_factory=list)
    detection_ratio: float = 0.0
    note: str = ""

    @property
    def is_tracking(self) -> bool:
        return self.mode == MODE_TRACK and len(self.keyframes) > 0

    def crop_x_expression(self) -> str:
        """Ekspresi ``x`` untuk filter crop (0 kalau tidak melacak)."""

        if not self.keyframes:
            return "0"
        return piecewise_expression(self.keyframes)

    def describe(self) -> str:
        if self.mode == MODE_TRACK:
            return (
                f"track ({len(self.keyframes)} keyframe, "
                f"wajah terdeteksi {self.detection_ratio * 100:.0f}% sampel)"
            )
        return f"{self.mode}{f' - {self.note}' if self.note else ''}"


# --------------------------------------------------------------------------
# Pemulusan (fungsi murni, diuji di tests/test_reframe.py)
# --------------------------------------------------------------------------
def median_filter(values: Sequence[float], window: int) -> List[float]:
    """Median bergerak untuk membuang deteksi yang melompat tiba-tiba."""

    data = [float(v) for v in values]
    if window <= 1 or len(data) <= 2:
        return data
    if window % 2 == 0:
        window += 1
    half = window // 2
    # Tepi kiri/kanan diperpanjang (edge padding) supaya lonjakan di frame
    # pertama/terakhir tidak ikut bocor ke hasil.
    padded = [data[0]] * half + data + [data[-1]] * half
    return [
        statistics.median(padded[index : index + window]) for index in range(len(data))
    ]


def exponential_smoothing(values: Sequence[float], alpha: float) -> List[float]:
    """EMA: makin kecil ``alpha``, makin halus (dan makin lambat mengikuti)."""

    data = [float(v) for v in values]
    if not data:
        return []
    alpha = clamp(float(alpha), 0.01, 1.0)
    out = [data[0]]
    for value in data[1:]:
        out.append(out[-1] + alpha * (value - out[-1]))
    return out


def apply_deadzone(values: Sequence[float], threshold: float) -> List[float]:
    """Diamkan kamera untuk gerakan kecil supaya hasil tidak goyang."""

    data = [float(v) for v in values]
    if not data or threshold <= 0:
        return data
    out = [data[0]]
    for value in data[1:]:
        out.append(value if abs(value - out[-1]) >= threshold else out[-1])
    return out


def clamp_positions(
    centers: Sequence[float], crop_width: float, frame_width: float
) -> List[float]:
    """Ubah titik pusat jadi koordinat kiri crop yang selalu di dalam frame."""

    limit = max(0.0, float(frame_width) - float(crop_width))
    return [clamp(float(center) - crop_width / 2.0, 0.0, limit) for center in centers]


def smooth_centers(
    detections: Sequence[Detection],
    *,
    window: int,
    alpha: float,
    deadzone: float,
) -> List[Tuple[float, float]]:
    """Gabungan median -> EMA -> deadzone. Hasil: ``[(waktu, pusat_x)]``."""

    if not detections:
        return []
    times = [d.time for d in detections]
    centers = [d.center_x for d in detections]
    centers = median_filter(centers, window)
    centers = exponential_smoothing(centers, alpha)
    centers = apply_deadzone(centers, deadzone)
    return list(zip(times, centers))


def interpolate_missing(
    detections: Sequence[Detection],
    times: Sequence[float],
    *,
    fallback_center: float,
) -> List[Detection]:
    """Isi waktu tanpa deteksi dengan nilai deteksi terdekat/interpolasi.

    Tanpa ini, wajah yang hilang sebentar (menoleh, terhalang) membuat kamera
    melompat ke tengah lalu balik lagi.
    """

    if not times:
        return []
    known = sorted(detections, key=lambda d: d.time)
    if not known:
        return [Detection(time=t, center_x=fallback_center, center_y=0.0, confidence=0.0)
                for t in times]

    out: List[Detection] = []
    for time in times:
        before = [d for d in known if d.time <= time]
        after = [d for d in known if d.time >= time]
        if before and after:
            left, right = before[-1], after[0]
            span = right.time - left.time
            if span <= 0:
                center = right.center_x
            else:
                ratio = (time - left.time) / span
                center = left.center_x + (right.center_x - left.center_x) * ratio
            confidence = min(left.confidence, right.confidence)
        elif before:
            center, confidence = before[-1].center_x, before[-1].confidence
        else:
            center, confidence = after[0].center_x, after[0].confidence
        out.append(
            Detection(time=time, center_x=center, center_y=0.0, confidence=confidence)
        )
    return out


def build_keyframes(
    samples: Sequence[Tuple[float, float]], *, interval: float = 0.4
) -> List[Keyframe]:
    """Kurangi jumlah sampel jadi keyframe tiap ``interval`` detik."""

    if not samples:
        return []
    picked: List[Tuple[float, float]] = [samples[0]]
    for time, value in samples[1:]:
        if time - picked[-1][0] >= max(0.05, interval):
            picked.append((time, value))
    if picked[-1][0] != samples[-1][0]:
        picked.append(samples[-1])
    return simplify_keyframes(make_keyframes(picked), tolerance=2.0)


def crop_size_for(frame_width: int, frame_height: int, target_ratio: float) -> Tuple[int, int]:
    """Ukuran crop terbesar dengan rasio target yang masih muat di frame."""

    frame_width = int(frame_width)
    frame_height = int(frame_height)
    if frame_width <= 0 or frame_height <= 0:
        return 0, 0
    crop_height = frame_height
    crop_width = even(crop_height * target_ratio)
    if crop_width > frame_width:
        crop_width = even(frame_width)
        crop_height = even(crop_width / target_ratio)
    return max(2, crop_width), max(2, min(even(crop_height), even(frame_height)))


# --------------------------------------------------------------------------
# Deteksi
# --------------------------------------------------------------------------
def _load_cv2():
    try:
        import cv2  # type: ignore[import-not-found]

        return cv2
    except ImportError:
        return None


def _detect_with_mediapipe(cv2, frames: Sequence[Tuple[float, object]], min_confidence: float):
    try:
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError:
        return None

    detections: List[Detection] = []
    solution = mp.solutions.face_detection
    with solution.FaceDetection(
        model_selection=1, min_detection_confidence=min_confidence
    ) as detector:
        for time, frame in frames:
            height, width = frame.shape[:2]  # type: ignore[union-attr]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            if not result.detections:
                continue
            best = max(
                result.detections,
                key=lambda d: d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height,
            )
            box = best.location_data.relative_bounding_box
            detections.append(
                Detection(
                    time=time,
                    center_x=(box.xmin + box.width / 2.0) * width,
                    center_y=(box.ymin + box.height / 2.0) * height,
                    confidence=float(best.score[0]) if best.score else 1.0,
                )
            )
    return detections


def _detect_with_opencv(cv2, frames: Sequence[Tuple[float, object]]):
    cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        log.warning("Haar cascade tidak bisa dimuat dari %s", cascade_path)
        return None

    detections: List[Detection] = []
    for time, frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        detections.append(
            Detection(time=time, center_x=x + w / 2.0, center_y=y + h / 2.0, confidence=1.0)
        )
    return detections


def sample_frames(cv2, video_path: Path, sample_fps: float) -> Tuple[List[Tuple[float, object]], int, int]:
    """Baca frame pada interval tetap. Mengembalikan (frames, lebar, tinggi)."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        log.warning("OpenCV tidak bisa membuka %s", video_path)
        return [], 0, 0
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        step = max(1, int(round(fps / max(0.5, sample_fps))))

        frames: List[Tuple[float, object]] = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                frames.append((index / fps, frame))
            index += 1
        return frames, width, height
    finally:
        capture.release()


# --------------------------------------------------------------------------
# Perencanaan
# --------------------------------------------------------------------------
def fallback_plan(cfg: Config, frame_width: int, frame_height: int, note: str) -> ReframePlan:
    mode = str(cfg.get("reframe.fallback", "blur")).lower()
    mode = MODE_BLUR if mode == "blur" else MODE_CENTER
    log.info("Reframe fallback: %s (%s)", mode, note)
    return ReframePlan(
        mode=mode, frame_width=frame_width, frame_height=frame_height, note=note
    )


def plan_reframe(
    clip_path: str | Path,
    cfg: Config,
    *,
    frame_width: int = 0,
    frame_height: int = 0,
) -> ReframePlan:
    """Susun rencana reframe untuk satu klip yang sudah dipotong."""

    clip_path = Path(clip_path)
    target_width = int(cfg.get("render.width", 1080))
    target_height = int(cfg.get("render.height", 1920))
    target_ratio = target_width / max(1, target_height)

    if not bool(cfg.get("reframe.enabled", True)):
        return fallback_plan(cfg, frame_width, frame_height, "reframe dimatikan di config")

    detector_choice = str(cfg.get("reframe.detector", "auto")).lower()
    if detector_choice == "none":
        return fallback_plan(cfg, frame_width, frame_height, "detector: none")

    cv2 = _load_cv2()
    if cv2 is None:
        return fallback_plan(
            cfg, frame_width, frame_height,
            "OpenCV tidak terpasang (pip install opencv-python-headless)",
        )

    sample_fps = float(cfg.get("reframe.sample_fps", 5))
    frames, width, height = sample_frames(cv2, clip_path, sample_fps)
    frame_width = width or frame_width
    frame_height = height or frame_height
    if not frames or not frame_width or not frame_height:
        return fallback_plan(cfg, frame_width, frame_height, "gagal membaca frame")

    if frame_width / frame_height <= target_ratio + 1e-6:
        return fallback_plan(
            cfg, frame_width, frame_height,
            f"sumber {frame_width}x{frame_height} sudah tidak lebih lebar dari 9:16",
        )

    min_confidence = float(cfg.get("reframe.min_face_confidence", 0.5))
    detections: Optional[List[Detection]] = None
    used = ""
    if detector_choice in {"auto", "mediapipe"}:
        detections = _detect_with_mediapipe(cv2, frames, min_confidence)
        used = "mediapipe"
    if not detections and detector_choice in {"auto", "opencv"}:
        detections = _detect_with_opencv(cv2, frames)
        used = "opencv"

    if not detections:
        return fallback_plan(cfg, frame_width, frame_height, "tidak ada wajah terdeteksi")

    ratio = len(detections) / len(frames)
    log.info(
        "Deteksi wajah (%s): %d/%d sampel (%.0f%%)",
        used, len(detections), len(frames), ratio * 100,
    )
    if ratio < 0.15:
        return fallback_plan(
            cfg, frame_width, frame_height,
            f"wajah hanya terdeteksi di {ratio * 100:.0f}% sampel",
        )

    crop_width, crop_height = crop_size_for(frame_width, frame_height, target_ratio)
    filled = interpolate_missing(
        detections, [time for time, _ in frames], fallback_center=frame_width / 2.0
    )
    samples = smooth_centers(
        filled,
        window=int(cfg.get("reframe.smoothing_window", 9)),
        alpha=float(cfg.get("reframe.ema_alpha", 0.25)),
        deadzone=float(cfg.get("reframe.deadzone_px", 24)),
    )
    times = [time for time, _ in samples]
    positions = clamp_positions([center for _, center in samples], crop_width, frame_width)
    keyframes = build_keyframes(
        list(zip(times, positions)),
        interval=float(cfg.get("reframe.keyframe_interval", 0.4)),
    )

    plan = ReframePlan(
        mode=MODE_TRACK,
        crop_width=crop_width,
        crop_height=crop_height,
        frame_width=frame_width,
        frame_height=frame_height,
        keyframes=keyframes,
        detection_ratio=ratio,
        note=used,
    )
    log.info(
        "Rencana reframe: crop %dx%d dari %dx%d, %d keyframe",
        crop_width, crop_height, frame_width, frame_height, len(keyframes),
    )
    return plan
