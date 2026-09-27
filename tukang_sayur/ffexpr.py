"""Ekspresi ffmpeg untuk nilai yang berubah menurut waktu (crop yang mengikuti wajah).

ffmpeg tidak punya "keyframe" untuk filter ``crop``, jadi gerakan kamera kita
tuliskan sebagai satu ekspresi bertingkat::

    if(lt(t,1.20), 300, if(lt(t,1.60), (300+(340-300)*(t-1.20)/0.40), ...))

Supaya bisa diuji tanpa menjalankan ffmpeg, :func:`evaluate_keyframes`
menghitung nilai yang **sama persis** dengan ekspresi yang dihasilkan
:func:`piecewise_expression` di Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Keyframe:
    """Nilai pada satu titik waktu (detik, relatif awal klip)."""

    time: float
    value: float


def make_keyframes(pairs: Sequence[Tuple[float, float]]) -> List[Keyframe]:
    """Bentuk daftar keyframe terurut & unik dari pasangan ``(waktu, nilai)``."""

    frames = [Keyframe(float(t), float(v)) for t, v in pairs]
    frames.sort(key=lambda frame: frame.time)
    unique: List[Keyframe] = []
    for frame in frames:
        if unique and abs(frame.time - unique[-1].time) < 1e-6:
            unique[-1] = frame  # waktu sama -> pakai yang terakhir
            continue
        unique.append(frame)
    return unique


def simplify_keyframes(
    keyframes: Sequence[Keyframe], tolerance: float = 0.5
) -> List[Keyframe]:
    """Buang keyframe di tengah garis lurus supaya ekspresinya tidak kepanjangan."""

    frames = list(keyframes)
    if len(frames) <= 2:
        return frames

    kept: List[Keyframe] = [frames[0]]
    for index in range(1, len(frames) - 1):
        previous, current, following = kept[-1], frames[index], frames[index + 1]
        span = following.time - previous.time
        if span <= 0:
            continue
        ratio = (current.time - previous.time) / span
        interpolated = previous.value + (following.value - previous.value) * ratio
        if abs(interpolated - current.value) > tolerance:
            kept.append(current)
    kept.append(frames[-1])
    return kept


def evaluate_keyframes(keyframes: Sequence[Keyframe], time: float) -> float:
    """Nilai pada waktu ``time`` dengan interpolasi linear (dan tahan di ujung)."""

    frames = list(keyframes)
    if not frames:
        raise ValueError("Tidak ada keyframe untuk dievaluasi.")
    if len(frames) == 1 or time <= frames[0].time:
        return frames[0].value
    if time >= frames[-1].time:
        return frames[-1].value
    for index in range(len(frames) - 1):
        left, right = frames[index], frames[index + 1]
        if time < right.time:
            span = right.time - left.time
            if span <= 0:
                return right.value
            ratio = (time - left.time) / span
            return left.value + (right.value - left.value) * ratio
    return frames[-1].value


def _number(value: float, digits: int = 2) -> str:
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-"} else "0"


def piecewise_expression(
    keyframes: Sequence[Keyframe],
    *,
    variable: str = "t",
    digits: int = 2,
) -> str:
    """Ubah keyframe menjadi ekspresi ffmpeg (linear antar titik, datar di ujung)."""

    frames = list(keyframes)
    if not frames:
        raise ValueError("Tidak ada keyframe untuk dijadikan ekspresi.")
    if len(frames) == 1:
        return _number(frames[0].value, digits)
    if all(abs(frame.value - frames[0].value) < 1e-6 for frame in frames):
        return _number(frames[0].value, digits)

    expression = _number(frames[-1].value, digits)  # setelah keyframe terakhir: datar
    for index in range(len(frames) - 2, -1, -1):
        left, right = frames[index], frames[index + 1]
        span = right.time - left.time
        if span <= 0:
            continue
        slope = (right.value - left.value) / span
        # tanda dipisah supaya tidak muncul "+-50" yang membingungkan parser ffmpeg
        sign = "-" if slope < 0 else "+"
        segment = (
            f"({_number(left.value, digits)}{sign}{_number(abs(slope), 4)}"
            f"*({variable}-{_number(left.time, digits)}))"
        )
        expression = (
            f"if(lt({variable},{_number(right.time, digits)}),{segment},{expression})"
        )
    first = frames[0]
    return (
        f"if(lt({variable},{_number(first.time, digits)}),"
        f"{_number(first.value, digits)},{expression})"
    )


def expression_length(expression: str) -> int:
    return len(expression)
