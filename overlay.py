"""OpenCV overlay drawing functions."""

from __future__ import annotations

import cv2
import numpy as np


def draw_waveform(frame: np.ndarray, signal: list[float], x: int, y: int) -> None:
    values = np.asarray(signal, dtype=np.float32)
    if values.size < 2 or float(values.std()) <= 1e-6:
        return
    values = (values - values.mean()) / values.std()
    width, height = 400, 120
    x2, y2 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
    if x2 <= x or y2 <= y:
        return
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, "rPPG waveform (15s)", (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    plot = values[np.linspace(0, values.size - 1, max(2, x2 - x - 16)).astype(int)]
    low, high = float(plot.min()), float(plot.max())
    points = [(x + 8 + i, y2 - 8 - int((value - low) / max(high - low, 1e-6) * (y2 - y - 36))) for i, value in enumerate(plot)]
    for first, second in zip(points, points[1:]):
        cv2.line(frame, first, second, (0, 255, 255), 2, cv2.LINE_AA)


def draw_face(frame: np.ndarray, bbox: tuple[int, int, int, int], label: str, waveform: list[float], hr: float | None) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    lines = [label, f"HR: {hr:.0f} bpm" if hr is not None else "HR: -- bpm"]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x1, max(25, y1 - 12 - (len(lines) - index - 1) * 25)), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
    draw_waveform(frame, waveform, x1, y2 + 15)
