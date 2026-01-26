import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import cv2
import numpy as np


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_odd(value: int, min_value: int = 1) -> int:
    if value < min_value:
        value = min_value
    if value % 2 == 0:
        value += 1
    return value


def safe_resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def overlay_text(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def save_snapshot(frame: np.ndarray, directory: str) -> str:
    ensure_dir(directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    path = os.path.join(directory, filename)
    cv2.imwrite(path, frame)
    return path


class FPSCounter:
    def __init__(self, window_size: int = 30) -> None:
        self.window_size = max(1, window_size)
        self._times = []
        self._last_time = time.time()

    def update(self) -> float:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        self._times.append(dt)
        if len(self._times) > self.window_size:
            self._times.pop(0)
        return self.fps

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        total = sum(self._times)
        if total <= 0:
            return 0.0
        return len(self._times) / total
