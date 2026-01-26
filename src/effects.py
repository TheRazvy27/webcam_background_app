from typing import Optional, Tuple

import cv2
import numpy as np


def _ensure_3ch_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return np.repeat(mask[:, :, None], 3, axis=2)
    return mask


def _composite(frame: np.ndarray, bg_processed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_3ch = _ensure_3ch_mask(mask).astype(np.float32)
    frame_f = frame.astype(np.float32)
    bg_f = bg_processed.astype(np.float32)
    out = frame_f * mask_3ch + bg_f * (1.0 - mask_3ch)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_blur_background(
    frame: np.ndarray, mask: np.ndarray, ksize: int, sigma: float
) -> np.ndarray:
    ksize = max(3, ksize)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
    return _composite(frame, blurred, mask)


def apply_hsv_shift_background(
    frame: np.ndarray, mask: np.ndarray, dh: int, sat_scale: float
) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h = (h + dh) % 180.0
    s = np.clip(s * sat_scale, 0, 255)
    hsv_shifted = cv2.merge([h, s, v]).astype(np.uint8)
    shifted_bgr = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)
    return _composite(frame, shifted_bgr, mask)


def apply_replace_background(
    frame: np.ndarray, mask: np.ndarray, bg_image: Optional[np.ndarray]
) -> np.ndarray:
    if bg_image is None:
        return frame.copy()
    return _composite(frame, bg_image, mask)


def apply_person_highlight(
    frame: np.ndarray, mask: np.ndarray, alpha: float, beta: float, edge_strength: float
) -> np.ndarray:
    enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    out = _composite(enhanced, frame, mask)

    if edge_strength > 0:
        mask_u8 = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_u8, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, 1.0, edges_bgr, edge_strength, 0)
    return out
