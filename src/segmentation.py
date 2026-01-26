"""
Segmentation module: motion-based mask generation with thresholding.
Methods aligned with APDSV course: manual threshold, Otsu, iterative.
"""
from typing import Dict, Tuple

import cv2
import numpy as np

from .utils import safe_odd


def otsu_threshold(diff_gray: np.ndarray) -> float:
    """
    Compute Otsu threshold on a grayscale difference image.
    From APDSV Lab Segmentare: automatic threshold selection.
    """
    if diff_gray.max() == 0:
        return 0.0
    retval, _ = cv2.threshold(
        diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return float(retval)


def iterative_threshold(
    diff_gray: np.ndarray, max_iters: int = 20, epsilon: float = 0.5
) -> float:
    """
    Iterative threshold: T -> split into groups -> compute means -> Tnou = (m1+m2)/2.
    From APDSV Lab Segmentare: procedura iterativa de alegere prag.
    """
    pixels = diff_gray.reshape(-1).astype(np.float32)
    if pixels.size == 0 or pixels.max() == 0:
        return 0.0

    t = float(pixels.mean())
    for _ in range(max_iters):
        lower = pixels[pixels <= t]
        upper = pixels[pixels > t]
        if lower.size == 0 or upper.size == 0:
            break
        m1 = float(lower.mean())
        m2 = float(upper.mean())
        t_new = (m1 + m2) / 2.0
        if abs(t_new - t) < epsilon:
            t = t_new
            break
        t = t_new
    return float(t)


def compute_threshold(
    diff_gray: np.ndarray,
    method: str,
    manual_t: float,
    max_iters: int,
    epsilon: float,
) -> float:
    """Select threshold based on method: manual, otsu, or iterative."""
    if method == "otsu":
        return otsu_threshold(diff_gray)
    if method == "iterative":
        return iterative_threshold(diff_gray, max_iters=max_iters, epsilon=epsilon)
    return float(manual_t)


def _compute_diff(
    frame_bgr: np.ndarray, background_bgr: np.ndarray, mode: str
) -> np.ndarray:
    """
    Compute absolute difference between frame and background.
    Supports: gray, max_rgb, ycbcr_y, ycbcr_chroma.
    """
    bg_u8 = np.clip(background_bgr, 0, 255).astype(np.uint8)
    mode = str(mode).lower()
    
    if mode == "max_rgb":
        diff_b = cv2.absdiff(frame_bgr[:, :, 0], bg_u8[:, :, 0])
        diff_g = cv2.absdiff(frame_bgr[:, :, 1], bg_u8[:, :, 1])
        diff_r = cv2.absdiff(frame_bgr[:, :, 2], bg_u8[:, :, 2])
        return np.maximum(np.maximum(diff_b, diff_g), diff_r)
    
    if mode == "ycbcr_y":
        frame_ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        bg_ycc = cv2.cvtColor(bg_u8, cv2.COLOR_BGR2YCrCb)
        return cv2.absdiff(frame_ycc[:, :, 0], bg_ycc[:, :, 0])
    
    if mode == "ycbcr_chroma":
        frame_ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        bg_ycc = cv2.cvtColor(bg_u8, cv2.COLOR_BGR2YCrCb)
        diff_cb = cv2.absdiff(frame_ycc[:, :, 2], bg_ycc[:, :, 2])
        diff_cr = cv2.absdiff(frame_ycc[:, :, 1], bg_ycc[:, :, 1])
        return np.maximum(diff_cb, diff_cr)
    
    # Default: grayscale
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg_u8, cv2.COLOR_BGR2GRAY)
    return cv2.absdiff(frame_gray, bg_gray)


def _refine_mask(mask_u8: np.ndarray, refine_cfg: Dict) -> np.ndarray:
    """
    Apply refinements to binary mask: median filter, morphology.
    From APDSV: filtrare median pentru reducerea zgomotului.
    """
    # Median filter to remove salt-and-pepper noise
    median_ksize = int(refine_cfg.get("median_ksize", 0))
    if median_ksize >= 3:
        median_ksize = safe_odd(median_ksize, 3)
        mask_u8 = cv2.medianBlur(mask_u8, median_ksize)

    # Morphological operations (optional)
    morph_cfg = refine_cfg.get("morph", {})
    if morph_cfg.get("enabled", False):
        kernel_size = safe_odd(int(morph_cfg.get("kernel_size", 3)), 3)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mode = str(morph_cfg.get("mode", "close")).lower()
        iters = max(1, int(morph_cfg.get("dilate_iters", 1)))
        
        if mode == "close":
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=iters)
        elif mode == "open":
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=iters)
        elif mode == "dilate":
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=iters)
        elif mode == "erode":
            mask_u8 = cv2.erode(mask_u8, kernel, iterations=iters)
    
    return mask_u8


def motion_mask_with_diff(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray,
    method: str,
    manual_t: float,
    refine_cfg: Dict,
    diff_cfg: Dict,
    diff_blur_cfg: Dict,
    max_iters: int = 20,
    epsilon: float = 0.5,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute person mask using frame differencing.
    Returns: (mask_float, threshold_used, diff_image_for_debug)
    
    Pipeline:
    1. Compute |frame - background| (grayscale or color space)
    2. Optional: blur the diff to reduce noise
    3. Compute threshold (manual, Otsu, or iterative)
    4. Binarize: pixels > threshold = 255 (person), else 0 (background)
    5. Refine mask: median filter, morphology
    6. Convert to float [0, 1] with optional Gaussian feathering
    """
    # Step 1: Compute difference
    diff_u8 = _compute_diff(frame_bgr, background_bgr, diff_cfg.get("mode", "gray"))
    
    # Step 2: Blur the diff to reduce noise
    if diff_blur_cfg.get("enabled", True):
        ksize = int(diff_blur_cfg.get("ksize", 5))
        sigma = float(diff_blur_cfg.get("sigma", 0.0))
        if ksize >= 3:
            ksize = safe_odd(ksize, 3)
            diff_u8 = cv2.GaussianBlur(diff_u8, (ksize, ksize), sigma)
    
    # Save diff for debug display
    diff_debug = diff_u8.copy()
    
    # Step 3: Compute threshold
    threshold_value = compute_threshold(
        diff_u8, method, manual_t, max_iters=max_iters, epsilon=epsilon
    )
    
    # Step 4: Binarize - pixels above threshold = 255 (person/foreground)
    _, mask_u8 = cv2.threshold(diff_u8, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Step 5: Refine mask
    mask_u8 = _refine_mask(mask_u8, refine_cfg)
    
    # Step 6: Convert to float and optionally feather
    mask_float = mask_u8.astype(np.float32) / 255.0
    
    if refine_cfg.get("feather", True):
        gaussian_ksize = int(refine_cfg.get("gaussian_ksize", 0))
        gaussian_sigma = float(refine_cfg.get("gaussian_sigma", 0.0))
        if gaussian_ksize >= 3:
            gaussian_ksize = safe_odd(gaussian_ksize, 3)
            mask_float = cv2.GaussianBlur(
                mask_float, (gaussian_ksize, gaussian_ksize), gaussian_sigma
            )
        mask_float = np.clip(mask_float, 0.0, 1.0)
    
    return mask_float, float(threshold_value), diff_debug


# Keep old function signature for compatibility
def motion_mask(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray,
    method: str,
    manual_t: float,
    refine_cfg: Dict,
    diff_cfg: Dict,
    diff_blur_cfg: Dict,
    max_iters: int = 20,
    epsilon: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Wrapper that returns just mask and threshold (no diff debug)."""
    mask, threshold, _ = motion_mask_with_diff(
        frame_bgr, background_bgr, method, manual_t,
        refine_cfg, diff_cfg, diff_blur_cfg, max_iters, epsilon
    )
    return mask, threshold
