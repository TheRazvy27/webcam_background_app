import math
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EffectState:
    bg_image: np.ndarray | None = None
    bg_video: str | None = None
    bg_freeze: np.ndarray | None = None
    show_mask: bool = False
    show_edges: bool = False
    t0: float = time.time()

    def __post_init__(self):
        self._bg_cap = None
        self._bg_cached = None
        self._grid = None
        self._last_shape = None
        if self.bg_video:
            self._bg_cap = cv2.VideoCapture(self.bg_video)

    def close(self):
        if self._bg_cap is not None:
            self._bg_cap.release()

    def now(self):
        return time.time() - self.t0

    def set_freeze(self, frame):
        self.bg_freeze = frame.copy()

    def _resize_bg(self, bg, shape):
        h, w = shape[:2]
        return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

    def _dynamic_gradient(self, shape, t):
        h, w = shape[:2]
        if self._grid is None or self._last_shape != (h, w):
            xs = np.linspace(0, 1, w, dtype=np.float32)
            ys = np.linspace(0, 1, h, dtype=np.float32)
            grid_x, grid_y = np.meshgrid(xs, ys)
            self._grid = (grid_x, grid_y)
            self._last_shape = (h, w)
        grid_x, grid_y = self._grid
        r = 0.5 + 0.5 * np.sin(2 * math.pi * (grid_x * 0.7 + t * 0.08))
        g = 0.5 + 0.5 * np.sin(2 * math.pi * (grid_y * 0.6 + t * 0.11))
        b = 0.5 + 0.5 * np.sin(2 * math.pi * ((grid_x + grid_y) * 0.35 + t * 0.05))
        bg = np.dstack((b, g, r)) * 255.0
        return bg.astype(np.uint8)

    def get_background(self, shape):
        if self.bg_freeze is not None:
            return self._resize_bg(self.bg_freeze, shape)
        if self.bg_image is not None:
            if self._bg_cached is None or self._bg_cached.shape[:2] != shape[:2]:
                self._bg_cached = self._resize_bg(self.bg_image, shape)
            return self._bg_cached.copy()
        if self._bg_cap is not None:
            ok, frame = self._bg_cap.read()
            if not ok:
                self._bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._bg_cap.read()
            if ok:
                return self._resize_bg(frame, shape)
        return self._dynamic_gradient(shape, self.now())


def _blend(fg, bg, alpha):
    alpha = np.clip(alpha, 0.0, 1.0)
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    fg_f = fg.astype(np.float32)
    bg_f = bg.astype(np.float32)
    out = fg_f * alpha + bg_f * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def effect_bokeh(frame, mask_soft, _mask_bin, state):
    bg = cv2.GaussianBlur(frame, (0, 0), sigmaX=18, sigmaY=18)
    return _blend(frame, bg, mask_soft)


def effect_replace(frame, mask_soft, _mask_bin, state):
    bg = state.get_background(frame.shape)
    return _blend(frame, bg, mask_soft)


def effect_color_pop(frame, mask_soft, _mask_bin, state):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _blend(frame, bg, mask_soft)


def effect_pixelate(frame, mask_soft, _mask_bin, state):
    h, w = frame.shape[:2]
    scale = 0.08
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return _blend(frame, bg, mask_soft)


def _edge_glow(mask_bin, color=(0, 255, 255)):
    edges = cv2.Canny(mask_bin * 255, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glow = cv2.dilate(edges, kernel, iterations=1)
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=3)
    glow = glow.astype(np.float32) / 255.0
    glow = glow[:, :, None] * np.array(color, dtype=np.float32)[None, None, :]
    return glow


def effect_neon(frame, mask_soft, mask_bin, state):
    bg = state.get_background(frame.shape)
    tinted = cv2.addWeighted(bg, 0.65, frame, 0.35, 0)
    out = _blend(frame, tinted, mask_soft)
    glow = _edge_glow(mask_bin, color=(0, 255, 200))
    out_f = out.astype(np.float32)
    out_f = np.clip(out_f + glow * 0.9, 0, 255)
    return out_f.astype(np.uint8)


def effect_shadow(frame, mask_soft, mask_bin, state):
    bg = state.get_background(frame.shape)
    base = _blend(frame, bg, mask_soft)
    shadow = cv2.dilate(mask_bin * 255, None, iterations=8)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=9)
    shadow = (shadow.astype(np.float32) / 255.0) * 0.5
    shadow = shadow[:, :, None]
    shift = 12
    shadow_img = np.zeros_like(base, dtype=np.float32)
    shadow_img[shift:, shift:, :] = base[:-shift, :-shift, :]
    out = base.astype(np.float32) * (1.0 - shadow) + shadow_img * shadow
    return np.clip(out, 0, 255).astype(np.uint8)


def effect_glass(frame, mask_soft, _mask_bin, state):
    bg = state.get_background(frame.shape)
    # Subtle refraction-like warp for background
    h, w = frame.shape[:2]
    xs = np.linspace(0, 2 * math.pi, w, dtype=np.float32)
    ys = np.linspace(0, 2 * math.pi, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dx = (np.sin(grid_y * 2.0 + state.now() * 1.3) * 4).astype(np.float32)
    dy = (np.cos(grid_x * 2.0 + state.now() * 1.1) * 4).astype(np.float32)
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)
    warped = cv2.remap(bg, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return _blend(frame, warped, mask_soft)


EFFECTS = {
    "bokeh": effect_bokeh,
    "replace": effect_replace,
    "color_pop": effect_color_pop,
    "pixelate": effect_pixelate,
    "neon": effect_neon,
    "shadow": effect_shadow,
    "glass": effect_glass,
}
