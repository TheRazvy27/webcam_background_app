import cv2
import numpy as np


class FaceReplacer:
    def __init__(self, refresh=6, min_size=60):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._refresh = int(refresh)
        self._min_size = int(min_size)
        self._last_box = None
        self._last_update = -1

    def detect(self, frame_bgr, frame_idx):
        if self._last_box is not None and frame_idx - self._last_update < self._refresh:
            return self._last_box

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(self._min_size, self._min_size),
        )
        if len(faces) == 0:
            return self._last_box

        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        self._last_box = (int(x), int(y), int(w), int(h))
        self._last_update = frame_idx
        return self._last_box

    def apply(self, frame_bgr, face_img, box):
        if face_img is None or box is None:
            return frame_bgr

        h_frame, w_frame = frame_bgr.shape[:2]
        x, y, w, h = box

        scale = 1.15
        cx = x + w // 2
        cy = y + h // 2
        w = int(w * scale)
        h = int(h * scale)
        x = max(0, cx - w // 2)
        y = max(0, cy - h // 2)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        if w <= 0 or h <= 0:
            return frame_bgr

        face_resized = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_LINEAR)

        if len(face_resized.shape) == 2:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
        elif face_resized.shape[2] == 1:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)

        if face_resized.shape[2] == 4:
            alpha = face_resized[:, :, 3].astype(np.float32) / 255.0
            face_rgb = face_resized[:, :, :3]
        else:
            face_rgb = face_resized
            alpha = np.zeros((h, w), dtype=np.float32)
            center = (w // 2, h // 2)
            axes = (max(1, int(w * 0.45)), max(1, int(h * 0.55)))
            cv2.ellipse(alpha, center, axes, 0, 0, 360, 1.0, -1)
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=4)

        roi = frame_bgr[y : y + h, x : x + w].astype(np.float32)
        face_rgb = face_rgb.astype(np.float32)
        alpha = alpha[:, :, None]
        blended = roi * (1.0 - alpha) + face_rgb * alpha
        frame_bgr[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)
        return frame_bgr
