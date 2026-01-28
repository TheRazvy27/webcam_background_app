import math

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - runtime import guard
    mp = None


class FaceReplacer:
    """!Face overlay utility with optional landmark-based mouth estimation."""

    def __init__(self, refresh=6, min_size=60, landmarker_model_path=None, landmark_refresh=3):
        """!Initialize face replacement helper.

        @param refresh Frames between Haar face detections.
        @param min_size Minimum face size for detection.
        @param landmarker_model_path Optional MediaPipe face landmarker model.
        @param landmark_refresh Frames between landmark detections.
        """
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._refresh = int(refresh)
        self._min_size = int(min_size)
        self._landmark_refresh = int(landmark_refresh)
        self._last_box = None
        self._last_update = -1
        self._last_mouth = None
        self._last_landmark_update = -1
        self._landmarker = None

        if landmarker_model_path and mp is not None:
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision

                base = mp_python.BaseOptions(model_asset_path=landmarker_model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base,
                    num_faces=1,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            except Exception:
                self._landmarker = None

    def close(self):
        """!Release landmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()

    def _estimate_mouth_simple(self, frame_bgr, box):
        """!Estimate mouth openness with a simple intensity heuristic.

        @param frame_bgr Input frame (BGR).
        @param box Face bounding box (x,y,w,h).
        @return Mouth openness in [0,1] or None if not available.
        """
        if box is None:
            return None
        x, y, w, h = box
        y0 = y + int(h * 0.6)
        y1 = y + int(h * 0.88)
        x0 = x + int(w * 0.2)
        x1 = x + int(w * 0.8)
        h_frame, w_frame = frame_bgr.shape[:2]
        y0 = max(0, min(h_frame - 1, y0))
        y1 = max(0, min(h_frame, y1))
        x0 = max(0, min(w_frame - 1, x0))
        x1 = max(0, min(w_frame, x1))
        if y1 <= y0 or x1 <= x0:
            return None

        roi = frame_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ratio = float(np.mean(thresh)) / 255.0
        open_ratio = (ratio - 0.15) / 0.35
        return max(0.0, min(open_ratio, 1.0))

    def detect(self, frame_bgr, frame_idx, need_mouth=False):
        """!Detect face box and optional mouth openness.

        @param frame_bgr Input frame (BGR).
        @param frame_idx Current frame index.
        @param need_mouth Whether to estimate mouth openness.
        @return Tuple (box, mouth_open) where box=(x,y,w,h) or None.
        """
        if self._landmarker is not None and need_mouth:
            if (
                self._last_box is not None
                and frame_idx - self._last_landmark_update < self._landmark_refresh
            ):
                return self._last_box, self._last_mouth

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            results = self._landmarker.detect(mp_image)
            if results and results.face_landmarks:
                landmarks = results.face_landmarks[0]
                h, w = frame_bgr.shape[:2]
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                x_min = max(0, int(min(xs) * w))
                y_min = max(0, int(min(ys) * h))
                x_max = min(w, int(max(xs) * w))
                y_max = min(h, int(max(ys) * h))
                if x_max > x_min and y_max > y_min:
                    self._last_box = (x_min, y_min, x_max - x_min, y_max - y_min)

                def _dist(a, b):
                    return math.hypot(a.x - b.x, a.y - b.y)

                try:
                    upper = landmarks[13]
                    lower = landmarks[14]
                    left = landmarks[61]
                    right = landmarks[308]
                    open_ratio = _dist(upper, lower) / (_dist(left, right) + 1e-6)
                    open_ratio = max(0.0, min(open_ratio * 3.0, 1.0))
                    self._last_mouth = open_ratio
                except Exception:
                    self._last_mouth = None

                self._last_landmark_update = frame_idx
                return self._last_box, self._last_mouth

        if self._last_box is not None and frame_idx - self._last_update < self._refresh:
            return self._last_box, self._last_mouth

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(self._min_size, self._min_size),
        )
        if len(faces) == 0:
            return self._last_box, self._last_mouth

        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        self._last_box = (int(x), int(y), int(w), int(h))
        self._last_update = frame_idx
        if need_mouth:
            self._last_mouth = self._estimate_mouth_simple(frame_bgr, self._last_box)
        return self._last_box, self._last_mouth

    def _animate_mouth(self, face_img, openness):
        """!Simple mouth opening animation on the face overlay.

        @param face_img Face image (BGR/BGRA).
        @param openness Mouth openness in [0,1].
        @return Animated face image.
        """
        if openness is None or openness <= 0:
            return face_img
        h, w = face_img.shape[:2]
        split = int(h * 0.62)
        delta = int(openness * h * 0.12)
        if delta < 1 or split >= h - 1:
            return face_img

        out = face_img.copy()
        lower = face_img[split:, :, :]
        dest_start = split + delta
        if dest_start < h:
            out[dest_start:h, :, :] = lower[: h - dest_start, :, :]
        if out.shape[2] == 4:
            out[split:dest_start, :, 3] = 0
        else:
            out[split:dest_start, :, :] = (out[split:dest_start, :, :] * 0.2).astype(
                out.dtype
            )
        return out

    def apply(self, frame_bgr, face_img, box, scale=1.15, mouth_open=None):
        """!Overlay the selected face image onto the detected face box.

        @param frame_bgr Output frame to draw on (in-place).
        @param face_img Face image (BGR/BGRA).
        @param box Face bounding box (x,y,w,h).
        @param scale Scale factor applied to the box size.
        @param mouth_open Optional mouth openness for animation.
        @return Frame with face overlay applied.
        """
        if face_img is None or box is None:
            return frame_bgr

        h_frame, w_frame = frame_bgr.shape[:2]
        x, y, w, h = box

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
        face_resized = self._animate_mouth(face_resized, mouth_open)

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
