import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - runtime import guard
    mp = None
    _IMPORT_ERROR = exc


class PersonSegmenter:
    def __init__(
        self,
        model_selection=1,
        threshold=0.5,
        smooth=0.6,
        blur=7,
        morph=3,
        model_path=None,
    ):
        if mp is None:
            raise RuntimeError(
                "mediapipe is required for segmentation. Install with: pip install mediapipe"
            ) from _IMPORT_ERROR
        self._backend = None
        self._segmenter = None
        self._use_tasks = not hasattr(mp, "solutions")

        if not self._use_tasks:
            self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=model_selection
            )
            self._backend = "solutions"
        else:
            if not model_path:
                raise RuntimeError(
                    "MediaPipe Tasks detected (no mp.solutions). Provide a selfie segmentation model "
                    "via --model-path."
                )
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.ImageSegmenterOptions(
                base_options=base,
                output_category_mask=False,
                output_confidence_masks=True,
            )
            self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)
            self._backend = "tasks"
        self.threshold = float(threshold)
        self.smooth = float(smooth)
        self.blur = int(blur)
        self.morph = int(morph)
        self._prev_mask = None

    def reset(self):
        self._prev_mask = None

    def close(self):
        if self._segmenter is not None:
            self._segmenter.close()

    def __call__(self, frame_bgr):
        # Returns: mask_soft (float32 0..1), mask_bin (uint8 0/1)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            rgb.flags.writeable = False
            results = self._segmenter.process(rgb)
            rgb.flags.writeable = True

            if results is None or results.segmentation_mask is None:
                h, w = frame_bgr.shape[:2]
                mask = np.zeros((h, w), dtype=np.float32)
            else:
                mask = results.segmentation_mask.astype(np.float32)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self._segmenter.segment(mp_image)
            mask = None
            if results is not None and getattr(results, "confidence_masks", None):
                mask = results.confidence_masks[0].numpy_view()
            elif results is not None and getattr(results, "category_mask", None):
                cat = results.category_mask.numpy_view()
                mask = (cat > 0).astype(np.float32)
            if mask is None:
                h, w = frame_bgr.shape[:2]
                mask = np.zeros((h, w), dtype=np.float32)

        if self._prev_mask is None:
            self._prev_mask = mask
        else:
            mask = self.smooth * self._prev_mask + (1.0 - self.smooth) * mask
            self._prev_mask = mask

        if self.blur > 0:
            k = self.blur * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        mask_bin = (mask > self.threshold).astype(np.uint8)

        if self.morph > 0:
            k = self.morph * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)

        mask_soft = mask * mask_bin
        return mask_soft, mask_bin
