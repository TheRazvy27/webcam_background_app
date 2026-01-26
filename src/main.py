"""
Main entry point for the real-time background segmentation app.
Simple motion-based segmentation (frame differencing).
"""
import argparse
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from . import effects, segmentation, utils


EFFECT_KEYS = {
    ord("1"): "BG_BLUR",
    ord("2"): "BG_COLOR_SHIFT",
    ord("3"): "BG_REPLACE",
    ord("4"): "PERSON_HIGHLIGHT",
}

THRESHOLD_METHODS = ["manual", "otsu", "iterative"]


def _load_background_image(path: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load and resize a background image for BG_REPLACE effect."""
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return utils.safe_resize(img, size)


def _apply_effect(
    frame: np.ndarray,
    mask: np.ndarray,
    config: Dict,
    bg_image: Optional[np.ndarray],
    effect_name: str,
) -> np.ndarray:
    """Apply the selected visual effect using the person mask."""
    if effect_name == "BG_BLUR":
        blur_cfg = config["effects"]["bg_blur"]
        return effects.apply_blur_background(
            frame,
            mask,
            ksize=int(blur_cfg.get("ksize", 25)),
            sigma=float(blur_cfg.get("sigma", 0.0)),
        )
    if effect_name == "BG_COLOR_SHIFT":
        shift_cfg = config["effects"]["bg_hsv_shift"]
        return effects.apply_hsv_shift_background(
            frame,
            mask,
            dh=int(shift_cfg.get("dh", 20)),
            sat_scale=float(shift_cfg.get("sat_scale", 1.2)),
        )
    if effect_name == "BG_REPLACE":
        return effects.apply_replace_background(frame, mask, bg_image)
    if effect_name == "PERSON_HIGHLIGHT":
        hl_cfg = config["effects"]["person_highlight"]
        return effects.apply_person_highlight(
            frame,
            mask,
            alpha=float(hl_cfg.get("alpha", 1.1)),
            beta=float(hl_cfg.get("beta", 10)),
            edge_strength=float(hl_cfg.get("edge_strength", 0.4)),
        )
    return frame.copy()


def _toggle_method(current: str) -> str:
    """Cycle through threshold methods: manual -> otsu -> iterative."""
    if current not in THRESHOLD_METHODS:
        return THRESHOLD_METHODS[0]
    idx = THRESHOLD_METHODS.index(current)
    return THRESHOLD_METHODS[(idx + 1) % len(THRESHOLD_METHODS)]


def main() -> None:
    parser = argparse.ArgumentParser(description="APDSV real-time background effects")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    cam_cfg = config["camera"]
    threshold_cfg = config["threshold"]
    refine_cfg = config["mask_refine"]
    diff_cfg = config.get("diff", {})
    diff_blur_cfg = config.get("diff_blur", {})

    camera_index = args.camera if args.camera is not None else int(cam_cfg.get("index", 0))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    width = int(cam_cfg.get("width", 640))
    height = int(cam_cfg.get("height", 480))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, int(cam_cfg.get("fps", 30)))

    # Read first frame - DON'T use as background yet
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Could not read from camera.")

    frame = utils.safe_resize(frame, (width, height))
    
    # Background starts as None - user MUST press 'b' to capture it
    background_bgr: Optional[np.ndarray] = None

    effect_name = config["effects"].get("active", "BG_BLUR")
    method = threshold_cfg.get("method", "manual")
    manual_t = float(threshold_cfg.get("value", 30))
    iter_max = int(threshold_cfg.get("iterative_max_iters", 20))
    iter_eps = float(threshold_cfg.get("iterative_epsilon", 0.5))

    bg_path = config["effects"]["bg_replace"].get("path", "")
    bg_image = _load_background_image(bg_path, (width, height))

    output_cfg = config["output"]
    show_windows = bool(output_cfg.get("show_windows", True))
    snapshot_dir = output_cfg.get("save_snapshots_dir", "outputs/snapshots")
    video_enabled = bool(output_cfg.get("video_enabled", False))
    video_path = output_cfg.get("video_path", "outputs/output.mp4")
    video_fps = int(output_cfg.get("video_fps", 20))

    writer = None
    if video_enabled:
        utils.ensure_dir(os.path.dirname(video_path) or ".")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))

    fps_counter = utils.FPSCounter()
    font_scale = float(config["ui"].get("font_scale", 0.5))
    thickness = int(config["ui"].get("thickness", 1))
    
    # Debug: show diff image
    show_diff = False

    print("=" * 60)
    print("INSTRUCTIONS:")
    print("1. Step OUT of the camera view (show only background)")
    print("2. Press 'b' to capture the background")
    print("3. Step INTO the camera view")
    print("4. You should appear as WHITE in the Mask window")
    print("5. Use +/- to adjust threshold if needed")
    print("6. Press 'd' to toggle diff view (debug)")
    print("=" * 60)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = utils.safe_resize(frame, (width, height))

        # If no background captured yet, show message and wait
        if background_bgr is None:
            output = frame.copy()
            cv2.putText(output, "Press 'b' to capture background", (50, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(output, "(step out of frame first!)", (50, height // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mask = np.zeros((height, width), dtype=np.float32)
            diff_img = np.zeros((height, width), dtype=np.uint8)
        else:
            # Compute motion mask
            mask, used_t, diff_img = segmentation.motion_mask_with_diff(
                frame,
                background_bgr,
                method=method,
                manual_t=manual_t,
                refine_cfg=refine_cfg,
                diff_cfg=diff_cfg,
                diff_blur_cfg=diff_blur_cfg,
                max_iters=iter_max,
                epsilon=iter_eps,
            )
            output = _apply_effect(frame, mask, config, bg_image, effect_name)
            manual_t = used_t if method == "manual" else manual_t

        # Overlay info
        fps = fps_counter.update()
        if background_bgr is not None:
            utils.overlay_text(
                output,
                f"FPS: {fps:.1f}  Method: {method}  T: {manual_t:.1f}",
                (10, 20),
                font_scale=font_scale,
                thickness=thickness,
            )
            utils.overlay_text(
                output,
                f"Effect: {effect_name}  (d=show diff, b=reset bg)",
                (10, 40),
                font_scale=font_scale,
                thickness=thickness,
            )

        if writer is not None:
            writer.write(output)

        if show_windows:
            cv2.imshow("Original", frame)
            cv2.imshow("Mask (white=person)", (mask * 255).astype(np.uint8))
            cv2.imshow("Output", output)
            if show_diff and background_bgr is not None:
                cv2.imshow("Diff (debug)", diff_img)
            elif not show_diff:
                try:
                    cv2.destroyWindow("Diff (debug)")
                except cv2.error:
                    pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            # Capture current frame as background
            background_bgr = frame.copy()
            print(f"[INFO] Background captured! Now step into frame.")
        if key in EFFECT_KEYS:
            effect_name = EFFECT_KEYS[key]
            print(f"[INFO] Effect: {effect_name}")
        if key == ord("m"):
            method = _toggle_method(method)
            print(f"[INFO] Threshold method: {method}")
        if key in (ord("+"), ord("=")) and method == "manual":
            manual_t = min(255.0, manual_t + 2)
            print(f"[INFO] Threshold: {manual_t}")
        if key in (ord("-"), ord("_")) and method == "manual":
            manual_t = max(0.0, manual_t - 2)
            print(f"[INFO] Threshold: {manual_t}")
        if key == ord("s"):
            utils.save_snapshot(output, snapshot_dir)
        if key == ord("v"):
            if writer is None:
                utils.ensure_dir(os.path.dirname(video_path) or ".")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))
                print("[INFO] Video recording started.")
            else:
                writer.release()
                writer = None
                print("[INFO] Video recording stopped.")
        if key == ord("d"):
            show_diff = not show_diff
            print(f"[INFO] Show diff: {show_diff}")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
