import argparse
import glob
import os
import time

import cv2

from effects import EFFECTS, EffectState
from face_replace import FaceReplacer
from segmentation import PersonSegmenter
from utils import FPSCounter, draw_hud


EFFECT_ORDER = ["bokeh", "replace", "color_pop", "pixelate", "neon", "shadow", "glass"]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_THRESHOLD = 0.5
DEFAULT_SMOOTH = 0.6
DEFAULT_BLUR = 3
DEFAULT_MORPH = 1


def default_model_path():
    candidates = [
        os.path.join(ROOT_DIR, "models", "selfie_segmenter.tflite"),
        os.path.join(ROOT_DIR, "model", "selfie_segmenter.tflite"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_resource_images(prefix, flags=cv2.IMREAD_COLOR):
    patterns = [
        os.path.join(RESOURCES_DIR, f"{prefix}*.jpg"),
        os.path.join(RESOURCES_DIR, f"{prefix}*.jpeg"),
        os.path.join(RESOURCES_DIR, f"{prefix}*.png"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(files)

    images = []
    for path in files:
        img = cv2.imread(path, flags)
        if img is not None:
            images.append((path, img))
    return images


def init_controls(bg_count, face_count):
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 420, 260)
    cv2.createTrackbar("Effect", "Controls", 0, max(0, len(EFFECT_ORDER) - 1), lambda _: None)
    cv2.createTrackbar("Background", "Controls", 0, max(0, bg_count), lambda _: None)
    cv2.createTrackbar("Face", "Controls", 0, max(0, face_count), lambda _: None)
    cv2.createTrackbar("Threshold", "Controls", int(DEFAULT_THRESHOLD * 100), 100, lambda _: None)
    cv2.createTrackbar("Smooth", "Controls", int(DEFAULT_SMOOTH * 100), 100, lambda _: None)
    cv2.createTrackbar("Blur", "Controls", DEFAULT_BLUR, 15, lambda _: None)
    cv2.createTrackbar("Morph", "Controls", DEFAULT_MORPH, 10, lambda _: None)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Real-time person segmentation and visual augmentation"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-flip", action="store_true", help="Disable mirror flip")
    parser.add_argument("--record", type=str, default=None, help="Start recording to file")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a MediaPipe selfie segmentation model (optional)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.model_path is None:
        args.model_path = default_model_path()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    bg_resources = load_resource_images("background", flags=cv2.IMREAD_COLOR)
    face_resources = load_resource_images("face", flags=cv2.IMREAD_UNCHANGED)
    bg_images = [img for _, img in bg_resources]
    face_images = [img for _, img in face_resources]

    if bg_resources:
        print(f"[INFO] Loaded {len(bg_resources)} backgrounds from {RESOURCES_DIR}")
    if face_resources:
        print(f"[INFO] Loaded {len(face_resources)} faces from {RESOURCES_DIR}")

    state = EffectState(bg_image=None, bg_video=None)
    segmenter = PersonSegmenter(
        model_selection=1,
        threshold=DEFAULT_THRESHOLD,
        smooth=DEFAULT_SMOOTH,
        blur=DEFAULT_BLUR,
        morph=DEFAULT_MORPH,
        model_path=args.model_path,
    )
    face_replacer = FaceReplacer(refresh=6, min_size=60)

    init_controls(len(bg_images), len(face_images))
    cv2.namedWindow("AugViz", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    fps_counter = FPSCounter(window=30)

    run_dir = None
    frame_idx = 0
    writer = None
    recording = False
    dump_enabled = False
    dump_every = 15
    output_dir = os.path.join(ROOT_DIR, "runs")

    def ensure_run_dir():
        nonlocal run_dir
        if run_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(output_dir, stamp)
            os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def start_recording(path=None):
        nonlocal writer, recording
        if recording:
            return
        if path is None:
            out_dir = ensure_run_dir()
            path = os.path.join(out_dir, "record.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
        recording = writer.isOpened()
        if not recording:
            writer = None
            print("[WARN] Could not start recording.")
        else:
            print(f"[INFO] Recording started: {path}")

    def stop_recording():
        nonlocal writer, recording
        if writer is not None:
            writer.release()
        writer = None
        recording = False

    if args.record:
        start_recording(args.record)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            effect_idx = cv2.getTrackbarPos("Effect", "Controls")
            effect_idx = min(effect_idx, len(EFFECT_ORDER) - 1)
            effect_name = EFFECT_ORDER[effect_idx]

            bg_idx = cv2.getTrackbarPos("Background", "Controls")
            if bg_idx > 0 and bg_images:
                state.bg_image = bg_images[min(bg_idx - 1, len(bg_images) - 1)]
            else:
                state.bg_image = None

            face_idx = cv2.getTrackbarPos("Face", "Controls")
            face_img = None
            if face_idx > 0 and face_images:
                face_img = face_images[min(face_idx - 1, len(face_images) - 1)]

            segmenter.threshold = cv2.getTrackbarPos("Threshold", "Controls") / 100.0
            segmenter.smooth = cv2.getTrackbarPos("Smooth", "Controls") / 100.0
            blur_val = cv2.getTrackbarPos("Blur", "Controls")
            if blur_val < 1:
                blur_val = 1
                cv2.setTrackbarPos("Blur", "Controls", blur_val)
            segmenter.blur = blur_val
            segmenter.morph = cv2.getTrackbarPos("Morph", "Controls")

            mask_soft, mask_bin = segmenter(frame)
            effect_fn = EFFECTS.get(effect_name, EFFECTS["bokeh"])
            output = effect_fn(frame, mask_soft, mask_bin, state)

            if face_img is not None:
                face_box = face_replacer.detect(frame, frame_idx)
                output = face_replacer.apply(output, face_img, face_box)

            if state.show_mask:
                mask_vis = (mask_soft * 255).astype(np.uint8)
                mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_TURBO)
                h, w = output.shape[:2]
                m_h = int(h * 0.25)
                m_w = int(w * 0.25)
                mask_vis = cv2.resize(mask_vis, (m_w, m_h))
                output[10 : 10 + m_h, w - m_w - 10 : w - 10] = mask_vis

            fps = fps_counter.tick()
            output = draw_hud(output, fps, effect_name, state, help_text=True)

            if recording and writer is not None:
                writer.write(output)

            if dump_enabled and frame_idx % max(1, dump_every) == 0:
                out_dir = ensure_run_dir()
                cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:06d}.png"), frame)
                cv2.imwrite(
                    os.path.join(out_dir, f"mask_{frame_idx:06d}.png"),
                    (mask_soft * 255).astype(np.uint8),
                )
                cv2.imwrite(
                    os.path.join(out_dir, f"mask_bin_{frame_idx:06d}.png"),
                    (mask_bin * 255).astype(np.uint8),
                )

            cv2.imshow("AugViz", output)
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("m"):
                state.show_mask = not state.show_mask
            if key == ord("f"):
                state.set_freeze(frame)
                print("[INFO] Background frozen.")
            if key == ord("r"):
                if recording:
                    stop_recording()
                    print("[INFO] Recording stopped.")
                else:
                    start_recording()
            if key == ord("s"):
                out_dir = ensure_run_dir()
                snap_path = os.path.join(out_dir, f"snapshot_{frame_idx:06d}.png")
                cv2.imwrite(snap_path, output)
                print(f"[INFO] Snapshot saved: {snap_path}")
            if key == ord("d"):
                dump_enabled = not dump_enabled
                print(f"[INFO] Dump {'enabled' if dump_enabled else 'disabled'}.")
            if ord("1") <= key <= ord("7"):
                idx = key - ord("1")
                if idx < len(EFFECT_ORDER):
                    cv2.setTrackbarPos("Effect", "Controls", idx)

            frame_idx += 1
    finally:
        stop_recording()
        segmenter.close()
        state.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
