"""
detect_mask_video.py
--------------------
Real-time face mask detection using webcam.

Flow:
  1. Detect faces using OpenCV DNN face detector
  2. For each face ROI → run mask classifier (MobileNetV2)
  3. Overlay bounding box + label + confidence

Usage:
  python detect_mask_video.py
  python detect_mask_video.py --model model/mask_detector.h5 --confidence 0.5
"""

import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# ─── ARGUMENT PARSER ───────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Real-time Face Mask Detection")
    ap.add_argument("-f", "--face",       default="face_detector",
                    help="Path to face detector model directory")
    ap.add_argument("-m", "--model",      default="model/mask_detector.h5",
                    help="Path to trained mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="Minimum face detection confidence")
    ap.add_argument("--camera",           type=int, default=0,
                    help="Camera device index (default: 0)")
    ap.add_argument("--width",            type=int, default=1280,
                    help="Camera frame width")
    ap.add_argument("--height",           type=int, default=720,
                    help="Camera frame height")
    return vars(ap.parse_args())


# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASSES     = ["with_mask", "without_mask"]

# Color palette (BGR)
COLOR_MASK    = (0, 200, 0)      # Green
COLOR_NOMASK  = (0, 0, 220)      # Red
COLOR_INFO    = (255, 255, 255)  # White
COLOR_BG      = (40, 40, 40)     # Dark bg for text


# ─── FACE DETECTOR LOADER ──────────────────────────────────────────────────────
def load_face_detector(face_dir: str):
    """Load OpenCV DNN face detector (ResNet10 + SSD)."""
    prototxt = os.path.join(face_dir, "deploy.prototxt")
    weights  = os.path.join(face_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt) or not os.path.exists(weights):
        print("[INFO] Face detector files not found, downloading ...")
        _download_face_detector(face_dir)

    net = cv2.dnn.readNet(prototxt, weights)
    return net


def _download_face_detector(face_dir: str):
    """Download the OpenCV DNN face detector model files."""
    import urllib.request
    os.makedirs(face_dir, exist_ok=True)

    files = {
        "deploy.prototxt":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel":
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }

    for fname, url in files.items():
        dest = os.path.join(face_dir, fname)
        if not os.path.exists(dest):
            print(f"  Downloading {fname} ...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  ✓ Saved → {dest}")
            except Exception as e:
                print(f"  ✗ Download failed: {e}")
                print(f"  Please manually download from:\n    {url}")
                sys.exit(1)


# ─── DETECTION FUNCTION ────────────────────────────────────────────────────────
def detect_and_predict_mask(frame, face_net, mask_net, min_confidence: float):
    """
    Detect faces in frame and classify mask/no-mask for each face.

    Returns:
        locs   : list of (startX, startY, endX, endY) face bounding boxes
        preds  : list of (with_mask_prob, without_mask_prob)
    """
    (h, w) = frame.shape[:2]
    locs, preds = [], []

    # Create blob from frame for face detector
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue

        box    = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (sX, sY, eX, eY) = box.astype("int")

        # Clamp to frame dimensions
        sX, sY = max(0, sX), max(0, sY)
        eX, eY = min(w - 1, eX), min(h - 1, eY)

        face = frame[sY:eY, sX:eX]
        if face.size == 0:
            continue

        # Preprocess for mask classifier
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((sX, sY, eX, eY))

    # Batch predict
    if faces:
        faces_arr = np.array(faces, dtype="float32")
        preds     = mask_net.predict(faces_arr, batch_size=32, verbose=0)

    return locs, preds


# ─── DRAWING UTILITIES ─────────────────────────────────────────────────────────
def draw_label(frame, text, origin, color, bg_color=COLOR_BG,
               font_scale=0.6, thickness=2):
    """Draw text with a filled background rectangle."""
    (x, y) = origin
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y + baseline), bg_color, -1)
    cv2.putText(frame, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_stats_panel(frame, total_faces: int, mask_count: int, fps: float):
    """Draw an informational HUD in the top-left corner."""
    no_mask_count = total_faces - mask_count
    panel_h, panel_w = 110, 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1)
    cv2.putText(frame, f"Faces detected: {total_faces}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1)
    cv2.putText(frame, f"With mask   : {mask_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MASK, 1)
    cv2.putText(frame, f"Without mask: {no_mask_count}", (10, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_NOMASK, 1)


# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Load models ────────────────────────────────────────────────────────────
    print("[INFO] Loading face detector ...")
    face_net = load_face_detector(args["face"])

    if not os.path.exists(args["model"]):
        print(f"[ERROR] Mask detector model not found: {args['model']}")
        print("  Run 'python train_mask_detector.py' first to train the model.")
        sys.exit(1)

    print("[INFO] Loading mask detector model ...")
    mask_net = load_model(args["model"])

    # ── Open camera ────────────────────────────────────────────────────────────
    print(f"[INFO] Opening camera {args['camera']} ...")
    cap = cv2.VideoCapture(args["camera"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args["height"])

    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check --camera index.")
        sys.exit(1)

    print("[INFO] Press 'q' to quit | 's' to save screenshot")
    prev_time = time.time()
    screenshot_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame, skipping ...")
            continue

        # Flip horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Detect faces + predict mask
        locs, preds = detect_and_predict_mask(
            frame, face_net, mask_net, args["confidence"]
        )

        # ── Draw results ───────────────────────────────────────────────────────
        mask_count = 0
        for (box, pred) in zip(locs, preds):
            (sX, sY, eX, eY) = box
            with_mask_prob, without_mask_prob = pred

            if with_mask_prob > without_mask_prob:
                label  = "Mask"
                color  = COLOR_MASK
                prob   = with_mask_prob
                mask_count += 1
            else:
                label  = "No Mask"
                color  = COLOR_NOMASK
                prob   = without_mask_prob

            text = f"{label}: {prob * 100:.1f}%"

            # Bounding box
            cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
            # Label tag
            draw_label(frame, text, (sX, sY), color)

        # FPS
        curr_time = time.time()
        fps        = 1.0 / max(curr_time - prev_time, 1e-9)
        prev_time  = curr_time

        # HUD panel
        draw_stats_panel(frame, len(locs), mask_count, fps)

        # Controls hint
        cv2.putText(frame, "Q: Quit | S: Screenshot", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Face Mask Detector — Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"images/screenshot_{screenshot_idx:04d}.jpg"
            os.makedirs("images", exist_ok=True)
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved → {fname}")
            screenshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed.")


if __name__ == "__main__":
    main()
