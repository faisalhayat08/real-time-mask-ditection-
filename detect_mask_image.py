"""
detect_mask_image.py
---------------------
Detect face masks in a single image file.

Usage:
  python detect_mask_image.py --image path/to/image.jpg
  python detect_mask_image.py -i photo.jpg -o output.jpg
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# ─── ARGUMENT PARSER ───────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Image Face Mask Detection")
    ap.add_argument("-i", "--image",      required=True,
                    help="Path to input image")
    ap.add_argument("-o", "--output",     default=None,
                    help="Path to save output image (optional)")
    ap.add_argument("-f", "--face",       default="face_detector",
                    help="Path to face detector model directory")
    ap.add_argument("-m", "--model",      default="model/mask_detector.h5",
                    help="Path to trained mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="Minimum face detection confidence")
    return vars(ap.parse_args())


IMG_SIZE   = 224
COLOR_MASK   = (0, 200, 0)
COLOR_NOMASK = (0, 0, 220)
COLOR_BG     = (40, 40, 40)


def load_face_detector(face_dir: str):
    prototxt = os.path.join(face_dir, "deploy.prototxt")
    weights  = os.path.join(face_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt) or not os.path.exists(weights):
        _download_face_detector(face_dir)

    return cv2.dnn.readNet(prototxt, weights)


def _download_face_detector(face_dir: str):
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
            except Exception as e:
                print(f"  Download failed: {e}")
                sys.exit(1)


def process_image(image_path: str, face_net, mask_net, min_confidence: float):
    """Run face detection + mask classification on a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig = frame.copy()
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []
    faces, locs = [], []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue

        box   = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (sX, sY, eX, eY) = box.astype("int")
        sX, sY = max(0, sX), max(0, sY)
        eX, eY = min(w - 1, eX), min(h - 1, eY)

        face = frame[sY:eY, sX:eX]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((sX, sY, eX, eY))

    if not faces:
        print("[WARN] No faces detected in the image.")
        return frame, []

    preds = mask_net.predict(np.array(faces, dtype="float32"), verbose=0)

    for (box, pred) in zip(locs, preds):
        (sX, sY, eX, eY) = box
        with_mask_prob, without_mask_prob = pred

        if with_mask_prob > without_mask_prob:
            label, color, prob = "MASK", COLOR_MASK, with_mask_prob
        else:
            label, color, prob = "NO MASK", COLOR_NOMASK, without_mask_prob

        text = f"{label}: {prob * 100:.1f}%"

        # Draw thick bounding box
        cv2.rectangle(frame, (sX, sY), (eX, eY), color, 3)

        # Draw filled label background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (sX, sY - th - 12), (sX + tw + 8, sY), color, -1)
        cv2.putText(frame, text, (sX + 4, sY - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        results.append({
            "label":       label,
            "probability": float(prob),
            "box":         (sX, sY, eX, eY)
        })

    # Summary strip
    mask_n    = sum(1 for r in results if r["label"] == "MASK")
    no_mask_n = len(results) - mask_n
    summary   = f"Faces: {len(results)}  |  Mask: {mask_n}  |  No Mask: {no_mask_n}"
    (sw, sh), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (0, h - sh - 16), (sw + 16, h), COLOR_BG, -1)
    cv2.putText(frame, summary, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame, results


def main():
    args = parse_args()

    print("[INFO] Loading face detector ...")
    face_net = load_face_detector(args["face"])

    if not os.path.exists(args["model"]):
        print(f"[ERROR] Model not found: {args['model']}")
        print("  Run 'python train_mask_detector.py' first.")
        sys.exit(1)

    print("[INFO] Loading mask detector model ...")
    mask_net = load_model(args["model"])

    print(f"[INFO] Processing image: {args['image']}")
    output_frame, results = process_image(
        args["image"], face_net, mask_net, args["confidence"]
    )

    print(f"\n[RESULTS] Detected {len(results)} face(s):")
    for i, r in enumerate(results):
        print(f"  Face {i+1}: {r['label']} ({r['probability'] * 100:.2f}%)")

    # Save output
    if args["output"]:
        out_path = args["output"]
    else:
        base, ext = os.path.splitext(args["image"])
        out_path  = f"{base}_detected{ext or '.jpg'}"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, output_frame)
    print(f"\n[INFO] Result saved → {out_path}")

    # Display
    cv2.imshow("Face Mask Detection — Result", output_frame)
    print("[INFO] Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
