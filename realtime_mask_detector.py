"""
realtime_mask_detector.py
═══════════════════════════════════════════════════════════════════════════════
COMPLETE Real-Time Face Mask Detector — Single File, Auto Setup
Just run:  python realtime_mask_detector.py

What this script does automatically:
  1. Downloads the OpenCV DNN face detector (ResNet10 SSD)
  2. Downloads a tiny sample dataset from GitHub
  3. Trains MobileNetV2 mask classifier (fast, ~3 min)
  4. Launches your laptop camera with live detection overlay

Controls (while camera is running):
  Q       → Quit
  S       → Save screenshot
  SPACE   → Pause/Resume
  +/-     → Increase/Decrease detection sensitivity
═══════════════════════════════════════════════════════════════════════════════
"""

# ─── STDLIB ────────────────────────────────────────────────────────────────────
import os, sys, time, urllib.request, zipfile, shutil, argparse, warnings, math
warnings.filterwarnings("ignore")

# ─── THIRD PARTY — checked at runtime ─────────────────────────────────────────
def _require(package, pip_name=None):
    """Attempt to import; print install hint if missing."""
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        name = pip_name or package
        print(f"\n[ERROR] Missing package: '{name}'")
        print(f"  Fix:  pip install {name}\n")
        sys.exit(1)

np   = _require("numpy")
cv2  = _require("cv2",  "opencv-python")
tf   = _require("tensorflow")
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     BatchNormalization, Input)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# ─── PATHS ─────────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
FACE_DIR      = os.path.join(ROOT, "face_detector")
MODEL_PATH    = os.path.join(ROOT, "model", "mask_detector.h5")
DATASET_DIR   = os.path.join(ROOT, "dataset")
SCREENSHOT_DIR= os.path.join(ROOT, "screenshots")
os.makedirs(os.path.join(ROOT, "model"),  exist_ok=True)
os.makedirs(SCREENSHOT_DIR,               exist_ok=True)

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
CLASSES       = ["with_mask", "without_mask"]

# BGR colors
C_GREEN   = (34, 197, 94)
C_RED     = (60, 60, 235)
C_YELLOW  = (0, 200, 255)
C_WHITE   = (240, 240, 240)
C_BLACK   = (10, 10, 10)
C_PANEL   = (18, 18, 28)
C_ACCENT  = (255, 165, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DOWNLOAD FACE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
def download_face_detector():
    os.makedirs(FACE_DIR, exist_ok=True)
    files = {
        "deploy.prototxt": (
            "https://raw.githubusercontent.com/opencv/opencv/master/"
            "samples/dnn/face_detector/deploy.prototxt"
        ),
        "res10_300x300_ssd_iter_140000.caffemodel": (
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
            "dnn_samples_face_detector_20170830/"
            "res10_300x300_ssd_iter_140000.caffemodel"
        ),
    }
    all_ok = True
    for fname, url in files.items():
        dest = os.path.join(FACE_DIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            continue
        print(f"  ⬇  Downloading {fname} …")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"     ✓ Done ({os.path.getsize(dest)//1024} KB)")
        except Exception as e:
            print(f"     ✗ Failed: {e}")
            all_ok = False
    return all_ok


def load_face_net():
    prototxt = os.path.join(FACE_DIR, "deploy.prototxt")
    weights  = os.path.join(FACE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    if not (os.path.exists(prototxt) and os.path.exists(weights)):
        print("\n[STEP 1] Downloading face detector …")
        if not download_face_detector():
            print("[ERROR] Could not download face detector. Check your internet.")
            sys.exit(1)
    return cv2.dnn.readNet(prototxt, weights)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DOWNLOAD DATASET (if needed)
# ══════════════════════════════════════════════════════════════════════════════
def dataset_ok():
    for cls in CLASSES:
        p = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(p) or len(os.listdir(p)) < 20:
            return False
    return True


def download_dataset():
    """Try to clone dataset from GitHub (prajnasb/observations)."""
    TMP = os.path.join(ROOT, "_tmp_dataset")
    print("  ⬇  Cloning dataset from GitHub …")
    ret = os.system(
        f'git clone --depth=1 https://github.com/prajnasb/observations.git "{TMP}" 2>&1'
    )
    if ret != 0:
        shutil.rmtree(TMP, ignore_errors=True)
        return False

    src_w  = os.path.join(TMP, "experiements", "data", "with_mask")
    src_nw = os.path.join(TMP, "experiements", "data", "without_mask")
    if not (os.path.exists(src_w) and os.path.exists(src_nw)):
        shutil.rmtree(TMP, ignore_errors=True)
        return False

    os.makedirs(os.path.join(DATASET_DIR, "with_mask"),    exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "without_mask"), exist_ok=True)
    shutil.copytree(src_w,  os.path.join(DATASET_DIR, "with_mask"),    dirs_exist_ok=True)
    shutil.copytree(src_nw, os.path.join(DATASET_DIR, "without_mask"), dirs_exist_ok=True)
    shutil.rmtree(TMP, ignore_errors=True)
    return True


def build_synthetic_dataset(n=150):
    """Generate synthetic face images when no real dataset available."""
    from PIL import Image, ImageDraw, ImageFilter
    print(f"  ✎  Generating {n} synthetic images per class …")
    import random

    skin_tones = [(255,220,185),(240,195,160),(200,150,110),(160,110,80),(100,70,50)]

    def make_face(with_mask=True, size=128):
        skin = random.choice(skin_tones)
        bg   = (random.randint(50,200), random.randint(50,200), random.randint(50,200))
        img  = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        # Face oval
        cx, cy = size//2, size//2
        fw, fh = int(size*0.65), int(size*0.78)
        draw.ellipse([cx-fw//2, cy-fh//2, cx+fw//2, cy+fh//2], fill=skin)

        # Eyes
        ew = int(size*0.08)
        for ex in [cx - fw//5, cx + fw//5]:
            draw.ellipse([ex-ew, cy-fh//8-ew, ex+ew, cy-fh//8+ew],
                         fill=(random.randint(30,80), random.randint(20,60), 20))
            # Highlight
            draw.ellipse([ex+ew//3, cy-fh//8-ew+2, ex+ew//3+3, cy-fh//8-ew+5], fill=(255,255,255))

        if with_mask:
            mask_col = (
                random.randint(50,200), random.randint(50,200), random.randint(50,200)
            )
            pts = [
                (cx - fw//2 + 4, cy),
                (cx - fw//2,     cy + fh//3),
                (cx + fw//2,     cy + fh//3),
                (cx + fw//2 - 4, cy),
                (cx,             cy - fh//12),
            ]
            draw.polygon(pts, fill=mask_col)
            # Mask straps
            draw.line([(cx-fw//2+4, cy), (cx-fw//2-4, cy-fh//4)],
                      fill=mask_col, width=3)
            draw.line([(cx+fw//2-4, cy), (cx+fw//2+4, cy-fh//4)],
                      fill=mask_col, width=3)
        else:
            nose_col = tuple(max(0, c-30) for c in skin)
            draw.polygon([(cx, cy-5), (cx-6, cy+12), (cx+6, cy+12)], fill=nose_col)
            mouth_y = cy + fh//5
            draw.arc([cx-fw//6, mouth_y-8, cx+fw//6, mouth_y+8],
                     start=0, end=180, fill=(160,60,60), width=3)

        # Noise + blur for realism
        arr = np.array(img)
        arr = np.clip(arr + np.random.randint(-25, 25, arr.shape), 0, 255).astype("uint8")
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=random.uniform(0,0.8)))
        return img

    for cls, mask_flag in [("with_mask", True), ("without_mask", False)]:
        d = os.path.join(DATASET_DIR, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            make_face(with_mask=mask_flag).save(os.path.join(d, f"img_{i:04d}.png"))

    print(f"  ✓ Synthetic dataset ready ({n} per class)")


def ensure_dataset():
    if dataset_ok():
        counts = {c: len(os.listdir(os.path.join(DATASET_DIR, c))) for c in CLASSES}
        print(f"  ✓ Dataset already exists: {counts}")
        return

    print("\n[STEP 2] Preparing dataset …")
    if not download_dataset():
        print("  ⚠  Git clone failed. Generating synthetic dataset …")
        build_synthetic_dataset(n=200)
    else:
        counts = {c: len(os.listdir(os.path.join(DATASET_DIR, c))) for c in CLASSES}
        print(f"  ✓ Dataset ready: {counts}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_images(img_size=IMG_SIZE):
    data, labels = [], []
    for cls in CLASSES:
        cls_path = os.path.join(DATASET_DIR, cls)
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            try:
                img = load_img(fpath, target_size=(img_size, img_size))
                arr = img_to_array(img)
                arr = preprocess_input(arr)
                data.append(arr)
                labels.append(cls)
            except Exception:
                pass
    return np.array(data, dtype="float32"), np.array(labels)


def build_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(2, activation="softmax")(x)
    return Model(inputs=base.input, outputs=x)


def train_model(epochs=20, batch=32):
    print("\n[STEP 3] Training mask detector …")
    print("         (MobileNetV2 + Transfer Learning)")

    data, labels_raw = load_images()
    print(f"  Loaded {len(data)} images")

    lb     = LabelBinarizer()
    labels = lb.fit_transform(labels_raw)
    labels = to_categorical(labels, num_classes=2)

    X_tr, X_val, y_tr, y_val = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )

    aug = ImageDataGenerator(
        rotation_range=25, zoom_range=0.2,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.15, horizontal_flip=True,
        fill_mode="nearest"
    )

    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=0),
        EarlyStopping(monitor="val_loss", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    print(f"  Training for up to {epochs} epochs …\n")
    model.fit(
        aug.flow(X_tr, y_tr, batch_size=batch),
        steps_per_epoch=max(1, len(X_tr) // batch),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    print(f"\n  ✓ Model saved → {MODEL_PATH}")
    return model


def ensure_model():
    if os.path.exists(MODEL_PATH):
        print(f"\n  ✓ Model found: {MODEL_PATH}")
        return load_model(MODEL_PATH)
    return train_model()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — INFERENCE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def detect_faces_and_masks(frame, face_net, mask_net, min_conf):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    dets = face_net.forward()

    faces, locs = [], []
    for i in range(dets.shape[2]):
        if dets[0, 0, i, 2] < min_conf:
            continue
        box   = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        sX, sY, eX, eY = box.astype("int")
        sX, sY = max(0, sX), max(0, sY)
        eX, eY = min(w-1, eX), min(h-1, eY)

        roi = frame[sY:eY, sX:eX]
        if roi.size == 0:
            continue

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi = preprocess_input(img_to_array(roi))
        faces.append(roi)
        locs.append((sX, sY, eX, eY))

    preds = mask_net.predict(np.array(faces, dtype="float32"), verbose=0) if faces else []
    return locs, preds


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING UTILITIES  (professional HUD)
# ══════════════════════════════════════════════════════════════════════════════
def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=2, filled=False):
    x1, y1 = pt1; x2, y2 = pt2
    if filled:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius),(x2-radius, y1+radius),
                        (x1+radius, y2-radius),(x2-radius, y2-radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        for (p1, p2) in [((x1+radius,y1),(x2-radius,y1)),
                          ((x1+radius,y2),(x2-radius,y2)),
                          ((x1,y1+radius),(x1,y2-radius)),
                          ((x2,y1+radius),(x2,y2-radius))]:
            cv2.line(img, p1, p2, color, thickness)
        for (cx,cy,a1,a2) in [(x1+radius,y1+radius,180,270),
                                (x2-radius,y1+radius,270,360),
                                (x1+radius,y2-radius, 90,180),
                                (x2-radius,y2-radius,  0, 90)]:
            cv2.ellipse(img,(cx,cy),(radius,radius),0,a1,a2,color,thickness)


def draw_face_box(frame, box, label, prob, color, anim_phase=0.0):
    sX, sY, eX, eY = box
    bw = eX - sX
    bh = eY - sY

    # Animated corner brackets
    seg = min(bw, bh) // 4
    ph  = int(abs(math.sin(anim_phase)) * 8)

    for (px, py, dx, dy) in [
        (sX, sY, 1, 1), (eX, sY, -1, 1),
        (sX, eY, 1,-1), (eX, eY, -1,-1)
    ]:
        cv2.line(frame, (px, py), (px + dx*(seg+ph), py), color, 3)
        cv2.line(frame, (px, py), (px, py + dy*(seg+ph)), color, 3)
        # Dot at corner
        cv2.circle(frame, (px, py), 4, color, -1)

    # Filled label pill
    icon  = "✓" if label == "MASK" else "✗"
    text  = f"  {label}  {prob*100:.0f}%"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad   = 8
    lx, ly = sX, max(sY - th - pad*2, 4)

    overlay = frame.copy()
    cv2.rectangle(overlay, (lx - pad//2, ly - pad//2),
                  (lx + tw + pad, ly + th + pad), color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, text, (lx + 2, ly + th),
                font, scale, C_WHITE, thick, cv2.LINE_AA)

    # Confidence bar under face
    bar_y  = eY + 8
    bar_x1 = sX
    bar_x2 = eX
    bar_h  = 6
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y+bar_h), (60,60,60), -1)
    filled = int((bar_x2 - bar_x1) * prob)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1+filled, bar_y+bar_h), color, -1)


def draw_hud(frame, total_faces, mask_n, no_mask_n, fps, conf_thresh,
             paused, screenshot_count, frame_count):
    h, w = frame.shape[:2]

    # ── Top-left panel ────────────────────────────────────────────────────────
    panel_w, panel_h = 280, 170
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    # Panel border
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (80, 80, 100), 1)

    def hud_text(txt, y, color=C_WHITE, scale=0.55, thick=1):
        cv2.putText(frame, txt, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    hud_text("FACE MASK DETECTOR", 22, C_ACCENT, 0.6, 2)
    cv2.line(frame, (12, 28), (panel_w-12, 28), (80, 80, 100), 1)

    fps_color = C_GREEN if fps > 20 else (C_YELLOW if fps > 10 else C_RED)
    hud_text(f"FPS      : {fps:>5.1f}", 52,  fps_color)
    hud_text(f"Faces    : {total_faces:>2}",       76,  C_WHITE)
    hud_text(f"Mask     : {mask_n:>2}",            100, C_GREEN  if mask_n    else C_WHITE)
    hud_text(f"No Mask  : {no_mask_n:>2}",         124, C_RED    if no_mask_n else C_WHITE)
    hud_text(f"Conf Thr : {conf_thresh:.2f}  [+/-]",148, (180,180,180))

    # ── Status badge (top-right) ───────────────────────────────────────────────
    if paused:
        badge_txt = "  PAUSED  "
        badge_col = C_YELLOW
    elif no_mask_n > 0:
        badge_txt = " ! NO MASK DETECTED ! "
        badge_col = C_RED
    else:
        badge_txt = "  ALL CLEAR  "
        badge_col = C_GREEN

    (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    bx = w - bw - 30
    by = 14
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (bx-8, by-4), (bx+bw+8, by+bh+8), badge_col, -1)
    cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, badge_txt, (bx, by+bh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (bx-8, by-4), (bx+bw+8, by+bh+8), badge_col, 1)

    # ── Bottom controls bar ────────────────────────────────────────────────────
    bar_y = h - 28
    cv2.rectangle(frame, (0, bar_y), (w, h), C_PANEL, -1)
    controls = " Q: Quit    S: Screenshot    SPACE: Pause    +/-: Sensitivity "
    cv2.putText(frame, controls, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,180), 1, cv2.LINE_AA)

    if screenshot_count > 0:
        info = f"Screenshot #{screenshot_count} saved!"
        cv2.putText(frame, info, (w - 220, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GREEN, 1, cv2.LINE_AA)

    # ── Scanline texture overlay (subtle) ────────────────────────────────────
    if frame_count % 2 == 0:
        for y_line in range(0, h, 4):
            cv2.line(frame, (0, y_line), (w, y_line), (0, 0, 0), 1)
        frame_region = frame.copy()
        cv2.addWeighted(frame_region, 0.92, frame, 0.08, 0, frame)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CAMERA LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_camera(face_net, mask_net, camera_idx=0, init_conf=0.5):
    print("\n[STEP 4] Launching camera …")
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        # Try index 1 (external webcam on some laptops)
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera. Check connection or try --camera 1")
            sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✓ Camera opened: {actual_w}×{actual_h}")
    print("  Controls: Q=Quit  S=Screenshot  SPACE=Pause  +/-=Sensitivity\n")

    conf_thresh   = init_conf
    paused        = False
    screenshot_n  = 0
    frame_count   = 0
    anim_phase    = 0.0
    show_ss_frames= 0

    # FPS tracking
    fps_times = []
    fps = 0.0

    while True:
        t_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame drop, retrying …")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)  # Mirror (selfie mode)
        frame_count += 1
        anim_phase  += 0.12

        if not paused:
            locs, preds = detect_faces_and_masks(frame, face_net, mask_net, conf_thresh)
        else:
            locs, preds = [], []

        # ── Draw faces ─────────────────────────────────────────────────────────
        mask_n, no_mask_n = 0, 0
        for (box, pred) in zip(locs, preds):
            wm_prob, nm_prob = pred
            if wm_prob >= nm_prob:
                label, color, prob = "MASK",    C_GREEN, float(wm_prob)
                mask_n += 1
            else:
                label, color, prob = "NO MASK", C_RED,   float(nm_prob)
                no_mask_n += 1
            draw_face_box(frame, box, label, prob, color, anim_phase)

        # ── HUD ────────────────────────────────────────────────────────────────
        draw_hud(frame, len(locs), mask_n, no_mask_n,
                 fps, conf_thresh, paused,
                 screenshot_n if show_ss_frames > 0 else 0,
                 frame_count)
        if show_ss_frames > 0:
            show_ss_frames -= 1

        cv2.imshow("Face Mask Detector — Live Camera", frame)

        # ── FPS ────────────────────────────────────────────────────────────────
        fps_times.append(time.time() - t_start)
        if len(fps_times) > 30:
            fps_times.pop(0)
        fps = 1.0 / (sum(fps_times) / len(fps_times) + 1e-9)

        # ── Keys ───────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:        # Q / Esc → quit
            break
        elif key == ord(" "):                   # Space → pause
            paused = not paused
            print(f"  {'[PAUSED]' if paused else '[RESUMED]'}")
        elif key == ord("s") or key == ord("S"):# S → screenshot
            screenshot_n += 1
            fname = os.path.join(SCREENSHOT_DIR,
                                 f"mask_detect_{time.strftime('%H%M%S')}.jpg")
            cv2.imwrite(fname, frame)
            print(f"  📸 Screenshot saved → {fname}")
            show_ss_frames = 90
        elif key == ord("+") or key == ord("="):# + → raise threshold
            conf_thresh = min(0.95, conf_thresh + 0.05)
            print(f"  Confidence threshold: {conf_thresh:.2f}")
        elif key == ord("-"):                   # - → lower threshold
            conf_thresh = max(0.1,  conf_thresh - 0.05)
            print(f"  Confidence threshold: {conf_thresh:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅  Session ended.")
    if screenshot_n:
        print(f"   {screenshot_n} screenshot(s) saved in: {SCREENSHOT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global MODEL_PATH
    ap = argparse.ArgumentParser(
        description="Real-Time Face Mask Detector — fully auto-setup"
    )
    ap.add_argument("--camera",      type=int,   default=0,
                    help="Camera index (0=built-in, 1=external)")
    ap.add_argument("--model",       default=MODEL_PATH,
                    help="Path to pre-trained .h5 model (skip training)")
    ap.add_argument("--confidence",  type=float, default=0.5,
                    help="Face detection threshold (0.1–0.95)")
    ap.add_argument("--retrain",     action="store_true",
                    help="Force re-train even if model exists")
    ap.add_argument("--epochs",      type=int,   default=20,
                    help="Training epochs (default 20)")
    args = ap.parse_args()

    print("═" * 60)
    print("  FACE MASK DETECTOR — Auto Setup & Run")
    print("═" * 60)
    print(f"  TensorFlow  : {tf.__version__}")
    print(f"  OpenCV      : {cv2.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPU         : {gpus if gpus else 'None (using CPU)'}")
    print("─" * 60)

    # 1. Face detector
    print("\n[STEP 1] Face detector …")
    face_net = load_face_net()
    print("  ✓ Face detector ready (ResNet10 SSD)")

    # 2. Dataset
    ensure_dataset()

    # 3. Mask classifier
    
    MODEL_PATH = args.model
    if args.retrain and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    mask_net = ensure_model()
    print("  ✓ Mask classifier ready (MobileNetV2)")

    # 4. Run camera
    run_camera(face_net, mask_net,
               camera_idx=args.camera,
               init_conf=args.confidence)


if __name__ == "__main__":
    main()
