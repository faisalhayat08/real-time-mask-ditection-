"""
train_mask_detector.py
-----------------------
Trains a face mask detector using Transfer Learning with MobileNetV2.

Architecture:
  MobileNetV2 (ImageNet pretrained, frozen) → GlobalAvgPool → Dense(128)
  → Dropout(0.5) → Dense(2, softmax)

Usage:
  python train_mask_detector.py
  python train_mask_detector.py --dataset dataset --model model/mask_detector.h5 --epochs 20
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import (AveragePooling2D, Dropout, Flatten,
                                     Dense, Input, GlobalAveragePooling2D, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.utils import to_categorical

from imutils import paths
import cv2
from tqdm import tqdm


# ─── ARGUMENT PARSER ───────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Face Mask Detector Trainer")
    ap.add_argument("-d", "--dataset",  default="dataset",
                    help="Path to input dataset directory")
    ap.add_argument("-m", "--model",    default="model/mask_detector.h5",
                    help="Path to save trained model")
    ap.add_argument("-p", "--plot",     default="logs/training_plot.png",
                    help="Path to save accuracy/loss plot")
    ap.add_argument("-e", "--epochs",   type=int, default=20,
                    help="Number of training epochs")
    ap.add_argument("-b", "--batch",    type=int, default=32,
                    help="Batch size")
    ap.add_argument("-lr","--lr",       type=float, default=1e-4,
                    help="Initial learning rate")
    ap.add_argument("--img-size",       type=int, default=224,
                    help="Input image size (224 recommended for MobileNetV2)")
    return vars(ap.parse_args())


# ─── CONFIG ────────────────────────────────────────────────────────────────────
CLASSES = ["with_mask", "without_mask"]
COLORS  = {"with_mask": "green", "without_mask": "red"}


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def load_dataset(dataset_path: str, img_size: int):
    """Load images from dataset directory and return (data, labels)."""
    print(f"\n[INFO] Loading images from '{dataset_path}' ...")
    data, labels = [], []

    for cls in CLASSES:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.exists(cls_path):
            raise FileNotFoundError(
                f"Class folder not found: {cls_path}\n"
                f"Run 'python prepare_dataset.py' first."
            )
        img_paths = list(paths.list_images(cls_path))
        print(f"  {cls:20s} → {len(img_paths):5d} images")

        for img_path in tqdm(img_paths, desc=f"  Loading {cls}", leave=False):
            try:
                img   = load_img(img_path, target_size=(img_size, img_size))
                arr   = img_to_array(img)
                arr   = preprocess_input(arr)   # MobileNetV2 preprocessing
                data.append(arr)
                labels.append(cls)
            except Exception as e:
                print(f"  [WARN] Skipping {img_path}: {e}")

    data   = np.array(data,   dtype="float32")
    labels = np.array(labels)
    print(f"\n[INFO] Total images loaded: {len(data)}")
    return data, labels


def build_model(img_size: int, num_classes: int = 2) -> Model:
    """Build the mask detector model using MobileNetV2 as feature extractor."""
    # Load MobileNetV2 without top layers (pre-trained on ImageNet)
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(img_size, img_size, 3))
    )
    # Freeze base model layers
    base_model.trainable = False

    # Build custom classification head
    head = base_model.output
    head = GlobalAveragePooling2D()(head)
    head = Dense(128, activation="relu")(head)
    head = BatchNormalization()(head)
    head = Dropout(0.5)(head)
    head = Dense(64, activation="relu")(head)
    head = Dropout(0.3)(head)
    head = Dense(num_classes, activation="softmax")(head)

    model = Model(inputs=base_model.input, outputs=head)
    return model


def plot_training(history, output_path: str):
    """Plot and save training accuracy/loss curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Face Mask Detector — Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="royalblue", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    color="tomato",    lw=2, linestyle="--")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", color="royalblue", lw=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="tomato",    lw=2, linestyle="--")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training plot saved → {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path: str):
    """Plot and save confusion matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved → {output_path}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    IMG_SIZE   = args["img_size"]
    EPOCHS     = args["epochs"]
    BATCH_SIZE = args["batch"]
    INIT_LR    = args["lr"]
    MODEL_PATH = args["model"]
    PLOT_PATH  = args["plot"]

    # Ensure output directories exist
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_PATH)  or ".", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("  FACE MASK DETECTOR — TRAINING")
    print("=" * 60)
    print(f"  Image size  : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Learning rate: {INIT_LR}")
    print(f"  Model output: {MODEL_PATH}")
    print(f"  GPU available: {tf.config.list_physical_devices('GPU')}")
    print("=" * 60)

    # ── 1. Load & preprocess data ──────────────────────────────────────────────
    data, labels_raw = load_dataset(args["dataset"], IMG_SIZE)

    # One-hot encode labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels_raw)
    labels = to_categorical(labels, num_classes=2)

    # Train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42
    )
    print(f"\n[INFO] Train samples : {len(X_train)}")
    print(f"[INFO] Val samples   : {len(X_val)}")

    # ── 2. Data Augmentation ───────────────────────────────────────────────────
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # ── 3. Build model ─────────────────────────────────────────────────────────
    print("\n[INFO] Building model ...")
    model = build_model(IMG_SIZE, num_classes=len(CLASSES))

    # ── 4. Compile ─────────────────────────────────────────────────────────────
    opt = Adam(learning_rate=INIT_LR)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # ── 5. Callbacks ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=f"logs/tb_{ts}",
            histogram_freq=1
        )
    ]

    # ── 6. Train ───────────────────────────────────────────────────────────────
    print("\n[INFO] Training network ...")
    history = model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # ── 7. Evaluate ────────────────────────────────────────────────────────────
    print("\n[INFO] Evaluating network ...")
    pred_probs = model.predict(X_val, batch_size=BATCH_SIZE)
    y_pred     = np.argmax(pred_probs, axis=1)
    y_true     = np.argmax(y_val, axis=1)

    # Map numeric indices to class names (LabelBinarizer order)
    target_names = lb.classes_

    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n" + report)

    # Save report to file
    with open("logs/classification_report.txt", "w") as f:
        f.write(report)
    print("[INFO] Classification report saved → logs/classification_report.txt")

    # ── 8. Plots ───────────────────────────────────────────────────────────────
    plot_training(history, PLOT_PATH)
    plot_confusion_matrix(y_true, y_pred, "logs/confusion_matrix.png")

    # ── 9. Save final model ────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n[INFO] Model saved → {MODEL_PATH}")
    print("\n✅  Training complete! Run detection with:")
    print("    python detect_mask_video.py    (webcam)")
    print("    python detect_mask_image.py --image <path>  (image file)")


if __name__ == "__main__":
    main()
