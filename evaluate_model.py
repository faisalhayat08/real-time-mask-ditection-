"""
evaluate_model.py
-----------------
Evaluate the trained mask detector and generate detailed analysis plots.

Usage:
  python evaluate_model.py
  python evaluate_model.py --model model/mask_detector.h5 --dataset dataset
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

from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

from imutils import paths
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",   default="model/mask_detector.h5")
    ap.add_argument("-d", "--dataset", default="dataset")
    ap.add_argument("-o", "--output",  default="logs")
    ap.add_argument("--img-size",      type=int, default=224)
    return vars(ap.parse_args())


CLASSES = ["with_mask", "without_mask"]


def load_dataset(dataset_path, img_size):
    data, labels = [], []
    for cls in CLASSES:
        cls_path = os.path.join(dataset_path, cls)
        for img_path in tqdm(list(paths.list_images(cls_path)), desc=f"  {cls}"):
            try:
                img = load_img(img_path, target_size=(img_size, img_size))
                arr = img_to_array(img)
                arr = preprocess_input(arr)
                data.append(arr)
                labels.append(cls)
            except Exception:
                pass
    return np.array(data, dtype="float32"), np.array(labels)


def plot_roc_curve(y_true_bin, y_scores, output_dir):
    fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="royalblue", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Mask Detector", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] ROC curve saved → {path}  (AUC = {roc_auc:.4f})")


def plot_precision_recall(y_true_bin, y_scores, output_dir):
    precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="tomato", lw=2,
             label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] PR curve saved → {path}")


def plot_sample_predictions(X_val, y_true, y_pred, y_probs, output_dir, n=16):
    """Show a grid of sample predictions (correct + incorrect)."""
    indices = np.random.choice(len(X_val), min(n, len(X_val)), replace=False)
    nrows   = int(np.ceil(np.sqrt(n)))
    ncols   = nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")

    class_names = {0: "With Mask", 1: "No Mask"}

    for ax, idx in zip(axes.flatten(), indices):
        # Reverse MobileNetV2 preprocessing for display
        img = X_val[idx].copy()
        img = (img + 1.0) * 127.5
        img = np.clip(img, 0, 255).astype("uint8")

        ax.imshow(img)
        pred_label = class_names[y_pred[idx]]
        true_label = class_names[y_true[idx]]
        prob       = y_probs[idx].max() * 100

        color = "green" if y_pred[idx] == y_true[idx] else "red"
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}\n{prob:.1f}%",
                     fontsize=8, color=color)
        ax.axis("off")

    # Hide unused axes
    for ax in axes.flatten()[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Sample predictions saved → {path}")


def main():
    args = parse_args()
    os.makedirs(args["output"], exist_ok=True)

    print("[INFO] Loading dataset ...")
    data, labels_raw = load_dataset(args["dataset"], args["img_size"])

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels_raw)
    labels = to_categorical(labels, num_classes=2)

    _, X_val, _, y_val = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42
    )

    print(f"[INFO] Loading model: {args['model']}")
    model = load_model(args["model"])

    print("[INFO] Running predictions ...")
    y_probs = model.predict(X_val, batch_size=32, verbose=1)
    y_pred  = np.argmax(y_probs, axis=1)
    y_true  = np.argmax(y_val,   axis=1)

    # Classification report
    target_names = lb.classes_
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n" + report)

    # Binary scores for ROC (probability of "without_mask" = class 1)
    y_scores    = y_probs[:, 1]
    y_true_bin  = y_true

    # Plot ROC + PR curves
    plot_roc_curve(y_true_bin, y_scores, args["output"])
    plot_precision_recall(y_true_bin, y_scores, args["output"])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix", fontweight="bold")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(args["output"], "confusion_matrix_eval.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {cm_path}")

    # Sample predictions
    plot_sample_predictions(X_val, y_true, y_pred, y_probs, args["output"])

    # Model summary
    print("\n[INFO] Model summary:")
    model.summary()

    print("\n✅  Evaluation complete. All plots saved to:", args["output"])


if __name__ == "__main__":
    main()
