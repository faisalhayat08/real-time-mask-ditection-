"""
prepare_dataset.py
------------------
Downloads and prepares the Face Mask Dataset.
Dataset: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
Since Kaggle needs auth, this script creates a sample synthetic dataset
for testing and shows instructions for the real dataset.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
CLASSES = ["with_mask", "without_mask"]


def check_dataset():
    """Check if dataset already exists and is valid."""
    for cls in CLASSES:
        cls_path = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_path):
            return False
        if len(os.listdir(cls_path)) < 10:
            return False
    return True


def print_dataset_instructions():
    """Print instructions to download the real dataset."""
    print("\n" + "=" * 60)
    print("  DATASET SETUP INSTRUCTIONS")
    print("=" * 60)
    print("""
Option 1 — Kaggle Dataset (Recommended):
  1. Visit: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
  2. Download the ZIP file
  3. Extract and place images as:
       dataset/
         with_mask/       ← images of people WITH mask
         without_mask/    ← images of people WITHOUT mask

Option 2 — GitHub Dataset (No login needed):
  Run this in your terminal:
    git clone https://github.com/prajnasb/observations.git
  Then copy:
    observations/experiements/data/with_mask     → dataset/with_mask
    observations/experiements/data/without_mask  → dataset/without_mask

Option 3 — Use the synthetic demo dataset created by this script
  (Already done! You can train with it, but accuracy will be limited.)
  For best results, use Options 1 or 2.
""")
    print("=" * 60 + "\n")


def try_download_github_dataset():
    """Attempt to clone the GitHub dataset (requires git)."""
    print("[INFO] Attempting to download dataset from GitHub...")
    try:
        ret = os.system(
            "git clone --depth=1 https://github.com/prajnasb/observations.git _tmp_obs 2>&1"
        )
        if ret != 0:
            return False

        src_with = os.path.join("_tmp_obs", "experiements", "data", "with_mask")
        src_without = os.path.join("_tmp_obs", "experiements", "data", "without_mask")

        if os.path.exists(src_with) and os.path.exists(src_without):
            shutil.copytree(src_with, os.path.join(DATASET_DIR, "with_mask"), dirs_exist_ok=True)
            shutil.copytree(src_without, os.path.join(DATASET_DIR, "without_mask"), dirs_exist_ok=True)
            shutil.rmtree("_tmp_obs", ignore_errors=True)
            print("[INFO] Dataset downloaded successfully from GitHub!")
            return True
        else:
            shutil.rmtree("_tmp_obs", ignore_errors=True)
            return False
    except Exception as e:
        print(f"[WARN] GitHub download failed: {e}")
        return False


def create_synthetic_dataset():
    """Create a minimal synthetic dataset for testing the pipeline."""
    print("[INFO] Creating synthetic demo dataset...")
    try:
        import numpy as np
        from PIL import Image, ImageDraw

        os.makedirs(os.path.join(DATASET_DIR, "with_mask"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "without_mask"), exist_ok=True)

        def draw_face(with_mask=True):
            img = Image.new("RGB", (128, 128), color=(255, 220, 185))
            draw = ImageDraw.Draw(img)
            # Head
            draw.ellipse([20, 10, 108, 110], fill=(255, 220, 185), outline=(180, 140, 100), width=2)
            # Eyes
            draw.ellipse([35, 35, 50, 50], fill=(80, 50, 30))
            draw.ellipse([78, 35, 93, 50], fill=(80, 50, 30))

            if with_mask:
                # Blue mask covering lower half
                draw.rectangle([18, 65, 110, 108], fill=(70, 130, 200), outline=(40, 90, 150), width=2)
                draw.line([(18, 65), (64, 55), (110, 65)], fill=(40, 90, 150), width=2)
            else:
                # Nose + mouth
                draw.polygon([(58, 58), (70, 58), (64, 72)], fill=(230, 180, 160))
                draw.arc([45, 72, 83, 95], start=0, end=180, fill=(150, 80, 80), width=2)

            # Add noise for variety
            arr = np.array(img)
            noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        n_samples = 100  # 100 per class for demo
        for i in range(n_samples):
            img = draw_face(with_mask=True)
            img.save(os.path.join(DATASET_DIR, "with_mask", f"mask_{i:04d}.jpg"))

            img = draw_face(with_mask=False)
            img.save(os.path.join(DATASET_DIR, "without_mask", f"nomask_{i:04d}.jpg"))

        print(f"[INFO] Created {n_samples} synthetic images per class.")
        print("[WARN] Synthetic dataset is for pipeline testing only.")
        print("[WARN] Use a real dataset for meaningful accuracy.\n")
        return True

    except ImportError:
        print("[ERROR] Pillow not installed. Run: pip install Pillow")
        return False


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    if check_dataset():
        counts = {cls: len(os.listdir(os.path.join(DATASET_DIR, cls))) for cls in CLASSES}
        print(f"[INFO] Dataset already exists: {counts}")
        return

    # Try real dataset first
    if not try_download_github_dataset():
        print_dataset_instructions()
        create_synthetic_dataset()
    
    # Final count
    for cls in CLASSES:
        path = os.path.join(DATASET_DIR, cls)
        if os.path.exists(path):
            print(f"[INFO] {cls}: {len(os.listdir(path))} images")


if __name__ == "__main__":
    main()
