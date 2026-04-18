# 😷 Face Mask Detection — Deep Learning Project

A complete real-time face mask detection system built with **MobileNetV2** (Transfer Learning) + **OpenCV** DNN face detector.

---

## 📁 Project Structure

```
face_mask_detection/
│
├── dataset/
│   ├── with_mask/          ← Images of people wearing masks
│   └── without_mask/       ← Images of people without masks
│
├── model/
│   └── mask_detector.h5    ← Saved trained model (generated)
│
├── face_detector/
│   ├── deploy.prototxt                          ← Auto-downloaded
│   └── res10_300x300_ssd_iter_140000.caffemodel ← Auto-downloaded
│
├── logs/
│   ├── training_plot.png         ← Accuracy/loss curves
│   ├── confusion_matrix.png      ← Confusion matrix
│   ├── roc_curve.png             ← ROC curve
│   ├── pr_curve.png              ← Precision-Recall curve
│   ├── sample_predictions.png    ← Sample prediction grid
│   └── classification_report.txt ← Detailed metrics
│
├── images/                        ← Test images / screenshots
│
├── .vscode/
│   ├── launch.json               ← Run configs (F5 to run)
│   └── settings.json             ← Python settings
│
├── prepare_dataset.py            ← Step 1: Get dataset
├── train_mask_detector.py        ← Step 2: Train model
├── detect_mask_video.py          ← Step 3a: Webcam detection
├── detect_mask_image.py          ← Step 3b: Image detection
├── evaluate_model.py             ← Step 4: Detailed evaluation
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

---

## ⚙️ Setup & Installation

### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> 💡 **GPU Support**: Install `tensorflow-gpu` for faster training if you have an NVIDIA GPU.

---

## 🗂️ Dataset

### Recommended Dataset (Best Results)
Download from Kaggle: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

Place images as:
```
dataset/
  with_mask/       ← ~3725 images
  without_mask/    ← ~3828 images
```

### Auto-Download (No account needed)
```bash
python prepare_dataset.py
```
This will:
1. Try to clone a GitHub dataset automatically
2. Fall back to a synthetic demo dataset if unavailable

---

## 🚀 Usage

### Step 1 — Prepare Dataset
```bash
python prepare_dataset.py
```

### Step 2 — Train the Model
```bash
# Default (20 epochs)
python train_mask_detector.py

# Custom settings
python train_mask_detector.py --epochs 50 --batch 32 --lr 0.0001
```

**Training arguments:**
| Argument | Default | Description |
|---|---|---|
| `--dataset` | `dataset` | Path to dataset folder |
| `--model` | `model/mask_detector.h5` | Where to save the model |
| `--epochs` | `20` | Number of training epochs |
| `--batch` | `32` | Batch size |
| `--lr` | `0.0001` | Initial learning rate |
| `--img-size` | `224` | Image input size |

### Step 3a — Real-Time Webcam Detection
```bash
python detect_mask_video.py
```
- Press `Q` to quit
- Press `S` to save a screenshot

### Step 3b — Detect in an Image
```bash
python detect_mask_image.py --image path/to/photo.jpg
python detect_mask_image.py --image photo.jpg --output result.jpg
```

### Step 4 — Evaluate the Model
```bash
python evaluate_model.py
```
Generates ROC curve, Precision-Recall curve, confusion matrix, and sample predictions.

---

## 🧠 Model Architecture

```
Input (224×224×3)
    │
MobileNetV2 (ImageNet pretrained, FROZEN)
    │  ~2.2M parameters
    │
GlobalAveragePooling2D
    │
Dense(128, ReLU)
    │
BatchNormalization
    │
Dropout(0.5)
    │
Dense(64, ReLU)
    │
Dropout(0.3)
    │
Dense(2, Softmax)
    │
Output: [with_mask_prob, without_mask_prob]
```

**Why MobileNetV2?**
- Lightweight (fast on CPU & mobile)
- Pre-trained on ImageNet → excellent feature extraction
- Depthwise separable convolutions → efficient

---

## 🏋️ Training Details

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Learning Rate | 1e-4 (ReduceLROnPlateau) |
| Data Augmentation | Rotation, Zoom, Shift, Flip |
| Early Stopping | patience=7 |
| Checkpointing | Best val_accuracy |

---

## 📊 Expected Results

With the full Kaggle dataset (~7500 images):
| Metric | Value |
|---|---|
| Validation Accuracy | ~98-99% |
| F1-Score | ~0.98 |
| AUC | ~0.99 |

---

## 🖥️ VS Code Integration

Press **F5** or go to **Run → Start Debugging** to launch any configuration:
- `1. Prepare Dataset`
- `2. Train Model (20 epochs)`
- `3. Train Model (50 epochs)`
- `4. Real-time Webcam Detection`
- `5. Detect Mask in Image`
- `6. Evaluate Model`

---

## 📈 Monitoring Training

TensorBoard logs are saved in `logs/tb_<timestamp>/`. View them:
```bash
tensorboard --logdir logs/
```
Then open [http://localhost:6006](http://localhost:6006) in your browser.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Camera not opening | Change `--camera 1` or `--camera 2` |
| Low accuracy | Use a bigger dataset (Kaggle), increase `--epochs` |
| OOM error | Reduce `--batch` to 16 or 8 |
| Face not detected | Lower `--confidence` to 0.3 |

---

## 📚 References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn)
- [Kaggle Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- [TensorFlow Keras API](https://keras.io/api/)

---

## 👨‍🎓 College Project Notes

This project demonstrates:
- **Transfer Learning** using a pre-trained CNN (MobileNetV2)
- **Object Detection** using OpenCV DNN module
- **Data Augmentation** for better generalization
- **Callbacks** (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- **Evaluation Metrics** (Accuracy, F1, ROC-AUC, Confusion Matrix)
- **Real-time inference** pipeline

---

*Built with TensorFlow 2.x, Keras, OpenCV, and Python 3.9+*
