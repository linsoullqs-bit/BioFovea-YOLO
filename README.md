# BioFovea-YOLO: A Bio-Inspired Multi-Scale Detector for Tiny Bubble Detection

This repository contains the official implementation of the paper:  
**"BioFovea-YOLO: A Bio-Inspired Multi-Scale Detector for Tiny Bubble Detection and Physical Consistency Verification"**

**BioFovea-YOLO** is a lightweight, real-time object detector designed specifically for dense, tiny bubbles in subcooled flow boiling. It integrates a high-resolution **P2 detection head** and **SE (Squeeze-and-Excitation)** channel attention modules into the YOLOv8 architecture to effectively suppress background noise and enhance small object recall.

![Detection Result](assets/val_batch0_pred.jpg)
*Figure 1: Detection results on the validation set. Left: Ground Truth; Right: BioFovea-YOLO Prediction.*

## 📂 Repository Structure

```
BioFovea-YOLO/
├── assets/                # Visualization results (Train/Val batches)
├── models/
│   └── yolov8s-p2-se.yaml # Core Model Configuration (BioFovea-YOLO)
├── weights/
│   └── yolov8s_p2_se_best.pt # Pre-trained Model Weights
├── train_all.py           # Training Interface Script
├── train_p2_se.py         # Specific Training Script for BioFovea-YOLO
└── README.md              # Project Documentation
```

## 🚀 Key Features

*   **P2 Detection Head**: Introduces a high-resolution feature map (stride 4) to detect tiny bubbles (< 8x8 pixels) that are typically lost in standard detectors.
*   **Bionic Attention Modulation**: Embeds SE modules in the backbone to filter metallic reflection noise at the early feature extraction stage.
*   **Physical Consistency**: Verified against thermodynamic parameters (void fraction, Sauter mean diameter).

## 📊 Performance

| Model | mAP@0.5 | Recall | FPS (RTX 5060 Ti) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **BioFovea-YOLO** | **0.875** | **0.821** | **~60** | Best balance for real-time applications |
| YOLOv8s (Baseline) | 0.820 | 0.749 | ~75 | Standard baseline with lower recall |
| RT-DETR-l | 0.857 | 0.810 | ~35 | High computational cost |

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/linsoullqs-bit/BioFovea-YOLO.git
    cd BioFovea-YOLO
    ```

2.  **Install dependencies:**
    This project relies on `ultralytics` (YOLOv8) and `torch`.
    ```bash
    pip install ultralytics torch torchvision opencv-python sahi
    ```

## 🏃 Usage

### 1. Data Preparation

Due to copyright restrictions, the dataset is not included in this repository.
*   **Main Dataset**: Publicly available from [Zhou et al. (2024)](https://doi.org/10.1016/j.ijheatmasstransfer.2023.125028).

Please organize your data in a `data/` folder (or update the data path in the training scripts) structure as follows:
```
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 2. Training

To train the **BioFovea-YOLO (P2-SE)** model using the provided script:

```bash
# Direct training using the dedicated script
python train_p2_se.py
```

This script is pre-configured to load `models/yolov8s-p2-se.yaml`.

### 3. Inference / Validation

To validate the model using the pre-trained weights included in `weights/`:

```bash
yolo detect val model=weights/yolov8s_p2_se_best.pt data=your_data.yaml
```

## 📝 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{BioFoveaYOLO2026,
  title={BioFovea-YOLO: A Bio-Inspired Multi-Scale Detector for Tiny Bubble Detection and Physical Consistency Verification},
  author={[Author List in Paper]},
  journal={Chemical Engineering Journal},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License.

