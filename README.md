# BioFovea-YOLO: A Bio-Inspired Multi-Scale Detector for Tiny Bubble Detection

This repository contains the official implementation of the paper:  
**"BioFovea-YOLO: A Bio-Inspired Multi-Scale Detector for Tiny Bubble Detection and Physical Consistency Verification"**

**BioFovea-YOLO** is a lightweight, real-time object detector designed specifically for dense, tiny bubbles in subcooled flow boiling. It integrates a high-resolution **P2 detection head** and **SE (Squeeze-and-Excitation)** channel attention modules into the YOLOv8 architecture to effectively suppress background noise and enhance small object recall.

![BioFovea-YOLO Architecture](assets/architecture.png)
*(Note: Please upload your architecture figure to an `assets` folder)*

## 🚀 Key Features

*   **P2 Detection Head**: Introduces a high-resolution feature map (stride 4) to detect tiny bubbles (< 8x8 pixels) that are typically lost in standard detectors.
*   **Bionic Attention Modulation**: Embeds SE modules in the backbone to filter metallic reflection noise at the early feature extraction stage.
*   **Dual-Mode Strategy**:
    *   **Standard Mode**: Real-time detection (~60 FPS) for online monitoring.
    *   **High-Precision Mode**: Integrated **SAHI (Slicing Aided Hyper Inference)** for offline analysis, achieving **0.95+ Recall**.
*   **Physical Consistency**: Verified against thermodynamic parameters (void fraction, Sauter mean diameter).

## 📊 Performance

| Model | mAP@0.5 | Recall | FPS (RTX 5060 Ti) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **BioFovea-YOLO (Standard)** | **0.875** | **0.821** | **~60** | Best balance for real-time applications |
| BioFovea-YOLO (High-Precision) | - | **0.954** | ~1.5 | SAHI-assisted mode for offline analysis |
| YOLOv8s (Baseline) | 0.820 | 0.749 | ~75 | Standard baseline with lower recall |
| RT-DETR-l | 0.857 | 0.810 | ~35 | High computational cost |

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/linsoullqs-bit/BioFovea-YOLO.git
    cd BioFovea-YOLO
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Required libraries: `ultralytics`, `torch`, `sahi`, `opencv-python`, `numpy`, `pandas`.*

## 📂 Data Preparation

Due to copyright restrictions, the dataset is not included in this repository.
*   **Main Dataset**: Publicly available from [Zhou et al. (2024)](https://doi.org/10.1016/j.ijheatmasstransfer.2023.125028).
*   **BubbleBench**: Available from [Cai et al. (2025)](https://github.com/BubbleBench).

Please organize your data as follows:
```
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## 🏃 Usage

### 1. Training (Reproduce BioFovea-YOLO)

To train the BioFovea-YOLO (P2-SE) model:

```bash
# Train using the custom configuration
python train_p2_se.py
```

Or using the standard YOLO command with our config:

```bash
yolo detect train data=data.yaml model=models/yolov8s-p2-se.yaml epochs=300 ensure_reproducibility=True
```

### 2. Inference (Standard Mode)

To run detection on images or videos:

```bash
python detect.py --source ./sample_video.mp4 --weights weights/best.pt --conf 0.5
```

### 3. High-Precision Mode (SAHI)

To run the SAHI-assisted inference for maximum recall (offline mode):

```bash
python measure_sahi_speed.py
# Or use the SAHI CLI directly:
# sahi predict --model_path weights/best.pt --source image_dir/ --slice_height 320 --slice_width 320 --overlap_height_ratio 0.3 --overlap_width_ratio 0.3
```

## 📝 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{BioFoveaYOLO2026,
  title={BioFovea-YOLO: A Bio-Inspired Multi-Scale Detector for Tiny Bubble Detection and Physical Consistency Verification},
  author={[Your Name] and [Co-authors]},
  journal={Chemical Engineering Journal},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
