# Holistic Detection and Severity Estimation of Eggplant (Brinjal) Leaf and Fruit Diseases

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)
![YOLO26](https://img.shields.io/badge/Backbone-YOLO26n--seg-00FFFF.svg)
![SAM](https://img.shields.io/badge/Refinement-SAM_2.1-green.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)

## 📌 Project Overview

This repository contains the core implementation for **Capstone Project Phase 3 (Project ID: 19)** focused on multi-disease detection, instance segmentation, and quantitative severity estimation of Eggplant (Brinjal) leaf and fruit diseases.

### 🌟 Core Novel Claim
**SAM-bootstrapped label-efficiency for multi-disease agricultural segmentation**: Demonstrating that Segment Anything Model (SAM 2.1) refined pseudo-labeling achieves baseline-equivalent segmentation performance while utilizing only a fraction of manual annotation effort.

---

## 👥 Project Team & Guidance

* **Project Guide**: Prof. Neha Sharma
* **Project ID / Batch**: 19
* **Team Members**:
  * **Ramkumar Babu Baragi** (`PES2UG23AM081`) — *Severity Estimation & Annotation Pipeline*
  * **Khushi B** (`PES2UG23AM051`) — *Classification & Preprocessing*
  * **Patil Srishanth** (`PES2UG23AM070`) — *Data Processing & Segmentation*
  * **Sumukh S** (`PES2UG23CS626`) — *Classification Module*

---

## 🏗️ System Architecture & Pipeline

The disease detection and severity estimation pipeline consists of:

```
[ Field Image Input ]
        │
        ▼
[ Preprocessing & LAB CLAHE Denoising ]
        │
        ▼
[ YOLO26n-seg Bounding Box & Lesion Detection ] ──(Low Confidence Box Prompts)──┐
        │                                                                     │
        ▼                                                                     ▼
[ Ground-Truth / Baseline Models ]                           [ SAM 2.1 Mask Refinement ]
        │                                                                     │
        └───────────────────────────────┬─────────────────────────────────────┘
                                        ▼
                   [ Rule-Based Severity Estimation Engine ]
               (Diseased Pixels / Total Plant Pixels × 100%)
                                        │
                                        ▼
                     [ Diagnostic & Severity Output ]
```

1. **Preprocessing & Filtering**: Image deduplication (MD5 hash), Laplacian blur filtering, Fast Non-Local Means denoising, LAB space CLAHE illumination normalization.
2. **Backbone Segmentation**: **YOLO26n-seg** instance segmentation models trained at 640×640 resolution across 100 epochs per target class.
3. **SAM 2.1 Bootstrapping**: Generating bounding box proposals from baseline YOLO models as prompts for SAM (`sam2_b.pt`) to produce refined high-quality pseudo-label masks.
4. **Severity Estimation Engine**: Calculating quantitative severity based on spatial lesion pixel ratios (Low / Medium / High grading).

---

## 🦠 Disease Scope & Baseline Results

The project targets 8 disease classes on eggplant leaves and fruits:

### Baseline Segmentation Performance (YOLO26n-seg, 100 Epochs, Imgsz 640)

| Class | Train/Val Split | Precision (M) | Recall (M) | mAP50 (M) | mAP50-95 (M) | Annotation Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fruit-borer** | 170 / 43 | **0.878** | **0.778** | **0.898** | **0.582** | 215 / 1,055 (20%) |
| **Fruit-rot** | 168 / 42 | 0.842 | 0.668 | 0.795 | 0.608 | 212 / 1,020 (21%) |
| **Wilt** | 147 / 36 | 0.760 | 0.545 | 0.642 | 0.432 | 183 / 361 (51%) |
| **Phytophthora-Blight** | 186 / ... | 0.681 | 0.639 | 0.642 | 0.376 | 186 / 423 (44%) |
| **Melon-thrips** | — | — | — | — | — | 132 / 929 (14%) |
| **Tobacco-Mosaic-Virus** | — | — | — | — | — | 0 / 406 (0%) |
| **Cercospora Leaf Spot** | — | — | — | — | — | *Scoped Out (Diffuse Lesions)* |
| **Phomopsis-blight** | — | — | — | — | — | *Scoped Out (Diffuse Lesions)* |

*Note: Cercospora Leaf Spot and Phomopsis-blight present diffuse, low-contrast speckling without distinct visual boundaries, making them unsuitable for SAM promptable segmentation.*

---

## 📂 Repository Contents

```
capstone/
├── annotate.py      # SAM-assisted interactive bounding-box to mask annotation tool
├── tuner.py         # Hyperparameter tuning and model configuration script
├── visual.py        # Segmentation mask & bounding box visualizer for qualitative evaluation
├── sam2_b.pt        # Segment Anything Model 2.1 checkpoint (tracked via Git LFS)
├── yolo26x.pt       # Pretrained YOLO26 backbone weights (tracked via Git LFS)
└── .gitignore       # Exclusion rules for local datasets and images
```

---

## ⚙️ Installation & Usage

### 1. Prerequisites
Ensure Python 3.10+ and standard PyTorch dependencies are installed:

```bash
pip install torch torchvision ultralytics opencv-python numpy matplotlib
```

For SAM 2 integration, install `segment-anything-2`:
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 2. Running Annotation & Visualization
- **Run SAM Annotation Tool**:
  ```bash
  python annotate.py
  ```
- **Visualize Segmentation Mask Predictions**:
  ```bash
  python visual.py
  ```
- **Run Model Tuning**:
  ```bash
  python tuner.py
  ```

---

## 📚 References

1. P. Nanthini et al., *"Brinjal Leaf Disease Detection Using Deep Learning,"* in Proc. IEEE ICOECA, 2024, pp. 1-6.
2. H. Sun et al., *"A novel efficient eggplant disease detection method with multi-scale learning and edge feature enhancement,"* Frontiers in Plant Science, 2025.
3. M. I. H. Siddiqui et al., *"Eggplant Disease Diagnosis Using a Robust Ensemble of Transfer Learning Architectures,"* in Proc. IEEE ECCE, 2025.
4. P. Gaur et al., *"Brinjal Leaf Pathology: An Improved Disease Classification in a Federated Learning-CNN Approach,"* IEEE, 2024.
5. X. Wang et al., *"A Multimodal Data Fusion and Embedding Attention Mechanism-Based Method for Eggplant Disease Detection,"* Plants, 2025.
6. Z. Xie et al., *"YOLOv5s-BiPCNeXt, a Lightweight Model for Detecting Disease in Eggplant Leaves,"* Plants, MDPI, 2024.
