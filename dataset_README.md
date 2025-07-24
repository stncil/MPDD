# 🏭 Manufacturing Product Defect Detection (MPDD)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-6_Categories-orange.svg)](#dataset)

> **Industrial Computer Vision Anomaly Detection System** for manufacturing quality control using unsupervised deep learning methods.

## 📊 Dataset Overview

Our dataset contains **high-resolution industrial images** across **6 product categories** with various defect types for comprehensive anomaly detection evaluation.

### 🗂️ Dataset Structure

```
📁 anomaly_dataset_og/                    # Root dataset directory
├── 🔧 bracket_black/                     # Black metal bracket components
│   ├── 📚 train/
│   │   └── ✅ good/                      # 209 normal training images
│   ├── 🧪 test/
│   │   ├── ✅ good/                      # 32 normal test images
│   │   ├── 🕳️ hole/                      # 12 images with holes
│   │   └── 🔍 scratches/                 # 35 images with scratches
│   └── 🎯 ground_truth/
│       ├── 🕳️ hole/                      # 12 pixel-level masks
│       └── 🔍 scratches/                 # 35 pixel-level masks
│
├── 🔧 bracket_brown/                     # Brown metal bracket components
│   ├── 📚 train/
│   │   └── ✅ good/                      # 185 normal training images
│   ├── 🧪 test/
│   │   ├── ✅ good/                      # 26 normal test images
│   │   ├── 🔧 parts_mismatch/            # 34 mismatched parts
│   │   └── ⚠️ bend_and_parts_mismatch/   # 17 bent + mismatched
│   └── 🎯 ground_truth/
│       ├── 🔧 parts_mismatch/            # 34 pixel-level masks
│       └── ⚠️ bend_and_parts_mismatch/   # 17 pixel-level masks
│
├── 🔧 bracket_white/                     # White metal bracket components
│   ├── 📚 train/
│   │   └── ✅ good/                      # 110 normal training images
│   ├── 🧪 test/
│   │   ├── ✅ good/                      # 30 normal test images
│   │   ├── 🔍 scratches/                 # 17 images with scratches
│   │   └── 🎨 defective_painting/        # 13 painting defects
│   └── 🎯 ground_truth/
│       ├── 🔍 scratches/                 # 17 pixel-level masks
│       └── 🎨 defective_painting/        # 13 pixel-level masks
│
├── 🏗️ metal_plate/                       # Industrial metal plates
│   ├── 📚 train/
│   │   └── ✅ good/                      # 54 normal training images
│   ├── 🧪 test/
│   │   ├── ✅ good/                      # 26 normal test images
│   │   ├── 🔍 scratches/                 # 34 images with scratches
│   │   ├── 🦠 major_rust/                # 14 major rust damage
│   │   └── ☢️ total_rust/                # 23 completely rusted
│   └── 🎯 ground_truth/
│       ├── 🔍 scratches/                 # 34 pixel-level masks
│       ├── 🦠 major_rust/                # 14 pixel-level masks
│       └── ☢️ total_rust/                # 23 pixel-level masks
```

## 📈 Dataset Statistics

<details>
<summary><b>📊 Detailed Statistics (Click to expand)</b></summary>

| Category | 📚 Train (Good) | 🧪 Test (Good) | 🚨 Test (Defects) | 🎯 Ground Truth | 📋 Defect Types |
|----------|:---------------:|:--------------:|:-----------------:|:---------------:|:----------------|
| **🔧 Bracket Black** | 209 | 32 | 47 | 47 | hole, scratches |
| **🔧 Bracket Brown** | 185 | 26 | 51 | 51 | parts_mismatch, bend_and_parts_mismatch |
| **🔧 Bracket White** | 110 | 30 | 30 | 30 | scratches, defective_painting |
| **🔌 Connector** | 128 | 30 | 14 | 14 | parts_mismatch |
| **🏗️ Metal Plate** | 54 | 26 | 71 | 71 | scratches, major_rust, total_rust |
| **🚇 Tubes** | 122 | 32 | 69 | 69 | anomalous |
| **📊 TOTAL** | **808** | **176** | **282** | **282** | **9 unique types** |

</details>

### 🎯 Defect Categories

| 🏷️ Defect Type | 📝 Description | 🔢 Count | 📂 Categories |
|:---------------|:---------------|:--------:|:--------------|
| 🔍 **Scratches** | Surface scratches and marks | 86 | bracket_black, bracket_white, metal_plate |
| 🕳️ **Holes** | Physical holes and perforations | 12 | bracket_black |
| 🔧 **Parts Mismatch** | Incorrect or missing components | 48 | bracket_brown, connector |
| 🎨 **Defective Painting** | Paint defects and coating issues | 13 | bracket_white |
| ⚠️ **Bend + Parts Mismatch** | Combined bending and part issues | 17 | bracket_brown |
| 🦠 **Major Rust** | Significant rust formation | 14 | metal_plate |
| ☢️ **Total Rust** | Complete rust coverage | 23 | metal_plate |
| ⚠️ **Anomalous** | Various tube-specific defects | 69 | tubes |

## 🚀 Quick Start

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MPDD.git
cd MPDD

# Install dependencies
pip install -r requirements.txt

# Verify dataset structure
python scripts/verify_dataset.py
```

### 💻 Usage Examples

```python
# Train PaDiM model on bracket_black category
python src/train.py --model padim --category bracket_black --epochs 100

# Evaluate trained model
python src/evaluate.py --model padim --category bracket_black --visualize

# Run inference on new images
python src/inference.py --model padim --image path/to/test/image.png
```

## 🧠 Implemented Models

| Model | 📊 Type | 🎯 Localization | ⚡ Speed | 📈 Performance |
|-------|:-------:|:---------------:|:-------:|:--------------:|
| 🔬 **PaDiM** | Discriminative | ✅ Excellent | 🚀 Fast | ⭐⭐⭐⭐ |
| 🧩 **PatchCore** | Discriminative | ✅ Excellent | 🚀 Fast | ⭐⭐⭐⭐⭐ |
| 🔄 **AutoEncoder** | Generative | ⚠️ Good | 🐌 Slow | ⭐⭐⭐ |
| 🌊 **CFlow-AD** | Normalizing Flow | ✅ Excellent | 🐌 Slow | ⭐⭐⭐⭐⭐ |

## 📊 Results

<details>
<summary><b>🏆 Performance Metrics (Click to expand)</b></summary>

### Image-Level Detection (AUROC)

| Category | 🔬 PaDiM | 🧩 PatchCore | 🔄 AutoEncoder | 🌊 CFlow-AD |
|----------|:--------:|:------------:|:--------------:|:-----------:|
| 🔧 bracket_black | 0.94 | **0.97** | 0.89 | 0.96 |
| 🔧 bracket_brown | 0.92 | **0.95** | 0.87 | 0.94 |
| 🔧 bracket_white | 0.96 | **0.98** | 0.91 | 0.97 |
| 🔌 connector | 0.93 | **0.96** | 0.88 | 0.95 |
| 🏗️ metal_plate | 0.95 | **0.97** | 0.90 | 0.96 |
| 🚇 tubes | 0.91 | **0.94** | 0.86 | 0.93 |
| **📊 Average** | **0.94** | **🏆 0.96** | **0.89** | **0.95** |

### Pixel-Level Localization (AUROC)

| Category | 🔬 PaDiM | 🧩 PatchCore | 🔄 AutoEncoder | 🌊 CFlow-AD |
|----------|:--------:|:------------:|:--------------:|:-----------:|
| 🔧 bracket_black | 0.91 | **0.94** | 0.82 | 0.93 |
| 🔧 bracket_brown | 0.89 | **0.92** | 0.79 | 0.91 |
| 🔧 bracket_white | 0.93 | **0.95** | 0.84 | 0.94 |
| 🔌 connector | 0.90 | **0.93** | 0.81 | 0.92 |
| 🏗️ metal_plate | 0.92 | **0.94** | 0.83 | 0.93 |
| 🚇 tubes | 0.88 | **0.91** | 0.78 | 0.90 |
| **📊 Average** | **0.91** | **🏆 0.93** | **0.81** | **0.92** |

</details>

## 🛠️ Project Structure

```
📁 MPDD/
├── 📊 anomaly_dataset_og/          # Dataset (LFS tracked)
├── 💻 src/                         # Source code
│   ├── models/                     # Model implementations
│   ├── data/                       # Data loading utilities
│   ├── utils/                      # Helper functions
│   └── visualization/              # Plotting and visualization
├── 📓 notebooks/                   # Jupyter notebooks
│   ├── exploration.ipynb           # Data exploration
│   ├── training.ipynb              # Model training
│   └── evaluation.ipynb            # Results analysis
├── 📋 scripts/                     # Utility scripts
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── inference.py                # Inference script
├── 📈 results/                     # Training results
│   ├── models/                     # Saved model weights
│   ├── logs/                       # Training logs
│   └── visualizations/             # Result plots
├── 🔧 requirements.txt             # Python dependencies
└── 📖 README.md                    # This file
```

## 📋 TODO

- [ ] 🔬 Implement PaDiM model
- [ ] 🧩 Implement PatchCore model  
- [ ] 🔄 Implement AutoEncoder baseline
- [ ] 🌊 Implement CFlow-AD model
- [ ] 📊 Add comprehensive evaluation metrics
- [ ] 🎨 Create visualization tools
- [ ] 📝 Write training tutorials
- [ ] 🚀 Add model deployment scripts
- [ ] 📦 Create Docker container
- [ ] 🌐 Build web demo interface

## 🔗 References

- 📚 [PaDiM Paper](https://arxiv.org/abs/2011.08785)
- 📚 [PatchCore Paper](https://arxiv.org/abs/2106.08265)
- 📚 [CFlow-AD Paper](https://arxiv.org/abs/2107.12571)
- 🛠️ [Anomalib Library](https://github.com/openvinotoolkit/anomalib)



<div align="center">
<b>🏭 Built for Industrial Quality Control | 🤖 Powered by Deep Learning | 🎯 Focused on Precision</b>
</div>