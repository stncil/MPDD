# Manufacturing Product Defect Detection (MPDD)

Computer Vision anomaly detection system for industrial quality control.

## Dataset Structure
Dataset_README.md

## Getting Started

### Installation
```bash
git clone https://github.com/yourusername/MPDD.git
cd MPDD
pip install -r requirements.txt
```

### Usage
```bash
# Train model
python src/train.py --model padim --category bracket_black

# Evaluate model
python src/evaluate.py --model padim --category bracket_black
```
## Models Implemented
- [ ] PaDiM (Patch Distribution Modeling)
- [ ] PatchCore
- [ ] AutoEncoder
- [ ] CFlow-AD

## Results
| Model | Category | AUROC | AP | Inference Time |
|-------|----------|-------|----|--------------------|
| PaDiM | bracket_black | - | - | - |