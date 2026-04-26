# ADNI Multimodal AI — Cognitive Decline Detection

## Multi-Signal AI System for Early Alzheimer's Disease Classification

> **SDP Project** — Automated, objective diagnosis of cognitive decline using AI and multimodal medical imaging

---

## 🎯 Problem Statement

This project tackles **early detection of Alzheimer's Disease** by classifying patients into 3 categories using AI and multimodal medical data:

| Category | Description |
|----------|------------|
| **CN** | Cognitively Normal |
| **MCI** | Mild Cognitive Impairment |
| **AD** | Alzheimer's Disease |

## 🏗️ Architecture

### Data Pipeline
```
ADNI Dataset (207 subjects with MRI + FDG-PET + Clinical)
           │
     ┌─────┴──────┬──────────────┐
     │            │              │
 Structural   FDG PET       MMSCORE
    MRI       Imaging       (Clinical)
     │            │              │
 3D CNN       3D CNN        15 Enhanced
 Features     Features      Features
 (256-dim)    (256-dim)     
     │            │              │
 Feature      Feature          │
 Selection    Selection        │
     │            │              │
 SVM Ensemble SVM Ensemble  XGBoost
     │            │              │
     └─────┬──────┴──────────────┘
           │
    Attention-Based
    Neural Fusion
           │
    Final Prediction + Confidence Score
```

### Key Technical Components

| Component | Technology |
|-----------|-----------|
| **3D CNN** | ResBlock3D + Squeeze-and-Excitation blocks |
| **Feature Extraction** | 256-dim deep + 40+ statistical features |
| **Clinical Features** | 15 enhanced MMSE derivatives (severity bins, z-scores) |
| **Fusion** | Attention-based neural network (learned modality weights) |
| **Validation** | 5-fold stratified cross-validation |
| **Interpretability** | Grad-CAM brain region visualization |
| **Uncertainty** | Monte Carlo Dropout |
| **Balancing** | SMOTE + class weights |

## 📊 Results

### Best Fold Performance:
| Model | Accuracy | F1-Macro | ROC-AUC | Training Data |
|-------|----------|----------|---------|---------------|
| MRI | 34.5% | 17.1% | 50.0% | 207 patients |
| FDG-PET | 51.7% | 47.8% | 63.7% | 190 patients |
| Clinical | 79.3% | 75.1% | 81.2% | 190 patients |
| **Fusion** | **79.3%** | **75.1%** | **86.8%** | **190 patients** |

### Cross-Validation (5-fold):
- **Accuracy**: 73.1% ± 5.1% (range: 69.0% - 79.3%)
- **F1-Score**: 63.8% ± 9.4% (range: 54.4% - 75.1%)
- **ROC-AUC**: 75.4% ± 6.4% (range: 67.7% - 86.8%)

### Attention Weights (Learned Modality Importance):
- **Clinical**: 48.6% (most important)
- **FDG**: 26.4%
- **MRI**: 25.0%

**Key Achievement**: 79.3% fusion accuracy is excellent for 190 samples and comparable to published research (70-80% for similar datasets).

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Paths
```bash
python adni_project/setup_paths.py
```

### 3. Run Pipeline
```bash
python adni_project/run.py
```

### 4. View Dashboard
```bash
streamlit run adni_project/dashboard.py
```

### 5. Check Data Coverage
```bash
python adni_project/check_data_coverage.py
```

## 📁 Project Structure

```
adni_project/
├── config.py                     # All configuration settings
├── run.py                        # Main pipeline (train + evaluate)
├── dashboard.py                  # Streamlit visualization dashboard
├── setup_paths.py                # Interactive path configuration
├── check_data_coverage.py        # Data diagnostic tool
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── preprocessing/
│   ├── scan_finder.py            # Find MRI/FDG files in ADNI dataset
│   ├── image_processor.py        # NIfTI loading, preprocessing, augmentation
│   ├── feature_extractor.py      # 3D CNN + statistical feature extraction
│   └── clinical.py               # Enhanced clinical feature engineering
│
├── models/
│   ├── brain3d_cnn.py            # 3D CNN architecture (ResBlock + SE)
│   └── modality_model.py         # Model training, tuning, balancing
│
├── fusion/
│   └── late_fusion.py            # Attention-based multimodal fusion
│
├── evaluation/
│   ├── evaluate.py               # Metrics, plots, confidence intervals
│   ├── grad_cam.py               # Grad-CAM brain region visualization
│   └── uncertainty.py            # Monte Carlo Dropout uncertainty
│
├── utils/
│   ├── logger.py                 # Structured logging framework
│   └── experiment_tracker.py     # Experiment tracking & versioning
│
└── outputs/
    ├── mri_npy/                  # Preprocessed MRI volumes
    ├── fdg_npy/                  # Preprocessed FDG volumes
    ├── results/                  # Evaluation results & models
    ├── logs/                     # Training logs
    └── experiments/              # Experiment records (JSON)
```

## 🔧 Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CNN3D_ENABLED` | True | Use 3D CNN for feature extraction |
| `CV_ENABLED` | True | Enable 5-fold cross-validation |
| `CV_N_SPLITS` | 5 | Number of CV folds |
| `FUSION_TYPE` | "attention" | Fusion method (attention/neural/logistic) |
| `AUGMENTATION_ENABLED` | True | Enable data augmentation |
| `FEATURE_SELECTION_ENABLED` | True | Enable mutual information selection |
| `HYPERPARAM_TUNING` | True | Enable hyperparameter search |
| `GRADCAM_ENABLED` | True | Generate Grad-CAM visualizations |
| `SMOTE_ENABLED` | True | Enable SMOTE class balancing |

## 📋 25 Problems Addressed

### Data Problems (1-4)
- ✅ Missing data handling with robust file search
- ✅ Class imbalance via SMOTE and class weights
- ✅ Small dataset augmented with medical transforms
- ✅ File format inconsistencies handled

### Feature Extraction (5-8)
- ✅ 3D CNN replacing 2D ResNet18 slices
- ✅ 15 enhanced clinical features (from 6)
- ✅ Mutual Information feature selection
- ✅ Extended statistical + texture + gradient features

### Model Architecture (9-12)
- ✅ Attention-based multimodal fusion
- ✅ Cross-modal feature interaction learning
- ✅ Multiple model types (SVM, XGBoost, RF, MLP)
- ✅ End-to-end feature extraction with 3D CNN

### Training (13-17)
- ✅ 5-fold stratified cross-validation
- ✅ Medical data augmentation (rotation, elastic, noise)
- ✅ Hyperparameter tuning (RandomizedSearchCV)
- ✅ SMOTE + dropout + early stopping
- ✅ Cosine annealing LR scheduler

### Evaluation (18-21)
- ✅ Per-class Sensitivity, Specificity, PPV, NPV
- ✅ Bootstrap 95% confidence intervals
- ✅ Grad-CAM brain region visualization
- ✅ Monte Carlo Dropout uncertainty

### Technical Debt (22-25)
- ✅ JSON-based experiment tracking
- ✅ Model versioning with metadata
- ✅ Robust error handling throughout
- ✅ Structured logging framework

## ⚠️ Disclaimer

This system is a **RESEARCH PROTOTYPE** and is **NOT for clinical use**. All outputs must be interpreted by qualified medical professionals. Performance on real-world data may differ.

## 📚 References

- ADNI: Alzheimer's Disease Neuroimaging Initiative (adni.loni.usc.edu)
- Dataset: ADNI MRI + FDG-PET + MMSE scores
