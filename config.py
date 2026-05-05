from pathlib import Path
import os

# ── Dataset Paths ──────────────────────────────────────────────────────────────
# Use environment variables if available, otherwise use defaults
ADNI_ROOT   = Path(os.getenv("ADNI_ROOT", r"C:\Users\Rishabh Khanna\OneDrive\Desktop\DataSet(Final)\ADNI"))
CSV_PATH    = Path(os.getenv("CSV_PATH", r"C:\Users\Rishabh Khanna\OneDrive\Desktop\DataSet(Clean)\FINAL_MULTIMODAL_207_FULLY_PAIRED_WITH_MMSE.csv"))

# ── Output Paths ───────────────────────────────────────────────────────────────
# Default to workspace-relative outputs if environment variable not set
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", Path(__file__).parent / "outputs"))
MRI_NPY_DIR = OUTPUT_ROOT / "mri_npy"
FDG_NII_DIR = OUTPUT_ROOT / "fdg_nii"
FDG_NPY_DIR = OUTPUT_ROOT / "fdg_npy"
RESULTS_DIR = OUTPUT_ROOT / "results"
LOG_DIR     = OUTPUT_ROOT / "logs"
EXPERIMENT_DIR = OUTPUT_ROOT / "experiments"

# ── Preprocessing Settings ─────────────────────────────────────────────────────
TARGET_SHAPE    = (64, 64, 64)
CLIP_LOW_PCT    = 0.5
CLIP_HIGH_PCT   = 99.5

# ── Training Settings ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15

# ── Cross-Validation Settings (Problem 13) ─────────────────────────────────────
CV_ENABLED   = True          # Enable stratified K-fold cross-validation
CV_N_SPLITS  = 5             # Number of folds
CV_SHUFFLE   = True

# ── Labels ─────────────────────────────────────────────────────────────────────
LABEL_MAP   = {"CN": 0, "MCI": 1, "AD": 2}
LABEL_NAMES = ["CN", "MCI", "AD"]

# ── CNN Feature Settings (Problem 5) ────────────────────────────────────────────
# Using enhanced statistical features only (47-dim)
# Small dataset (190 samples) works better with domain-specific features
CNN3D_ENABLED      = False   # Disabled for small dataset
USE_CNN_FEATURES   = False   # Disabled for small dataset
CNN3D_IN_CHANNELS  = 1
CNN3D_BASE_FILTERS = 32
CNN3D_NUM_BLOCKS   = 4
CNN3D_FEATURE_DIM  = 256     # Output feature dimension from 3D CNN

# ── Data Augmentation Settings (Problem 14) ────────────────────────────────────
AUGMENTATION_ENABLED = True
AUG_ROTATION_RANGE   = 10       # degrees
AUG_SCALE_RANGE      = (0.9, 1.1)
AUG_TRANSLATION      = 5        # voxels
AUG_NOISE_STD        = 0.05
AUG_FLIP_PROB        = 0.5      # Left-right flip (brain symmetry)
AUG_GAMMA_RANGE      = (-0.3, 0.3)
AUG_ELASTIC_ENABLED  = True
AUG_ELASTIC_ALPHA    = 7.5
AUG_MULTIPLIER       = 3        # How many augmented copies per original

# ── Attention Fusion Settings (Problem 9, 10) ──────────────────────────────────
FUSION_TYPE       = "attention"   # "logistic", "attention", or "neural"
FUSION_HIDDEN_DIM = 128
FUSION_DROPOUT    = 0.3
FUSION_LR         = 0.001
FUSION_EPOCHS     = 150
FUSION_PATIENCE   = 20            # Early stopping patience

# ── Hyperparameter Tuning (Problem 15) ──────────────────────────────────────────
HYPERPARAM_TUNING = True     # Enable tuning for better performance
TUNING_N_ITER     = 20       # Reduced from 30 for faster training

# ── Regularization (Problem 16) ─────────────────────────────────────────────────
DROPOUT_RATE      = 0.5
WEIGHT_DECAY      = 1e-4
EARLY_STOPPING    = True
EARLY_STOP_PATIENCE = 15

# ── Learning Rate Scheduling (Problem 17) ───────────────────────────────────────
LR_SCHEDULER      = "cosine"     # "cosine", "plateau", or "step"
LR_INITIAL         = 0.001
LR_MIN              = 1e-6
LR_WARMUP_EPOCHS   = 5

# ── Feature Selection (Problem 7) ──────────────────────────────────────────────
# Enable light feature selection to help with small dataset
FEATURE_SELECTION_ENABLED = True
FEATURE_SELECTION_METHOD  = "mutual_info"  # "mutual_info", "pca", "lasso"
FEATURE_SELECTION_K       = 30             # Use top 30 features (works well for FDG)

# ── Data Imbalance Handling (Problem 2) ─────────────────────────────────────────
CLASS_WEIGHT_METHOD = "smote"       # "smote", "oversample", "none" ("balanced" handled by class_weight param)
FOCAL_LOSS_GAMMA    = 2.0
SMOTE_ENABLED       = True

# ── Evaluation Settings (Problem 18, 19) ────────────────────────────────────────
BOOTSTRAP_N_ITERATIONS = 1000      # For confidence intervals
CONFIDENCE_LEVEL       = 0.95

# ── Interpretability (Problem 20) ───────────────────────────────────────────────
GRADCAM_ENABLED = True
GRADCAM_N_SAMPLES = 5              # Number of samples to visualize

# ── Uncertainty Quantification (Problem 21) ─────────────────────────────────────
UNCERTAINTY_ENABLED = True
MC_DROPOUT_N_FORWARD = 30          # Number of forward passes for MC Dropout

# ── Logging (Problem 25) ────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Print configuration on import (only in non-Streamlit environments)
if __name__ != "__main__":
    try:
        import streamlit
        # Running in Streamlit - skip verbose output
    except ImportError:
        # Not in Streamlit - show config
        print(f"Configuration loaded:")
        print(f"  ADNI_ROOT:   {ADNI_ROOT}")
        print(f"  CSV_PATH:    {CSV_PATH}")
        print(f"  OUTPUT_ROOT: {OUTPUT_ROOT}")
        print(f"  Features:    Enhanced Stats (47-dim) - domain-specific features")
        print(f"  SMOTE:       {'Enabled' if CLASS_WEIGHT_METHOD == 'smote' else 'Disabled'}")
        print(f"  Cross-Val:   {CV_N_SPLITS}-fold" if CV_ENABLED else "  Cross-Val:   Disabled")
        print(f"  Fusion:      {FUSION_TYPE}")
