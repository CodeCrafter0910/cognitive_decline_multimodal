"""
Enhanced Modality Model Training (Problem 11, 15, 16)

Upgrades:
1. Multiple model types: SVM ensemble, XGBoost, Random Forest, MLP
2. Hyperparameter tuning via RandomizedSearchCV (Problem 15)
3. Class-weight balancing and SMOTE (Problem 2)
4. Calibrated probability outputs
5. Model selection based on validation performance
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFIER BUILDERS (Problem 11 — Model Diversity)
# ═══════════════════════════════════════════════════════════════════════════════

def build_classifier(seed: int = 42, model_type: str = "xgboost"):
    """
    Build classifier based on model type.
    
    Args:
        seed: Random seed
        model_type: "xgboost", "svm", "svm_ensemble", "random_forest", "mlp", "gradient_boosting"
    
    Returns:
        Classifier instance
    """
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators  = 300,
            max_depth      = 4,
            learning_rate  = 0.05,
            subsample      = 0.8,
            colsample_bytree = 0.8,
            eval_metric    = "mlogloss",
            random_state   = seed,
            n_jobs         = -1,
            verbosity      = 0,
        )
    
    elif model_type == "svm":
        return SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=seed,
            class_weight='balanced'
        )
    
    elif model_type == "svm_ensemble":
        # Ensemble of 3 SVMs with different kernels and optimized parameters
        svm_rbf = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, random_state=seed, class_weight='balanced')
        svm_poly = SVC(kernel='poly', degree=2, C=5.0, gamma='scale', probability=True, random_state=seed, class_weight='balanced')
        svm_linear = SVC(kernel='linear', C=2.0, probability=True, random_state=seed, class_weight='balanced')
        
        return VotingClassifier(
            estimators=[
                ('svm_rbf', svm_rbf),
                ('svm_poly', svm_poly),
                ('svm_linear', svm_linear)
            ],
            voting='soft',
            n_jobs=-1
        )
    
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,          # Moderate number of trees
            max_depth=6,               # Shallow trees to prevent overfitting
            min_samples_split=15,      # Higher to prevent overfitting on small dataset
            min_samples_leaf=8,        # Higher to prevent overfitting
            max_features='log2',       # Use log2 of features for diversity
            max_samples=0.7,           # Bootstrap 70% of samples
            class_weight='balanced',   # Handle class imbalance
            random_state=seed,
            n_jobs=-1,
            oob_score=True,            # Use out-of-bag score for validation
            bootstrap=True             # Enable bootstrap sampling
        )
    
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=300,          # More trees for stability
            max_depth=5,               # Deeper trees than XGBoost
            learning_rate=0.05,        # Moderate learning rate
            subsample=0.8,             # Bootstrap sampling
            min_samples_split=8,       # Prevent overfitting
            min_samples_leaf=4,        # Prevent overfitting
            max_features='sqrt',       # Feature randomness
            random_state=seed,
            verbose=0
        )
    
    elif model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=1e-3,  # L2 regularization
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER TUNING (Problem 15)
# ═══════════════════════════════════════════════════════════════════════════════

def get_param_distributions(model_type: str) -> dict:
    """Get hyperparameter search space for each model type."""
    if model_type == "xgboost":
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 8],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.01, 0.1, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0, 5.0],
        }
    
    elif model_type == "random_forest":
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 8, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
        }
    
    elif model_type == "svm":
        return {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid'],
        }
    
    elif model_type == "mlp":
        return {
            'hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64), (512, 256, 128)],
            'alpha': [1e-4, 1e-3, 1e-2, 0.1],
            'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005],
            'batch_size': [16, 32, 64],
        }
    
    return {}


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                          model_type: str, seed: int = 42,
                          n_iter: int = 20) -> object:
    """
    Tune hyperparameters using RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Model type string
        seed: Random seed
        n_iter: Number of random search iterations
    
    Returns:
        Best estimator
    """
    param_dist = get_param_distributions(model_type)
    
    if not param_dist:
        print(f"    No tuning available for {model_type}, using defaults")
        return build_classifier(seed, model_type)
    
    base_clf = build_classifier(seed, model_type)
    
    # Use stratified k-fold for tuning
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    search = RandomizedSearchCV(
        base_clf,
        param_distributions=param_dist,
        n_iter=min(n_iter, len(param_dist) * 2),  # Don't exceed reasonable search
        cv=cv,
        scoring='f1_macro',
        random_state=seed,
        n_jobs=-1,
        verbose=0,
        error_score=0.0  # Return 0 for failed configurations
    )
    
    search.fit(X_train, y_train)
    
    print(f"    Best params: {search.best_params_}")
    print(f"    Best CV F1: {search.best_score_:.3f}")
    
    return search.best_estimator_


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA BALANCING (Problem 2)
# ═══════════════════════════════════════════════════════════════════════════════

def balance_data(X_train: np.ndarray, y_train: np.ndarray,
                  method: str = "smote", seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance class distribution using SMOTE or other methods.
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: "smote", "oversample", or "none"
        seed: Random seed
    
    Returns:
        Balanced X_train, y_train
    """
    if method == "smote" and SMOTE_AVAILABLE:
        try:
            # Check if we have enough samples per class for SMOTE
            min_class_count = min(np.bincount(y_train))
            k_neighbors = min(5, min_class_count - 1)
            
            if k_neighbors >= 1:
                sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)
                X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
                print(f"    SMOTE: {len(y_train)} → {len(y_resampled)} samples")
                print(f"    Class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
                return X_resampled, y_resampled
            else:
                print(f"    SMOTE skipped: too few samples per class (min={min_class_count})")
        except Exception as e:
            print(f"    SMOTE failed: {e}")
    
    elif method == "oversample":
        # Simple random oversampling
        max_count = max(np.bincount(y_train))
        X_list = [X_train]
        y_list = [y_train]
        
        for cls in np.unique(y_train):
            cls_mask = y_train == cls
            cls_count = cls_mask.sum()
            if cls_count < max_count:
                n_extra = max_count - cls_count
                indices = np.random.choice(np.where(cls_mask)[0], size=n_extra, replace=True)
                X_list.append(X_train[indices])
                y_list.append(y_train[indices])
        
        X_resampled = np.vstack(X_list)
        y_resampled = np.concatenate(y_list)
        
        # Shuffle
        perm = np.random.permutation(len(y_resampled))
        print(f"    Oversampling: {len(y_train)} → {len(y_resampled)} samples")
        return X_resampled[perm], y_resampled[perm]
    
    return X_train, y_train


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def scale_features(X_train: np.ndarray, X_val: np.ndarray,
                   X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    return X_train, X_val, X_test, scaler


def compute_metrics(clf, X: np.ndarray, y: np.ndarray, tag: str) -> Dict:
    """Compute comprehensive evaluation metrics (Problem 18)."""
    y_pred  = clf.predict(X)
    y_proba = clf.predict_proba(X)
    acc     = float(accuracy_score(y, y_pred))
    f1      = float(f1_score(y, y_pred, average="macro", zero_division=0))
    
    try:
        auc = float(roc_auc_score(
            label_binarize(y, classes=[0, 1, 2]),
            y_proba, multi_class="ovr", average="macro"
        ))
    except Exception:
        auc = float("nan")
    
    # Per-class sensitivity and specificity (Problem 18)
    from sklearn.metrics import recall_score, precision_score
    
    per_class = {}
    for cls_idx, cls_name in enumerate(["CN", "MCI", "AD"]):
        y_binary_true = (y == cls_idx).astype(int)
        y_binary_pred = (y_pred == cls_idx).astype(int)
        
        sens = float(recall_score(y_binary_true, y_binary_pred, zero_division=0))
        prec = float(precision_score(y_binary_true, y_binary_pred, zero_division=0))
        
        # Specificity = TN / (TN + FP)
        tn = ((y_binary_true == 0) & (y_binary_pred == 0)).sum()
        fp = ((y_binary_true == 0) & (y_binary_pred == 1)).sum()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        per_class[cls_name] = {
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
        }
    
    print(f"    {tag:<22}  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")
    
    return {
        "tag": tag, "accuracy": acc, "f1_macro": f1, "roc_auc": auc,
        "per_class": per_class
    }


def train_modality_model(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray,   y_val: np.ndarray,
                         modality: str, seed: int = 42,
                         model_type: str = "xgboost",
                         tune: bool = False,
                         balance: str = "none") -> Tuple:
    """
    Train a modality-specific model with optional tuning and balancing.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        modality: Name of modality (for logging)
        seed: Random seed
        model_type: Classifier type
        tune: Whether to perform hyperparameter tuning
        balance: Data balancing method ("smote", "oversample", "none")
    
    Returns:
        clf: Trained (and calibrated) classifier
        metrics: Performance metrics dictionary
    """
    print(f"\n  Training {modality} ({model_type})  —  X_train={X_train.shape}")
    
    # Balance data if requested
    X_balanced, y_balanced = balance_data(X_train, y_train, method=balance, seed=seed)
    
    # Build or tune classifier
    if tune:
        print(f"    Tuning hyperparameters...")
        clf = tune_hyperparameters(X_balanced, y_balanced, model_type, seed)
    else:
        clf = build_classifier(seed, model_type)
        clf.fit(X_balanced, y_balanced)

    # Calibrate probabilities
    n_cv = min(3, min(np.bincount(y_val.astype(int))))
    if n_cv >= 2:
        try:
            cal = CalibratedClassifierCV(clf, cv=n_cv, method="sigmoid")
            cal.fit(X_val, y_val)
            clf = cal
            print(f"    Calibration applied (cv={n_cv})")
        except Exception as e:
            print(f"    Calibration skipped: {e}")
    else:
        print(f"    Calibration skipped (insufficient samples per class)")

    metrics = compute_metrics(clf, X_val, y_val, f"{modality}/val")
    return clf, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL PERSISTENCE (Problem 23 — Versioning)
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(clf, path: Path, metadata: dict = None):
    """Save model with optional metadata for versioning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        "model": clf,
        "metadata": metadata or {}
    }
    
    with open(path, "wb") as f:
        pickle.dump(save_data, f)


def load_model(path: Path):
    """Load model (supports both old and new format)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Handle both old format (just model) and new format (dict with metadata)
    if isinstance(data, dict) and "model" in data:
        return data["model"]
    return data
