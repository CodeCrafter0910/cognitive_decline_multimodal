"""
Enhanced Feature Extractor (Problem 5, 7, 8)

Upgrades:
1. True 3D CNN features (Brain3DCNN) instead of 2D ResNet18 slices
2. Extended statistical features (texture, shape, regional)
3. Feature selection using Mutual Information / PCA
4. Combined deep + statistical features for maximum discriminative power
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import sobel, laplace

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Using statistical features only.")


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL FEATURE EXTRACTION (Problem 8 — Extended)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_volume_features(volume: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive statistical features from a 3D brain volume.
    
    Original: 17 features (mean, std, median, min, max, percentiles, etc.)
    Enhanced: 40+ features including texture, gradient, and regional features.
    """
    v = volume.flatten()

    # ── Basic Statistical Features (original 17) ───────────────────────
    basic_features = [
        v.mean(),                                           # 1. Global mean
        v.std(),                                            # 2. Global std
        float(np.median(v)),                               # 3. Median
        v.min(),                                            # 4. Min
        v.max(),                                            # 5. Max
        float(np.percentile(v, 5)),                        # 6. P5
        float(np.percentile(v, 25)),                       # 7. P25 (Q1)
        float(np.percentile(v, 75)),                       # 8. P75 (Q3)
        float(np.percentile(v, 95)),                       # 9. P95
        float(skew(v)),                                     # 10. Skewness
        float(kurtosis(v)),                                 # 11. Kurtosis
        float((v != 0).mean()),                            # 12. Non-zero fraction
        float((v ** 2).mean()),                            # 13. Energy (RMS)
        float(-np.sum(                                      # 14. Entropy
            (lambda h: h)(np.histogram(v, bins=64, density=True)[0]) *
            np.log2(np.histogram(v, bins=64, density=True)[0] + 1e-10)
        ) / np.log2(64)),
        float(volume.mean(axis=(1, 2)).std()),             # 15. Axial variation
        float(volume.mean(axis=(0, 2)).std()),             # 16. Coronal variation
        float(volume.mean(axis=(0, 1)).std()),             # 17. Sagittal variation
    ]
    
    # ── Texture Features (Problem 8) ───────────────────────────────────
    texture_features = []
    
    # Gradient-based features (Sobel filter)
    try:
        grad_x = sobel(volume, axis=0)
        grad_y = sobel(volume, axis=1)
        grad_z = sobel(volume, axis=2)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        texture_features.extend([
            float(grad_magnitude.mean()),                   # 18. Mean gradient
            float(grad_magnitude.std()),                    # 19. Gradient std
            float(grad_magnitude.max()),                    # 20. Max gradient
            float((grad_magnitude > grad_magnitude.mean()).mean()),  # 21. Edge density
        ])
    except Exception:
        texture_features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Laplacian features (2nd order derivative)
    try:
        lap = laplace(volume)
        texture_features.extend([
            float(np.abs(lap).mean()),                     # 22. Mean Laplacian
            float(np.abs(lap).std()),                      # 23. Laplacian std
        ])
    except Exception:
        texture_features.extend([0.0, 0.0])
    
    # ── Regional Features (Problem 8 — Shape) ─────────────────────────
    # Divide brain into 8 octants and compute per-region statistics
    regional_features = []
    d, h, w = volume.shape
    for di in [0, 1]:
        for hi in [0, 1]:
            for wi in [0, 1]:
                region = volume[
                    di*d//2:(di+1)*d//2,
                    hi*h//2:(hi+1)*h//2,
                    wi*w//2:(wi+1)*w//2
                ]
                regional_features.extend([
                    float(region.mean()),                   # Mean per octant
                    float(region.std()),                    # Std per octant
                ])
    # 16 regional features (8 octants × 2 stats each)
    
    # ── Asymmetry Features ─────────────────────────────────────────────
    # Left-right brain asymmetry (important for AD detection)
    left_half = volume[:, :, :w//2]
    right_half = volume[:, :, w//2:]
    right_flipped = right_half[:, :, ::-1]
    
    # Match shapes for comparison
    min_w = min(left_half.shape[2], right_flipped.shape[2])
    left_matched = left_half[:, :, :min_w]
    right_matched = right_flipped[:, :, :min_w]
    
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lr_corr = float(np.corrcoef(left_matched.flatten(), right_matched.flatten())[0, 1])
            if np.isnan(lr_corr):
                lr_corr = 0.0
    except Exception:
        lr_corr = 0.0
    
    asymmetry_features = [
        float(np.abs(left_matched.mean() - right_matched.mean())),     # Mean asymmetry
        float(np.abs(left_matched.std() - right_matched.std())),       # Std asymmetry
        lr_corr,                                                        # LR correlation
    ]
    
    # ── Intensity Distribution Features ────────────────────────────────
    distribution_features = [
        float(np.percentile(v, 10)),                       # P10
        float(np.percentile(v, 90)),                       # P90
        float(np.percentile(v, 75) - np.percentile(v, 25)),  # IQR
        float(v.var()),                                     # Variance
        float(np.mean(np.abs(v - v.mean()))),              # Mean absolute deviation
    ]
    
    # Combine all features
    all_features = (basic_features + texture_features + 
                   regional_features + asymmetry_features + distribution_features)
    
    return np.array(all_features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  3D CNN FEATURE EXTRACTION (Problem 5 — Replaces 2D ResNet18)
# ═══════════════════════════════════════════════════════════════════════════════

class CNN3DFeatureExtractor:
    """
    True 3D CNN feature extractor using Brain3DCNN.
    
    UPGRADE: Processes the ENTIRE 64×64×64 brain volume instead of just 3 middle slices.
    This captures full 3D spatial relationships and volumetric patterns.
    
    Falls back to 2D ResNet18 if 3D CNN is not available.
    """
    
    def __init__(self, use_3d=True, feature_dim=256, dropout=0.5):
        if not TORCH_AVAILABLE:
            self.model = None
            self.use_3d = False
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_3d = use_3d
        
        if use_3d:
            try:
                # Try absolute import first
                try:
                    from models.brain3d_cnn import Brain3DCNN
                except ImportError:
                    # Try relative import
                    import sys
                    from pathlib import Path
                    parent_dir = Path(__file__).resolve().parent.parent
                    if str(parent_dir) not in sys.path:
                        sys.path.insert(0, str(parent_dir))
                    from models.brain3d_cnn import Brain3DCNN
                
                self.model = Brain3DCNN(
                    in_channels=1,
                    num_classes=3,
                    feature_dim=feature_dim,
                    dropout_rate=dropout
                ).to(self.device)
                self.model.eval()
                n_params = sum(p.numel() for p in self.model.parameters())
                print(f"  ✓ 3D CNN initialized ({n_params:,} parameters)")
            except Exception as e:
                print(f"  ⚠ 3D CNN failed ({e}), falling back to 2D ResNet18")
                self.use_3d = False
                self._init_2d_model()
        else:
            self._init_2d_model()
    
    def _init_2d_model(self):
        """Initialize 2D ResNet18 as fallback."""
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
    
    def extract_cnn_features(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract CNN features from 3D brain volume.
        
        If use_3d=True: processes entire volume with 3D CNN → 256-dim features
        If use_3d=False: processes 3 middle slices with 2D ResNet → 512-dim features
        """
        if self.model is None:
            return np.array([], dtype=np.float32)
        
        try:
            if self.use_3d:
                return self._extract_3d_features(volume)
            else:
                return self._extract_2d_features(volume)
        except Exception as e:
            print(f"    CNN extraction error: {e}")
            return np.array([], dtype=np.float32)
    
    def _extract_3d_features(self, volume: np.ndarray) -> np.ndarray:
        """Extract features using 3D CNN (entire volume)."""
        # (D, H, W) → (1, 1, D, H, W)
        tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.extract_features(tensor)
        
        return features.squeeze().cpu().numpy().astype(np.float32)
    
    def _extract_2d_features(self, volume: np.ndarray) -> np.ndarray:
        """Extract features using 2D ResNet18 on middle slices (fallback)."""
        from scipy.ndimage import zoom
        
        d, h, w = volume.shape
        slices = [
            volume[d//2, :, :],      # Axial
            volume[:, h//2, :],      # Coronal
            volume[:, :, w//2],      # Sagittal
        ]
        
        slice_features = []
        with torch.no_grad():
            for slice_2d in slices:
                slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
                slice_rgb = np.stack([slice_norm, slice_norm, slice_norm], axis=0)
                factors = [1.0, 224.0 / slice_rgb.shape[1], 224.0 / slice_rgb.shape[2]]
                slice_resized = zoom(slice_rgb, factors, order=1)
                
                tensor = torch.from_numpy(slice_resized).float().unsqueeze(0).to(self.device)
                features = self.model(tensor).squeeze().cpu().numpy()
                slice_features.append(features)
        
        aggregated = np.mean(slice_features, axis=0).astype(np.float32)
        return aggregated


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE SELECTION (Problem 7)
# ═══════════════════════════════════════════════════════════════════════════════

def select_features(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, X_test: np.ndarray = None,
                    method: str = "mutual_info", k: int = 200) -> tuple:
    """
    Select top-K features using specified method.
    
    Args:
        X_train: (N, D) training features
        y_train: (N,) training labels
        X_val, X_test: validation/test features to transform
        method: "mutual_info", "pca", "f_classif", or "lasso"
        k: Number of features to keep
    
    Returns:
        Transformed (X_train, X_val, X_test, selector)
    """
    from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
    from sklearn.decomposition import PCA
    
    k = min(k, X_train.shape[1])  # Can't select more features than available
    
    if method == "mutual_info":
        selector = SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=42),
            k=k
        )
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val) if X_val is not None else None
        X_test_selected = selector.transform(X_test) if X_test is not None else None
        
        # Log which features were selected
        selected_mask = selector.get_support()
        print(f"    Feature selection ({method}): {X_train.shape[1]} → {k} features")
    
    elif method == "f_classif":
        # ANOVA F-test - good for continuous features
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val) if X_val is not None else None
        X_test_selected = selector.transform(X_test) if X_test is not None else None
        
        print(f"    Feature selection ({method}): {X_train.shape[1]} → {k} features")
        
    elif method == "pca":
        selector = PCA(n_components=k, random_state=42)
        X_train_selected = selector.fit_transform(X_train)
        X_val_selected = selector.transform(X_val) if X_val is not None else None
        X_test_selected = selector.transform(X_test) if X_test is not None else None
        
        explained_var = selector.explained_variance_ratio_.sum()
        print(f"    Feature selection (PCA): {X_train.shape[1]} → {k} components "
              f"({explained_var:.1%} variance explained)")
    
    else:
        # No selection
        return X_train, X_val, X_test, None
    
    return X_train_selected, X_val_selected, X_test_selected, selector


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all_features(npy_dir: Path, subject_ids: List[str],
                          use_cnn: bool = True, use_3d: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from volumes using both statistical and CNN-based methods.
    
    Args:
        npy_dir: Directory containing .npy volume files
        subject_ids: List of subject IDs to process
        use_cnn: If True and PyTorch available, use CNN features
        use_3d: If True, use 3D CNN; otherwise use 2D ResNet18 fallback
    
    Returns:
        X: Feature matrix (N x D)
        found: List of subject IDs successfully processed
    """
    features = []
    found    = []
    
    # Initialize CNN extractor
    cnn_extractor = None
    if use_cnn and TORCH_AVAILABLE:
        print(f"  Initializing {'3D CNN' if use_3d else '2D ResNet18'} feature extractor...")
        cnn_extractor = CNN3DFeatureExtractor(use_3d=use_3d)
        if cnn_extractor.model is None:
            cnn_extractor = None

    for idx, sid in enumerate(subject_ids):
        path = npy_dir / f"{sid}.npy"
        if not path.exists():
            print(f"  [MISSING] {sid}")
            continue
        
        try:
            vol = np.load(str(path)).astype(np.float32)
            
            # Extract enhanced statistical features
            stat_feat = extract_volume_features(vol)
            
            # Extract CNN features
            if cnn_extractor is not None:
                cnn_feat = cnn_extractor.extract_cnn_features(vol)
                if len(cnn_feat) > 0:
                    feat = np.concatenate([stat_feat, cnn_feat])
                else:
                    feat = stat_feat
            else:
                feat = stat_feat
            
            features.append(feat)
            found.append(sid)
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(subject_ids)} subjects...")
            
        except Exception as e:
            print(f"  [ERROR] {sid}: {e}")
            continue

    if len(features) == 0:
        print(f"  ERROR: No features extracted!")
        return np.array([]).reshape(0, 0), []
    
    X = np.vstack(features).astype(np.float32)
    
    cnn_type = "3D CNN" if (cnn_extractor and cnn_extractor.use_3d) else "2D ResNet18" if cnn_extractor else "none"
    stat_count = len(extract_volume_features(np.zeros((4,4,4))))
    cnn_count = X.shape[1] - stat_count if cnn_extractor else 0
    
    print(f"  ✓ Features extracted: {X.shape}")
    print(f"    Statistical features: {stat_count}")
    print(f"    CNN features ({cnn_type}): {cnn_count}")
    print(f"    Total: {X.shape[1]} | Missing: {len(subject_ids) - len(found)}")
    
    return X, found
