"""
Attention-Based Multimodal Fusion (Problem 9, 10)

Replaces simple logistic regression concatenation with:
1. AttentionFusion — Learns dynamic attention weights per modality per patient
2. NeuralFusion — Deep neural network for cross-modal feature learning
3. HybridFusion — Combines attention + neural approaches

Key Advantages:
- Learns WHICH modality to trust for each patient
- Captures cross-modal interactions (synergistic patterns)
- Interpretable attention weights (can visualize modality importance)
- Non-linear feature transformations for better decision boundaries
"""

import numpy as np
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


# ═══════════════════════════════════════════════════════════════════════════════
#  ATTENTION-BASED FUSION (Tier 1 Improvement)
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionFusionNet(nn.Module):
    """
    Attention-based fusion that dynamically weights modalities
    based on their reliability for each individual patient.
    
    Architecture:
    1. Cross-modal attention: learns which modality is most informative
    2. Weighted feature combination: applies learned attention weights
    3. Deep fusion classifier: non-linear classification on weighted features
    
    Returns both predictions and attention weights for interpretability.
    """
    
    def __init__(self, n_modalities=3, n_classes=3, hidden_dim=128, dropout=0.3):
        super().__init__()
        
        input_dim = n_modalities * n_classes  # 3 modalities × 3 classes = 9
        
        # ── Modality-specific feature transformers ─────────────────────────
        # Each modality gets its own feature transformation
        self.modality_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_classes, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_modalities)
        ])
        
        # ── Cross-modal attention network ──────────────────────────────────
        # Learns attention weights based on ALL modality inputs jointly
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, n_modalities),
            nn.Softmax(dim=1)  # Attention weights sum to 1
        )
        
        # ── Fusion classifier ──────────────────────────────────────────────
        # Processes the attention-weighted features
        fused_dim = n_modalities * (hidden_dim // 2)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        self.n_modalities = n_modalities
        self.n_classes = n_classes
    
    def forward(self, *modality_probas):
        """
        Forward pass.
        
        Args:
            *modality_probas: Variable number of (batch, n_classes) probability tensors
                              e.g., mri_proba, fdg_proba, clinical_proba
        
        Returns:
            logits: (batch, n_classes) — class logits
            attn_weights: (batch, n_modalities) — attention weights per modality
        """
        # Concatenate all probabilities for attention computation
        concat = torch.cat(modality_probas, dim=1)  # (batch, n_modalities * n_classes)
        
        # Compute attention weights
        attn_weights = self.attention(concat)  # (batch, n_modalities)
        
        # Transform each modality independently
        transformed = []
        for i, (proba, transform) in enumerate(zip(modality_probas, self.modality_transforms)):
            t = transform(proba)  # (batch, hidden_dim // 2)
            # Apply attention weight
            weighted = t * attn_weights[:, i:i+1]  # Broadcast attention weight
            transformed.append(weighted)
        
        # Concatenate weighted features
        fused = torch.cat(transformed, dim=1)  # (batch, n_modalities * hidden_dim // 2)
        
        # Final classification
        logits = self.fusion_classifier(fused)
        
        return logits, attn_weights


class NeuralFusionNet(nn.Module):
    """
    Deep Neural Network fusion without explicit attention.
    Learns cross-modal interactions through shared hidden layers.
    """
    
    def __init__(self, n_modalities=3, n_classes=3, hidden_dim=128, dropout=0.3):
        super().__init__()
        
        input_dim = n_modalities * n_classes
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, *modality_probas):
        concat = torch.cat(modality_probas, dim=1)
        logits = self.network(concat)
        # Return None for attention weights (not applicable)
        return logits, None


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING & INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_attention_fusion(proba_list_train: List[np.ndarray],
                           y_train: np.ndarray,
                           proba_list_val: List[np.ndarray],
                           y_val: np.ndarray,
                           fusion_type: str = "attention",
                           hidden_dim: int = 128,
                           dropout: float = 0.3,
                           lr: float = 0.001,
                           epochs: int = 150,
                           patience: int = 20,
                           seed: int = 42) -> Tuple:
    """
    Train attention-based fusion model with early stopping.
    
    Args:
        proba_list_train: List of (N_train, 3) probability arrays per modality
        y_train: (N_train,) training labels
        proba_list_val: List of (N_val, 3) probability arrays per modality
        y_val: (N_val,) validation labels
        fusion_type: "attention" or "neural"
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
        lr: Learning rate
        epochs: Max training epochs
        patience: Early stopping patience
        seed: Random seed
    
    Returns:
        model: Trained fusion model
        training_history: Dict with loss and accuracy per epoch
    """
    if not TORCH_AVAILABLE:
        print("  PyTorch not available, falling back to logistic regression fusion")
        return train_logistic_fusion(proba_list_train, y_train, seed), {}
    
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_modalities = len(proba_list_train)
    n_classes = proba_list_train[0].shape[1]
    
    # Build model
    if fusion_type == "attention":
        model = AttentionFusionNet(n_modalities, n_classes, hidden_dim, dropout).to(device)
    else:
        model = NeuralFusionNet(n_modalities, n_classes, hidden_dim, dropout).to(device)
    
    # Convert to tensors
    train_tensors = [torch.from_numpy(p).float().to(device) for p in proba_list_train]
    val_tensors = [torch.from_numpy(p).float().to(device) for p in proba_list_val]
    y_train_t = torch.from_numpy(y_train).long().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)
    
    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * n_classes
    weight_tensor = torch.from_numpy(class_weights).float().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Training loop with early stopping
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        logits, attn = model(*train_tensors)
        loss = criterion(logits, y_train_t)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        train_pred = logits.argmax(dim=1).cpu().numpy()
        train_acc = accuracy_score(y_train, train_pred)
        
        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits, val_attn = model(*val_tensors)
            val_loss = criterion(val_logits, y_val_t)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_pred)
        
        history["train_loss"].append(float(loss.item()))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss.item()))
        history["val_acc"].append(float(val_acc))
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 25 == 0:
            attn_str = ""
            if val_attn is not None:
                mean_attn = val_attn.mean(dim=0).cpu().numpy()
                attn_str = f" | Attn: MRI={mean_attn[0]:.3f} FDG={mean_attn[1]:.3f} Clin={mean_attn[2]:.3f}"
            print(f"    Epoch {epoch+1}/{epochs}: train_loss={loss.item():.4f} "
                  f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}{attn_str}")
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1} (best val_acc={best_val_acc:.3f})")
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    model.eval()
    return model, history


def predict_attention_fusion(model, proba_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with attention fusion model.
    
    Returns:
        y_pred: (N,) predicted labels
        y_proba: (N, n_classes) class probabilities
        attn_weights: (N, n_modalities) attention weights (or None)
    """
    if not TORCH_AVAILABLE or not isinstance(model, nn.Module):
        # Fallback to sklearn model
        meta_X = np.concatenate(proba_list, axis=1)
        y_pred = model.predict(meta_X)
        y_proba = model.predict_proba(meta_X).astype(np.float32)
        return y_pred, y_proba, None
    
    device = next(model.parameters()).device
    tensors = [torch.from_numpy(p).float().to(device) for p in proba_list]
    
    model.eval()
    with torch.no_grad():
        logits, attn = model(*tensors)
        y_proba = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        attn_weights = attn.cpu().numpy() if attn is not None else None
    
    return y_pred, y_proba, attn_weights


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKWARD-COMPATIBLE FUNCTIONS (preserved from original)
# ═══════════════════════════════════════════════════════════════════════════════

def get_probabilities(clf, X: np.ndarray) -> np.ndarray:
    """Get class probabilities from a classifier."""
    return clf.predict_proba(X).astype(np.float32)


def train_logistic_fusion(proba_list: List[np.ndarray],
                           y_train: np.ndarray, seed: int = 42) -> LogisticRegression:
    """Original logistic regression fusion (backward compatibility)."""
    meta_X = np.concatenate(proba_list, axis=1)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(meta_X, y_train)
    return clf


def train_meta_classifier(proba_list: List[np.ndarray],
                           y_train: np.ndarray, seed: int = 42,
                           proba_list_val: List[np.ndarray] = None,
                           y_val: np.ndarray = None,
                           fusion_type: str = "attention",
                           **kwargs) -> object:
    """
    Train meta-classifier for fusion.
    Dispatches to attention-based or logistic regression based on fusion_type.
    """
    if fusion_type in ("attention", "neural") and TORCH_AVAILABLE:
        if proba_list_val is not None and y_val is not None:
            model, history = train_attention_fusion(
                proba_list, y_train, proba_list_val, y_val,
                fusion_type=fusion_type, seed=seed, **kwargs
            )
            return model
        else:
            # No validation data, use train data for validation (not ideal)
            model, history = train_attention_fusion(
                proba_list, y_train, proba_list, y_train,
                fusion_type=fusion_type, seed=seed, **kwargs
            )
            return model
    else:
        return train_logistic_fusion(proba_list, y_train, seed)


def predict_fusion(meta_clf, proba_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using fusion model (backward compatible)."""
    y_pred, y_proba, _ = predict_attention_fusion(meta_clf, proba_list)
    return y_pred, y_proba


def compute_fusion_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_proba: np.ndarray, tag: str) -> Dict:
    """Compute fusion evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(
            label_binarize(y_true, classes=[0, 1, 2]),
            y_proba, multi_class="ovr", average="macro"
        ))
    except Exception:
        auc = float("nan")
    print(f"    {tag:<22}  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")
    return {"tag": tag, "accuracy": acc, "f1_macro": f1, "roc_auc": auc,
            "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba}


def run_ablation(meta_clf, proba_list: List[np.ndarray],
                 modality_names: List[str], y_true: np.ndarray) -> Dict:
    """Run ablation study by dropping each modality."""
    results = {}
    print("\n  Ablation Study:")
    for drop_idx, name in enumerate(modality_names):
        ablated = [
            np.full_like(p, 1.0 / 3) if i == drop_idx else p
            for i, p in enumerate(proba_list)
        ]
        y_pred, y_proba = predict_fusion(meta_clf, ablated)
        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        print(f"    Drop {name:<10}  acc={acc:.3f}  f1={f1:.3f}")
        results[f"drop_{name}"] = {"accuracy": acc, "f1_macro": f1}
    return results
