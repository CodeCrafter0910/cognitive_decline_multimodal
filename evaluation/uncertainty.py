"""
Uncertainty Quantification (Problem 21)

Methods:
1. Monte Carlo Dropout — Run multiple forward passes with dropout enabled
2. Ensemble Uncertainty — Standard deviation across multiple model predictions
3. Entropy-based — Prediction entropy as uncertainty measure

Provides:
- Per-sample confidence scores
- Uncertainty flags for borderline cases
- Calibrated uncertainty estimates
"""

import numpy as np
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def enable_mc_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    if not TORCH_AVAILABLE:
        return
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mc_dropout_uncertainty(model, input_tensor, n_forward: int = 30,
                            n_classes: int = 3) -> Dict:
    """
    Monte Carlo Dropout uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty via the variance of outputs.
    
    Args:
        model: Trained model with dropout layers
        input_tensor: Input tensor (1, C, D, H, W) or (1, F)
        n_forward: Number of forward passes
        n_classes: Number of output classes
    
    Returns:
        Dict with mean_prediction, std_prediction, entropy, uncertainty_score
    """
    if not TORCH_AVAILABLE:
        return {"mean_prediction": np.ones(n_classes) / n_classes,
                "uncertainty_score": 1.0}
    
    model.eval()
    enable_mc_dropout(model)  # Keep dropout active
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_forward):
            output = model(input_tensor)
            proba = F.softmax(output, dim=1)
            predictions.append(proba.cpu().numpy())
    
    predictions = np.array(predictions)  # (n_forward, batch, n_classes)
    
    # Mean prediction
    mean_pred = predictions.mean(axis=0)  # (batch, n_classes)
    
    # Standard deviation (epistemic uncertainty)
    std_pred = predictions.std(axis=0)    # (batch, n_classes)
    
    # Predictive entropy
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / max_entropy  # [0, 1]
    
    # Mutual information (epistemic uncertainty)
    expected_entropy = -np.mean(
        np.sum(predictions * np.log(predictions + 1e-10), axis=-1),
        axis=0
    )
    mutual_info = entropy - expected_entropy
    
    # Overall uncertainty score (0 = certain, 1 = uncertain)
    uncertainty_score = normalized_entropy
    
    model.eval()  # Reset to eval mode
    
    return {
        "mean_prediction": mean_pred.squeeze(),
        "std_prediction": std_pred.squeeze(),
        "entropy": entropy.squeeze(),
        "normalized_entropy": normalized_entropy.squeeze(),
        "mutual_information": mutual_info.squeeze(),
        "uncertainty_score": float(uncertainty_score.mean()),
        "all_predictions": predictions.squeeze(),
    }


def ensemble_uncertainty(predictions_list: List[np.ndarray]) -> Dict:
    """
    Compute uncertainty from an ensemble of model predictions.
    
    Args:
        predictions_list: List of (N, n_classes) probability arrays from different models
    
    Returns:
        Dict with mean, std, entropy, and uncertainty per sample
    """
    predictions = np.array(predictions_list)  # (n_models, N, n_classes)
    
    # Mean ensemble prediction
    mean_pred = predictions.mean(axis=0)  # (N, n_classes)
    
    # Standard deviation across models
    std_pred = predictions.std(axis=0)    # (N, n_classes)
    
    # Predictive entropy
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)  # (N,)
    max_entropy = np.log(mean_pred.shape[-1])
    
    # Per-sample uncertainty
    uncertainty_scores = entropy / max_entropy  # [0, 1]
    
    # Disagreement rate (fraction of models that disagree with majority)
    predicted_classes = predictions.argmax(axis=-1)  # (n_models, N)
    from scipy import stats
    mode_classes = stats.mode(predicted_classes, axis=0, keepdims=False)[0]
    disagreement = 1.0 - (predicted_classes == mode_classes[np.newaxis, :]).mean(axis=0)
    
    return {
        "mean_prediction": mean_pred,
        "std_prediction": std_pred,
        "entropy": entropy,
        "uncertainty_scores": uncertainty_scores,
        "disagreement_rate": disagreement,
        "predicted_class": mean_pred.argmax(axis=-1),
    }


def classify_uncertainty(uncertainty_scores: np.ndarray,
                          thresholds: tuple = (0.3, 0.6)) -> np.ndarray:
    """
    Classify predictions into confidence levels.
    
    Args:
        uncertainty_scores: (N,) uncertainty values [0, 1]
        thresholds: (low, high) thresholds
    
    Returns:
        confidence_levels: (N,) array of "High", "Medium", "Low"
    """
    low_thresh, high_thresh = thresholds
    
    levels = np.full(len(uncertainty_scores), "Medium", dtype=object)
    levels[uncertainty_scores < low_thresh] = "High"
    levels[uncertainty_scores > high_thresh] = "Low"
    
    return levels


def compute_uncertainty_report(y_true: np.ndarray, y_pred: np.ndarray,
                                uncertainty_scores: np.ndarray,
                                label_names: list = None) -> Dict:
    """
    Generate uncertainty analysis report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        uncertainty_scores: Per-sample uncertainty [0, 1]
        label_names: Class names
    
    Returns:
        Report dictionary with statistics
    """
    if label_names is None:
        label_names = ["CN", "MCI", "AD"]
    
    correct = y_true == y_pred
    
    report = {
        "overall": {
            "mean_uncertainty": float(uncertainty_scores.mean()),
            "std_uncertainty": float(uncertainty_scores.std()),
            "correct_mean_uncertainty": float(uncertainty_scores[correct].mean()) if correct.any() else 0,
            "incorrect_mean_uncertainty": float(uncertainty_scores[~correct].mean()) if (~correct).any() else 0,
        },
        "per_class": {},
        "confidence_calibration": {},
    }
    
    # Per-class uncertainty
    for cls_idx, cls_name in enumerate(label_names):
        cls_mask = y_true == cls_idx
        if cls_mask.any():
            report["per_class"][cls_name] = {
                "mean_uncertainty": float(uncertainty_scores[cls_mask].mean()),
                "accuracy": float(correct[cls_mask].mean()),
                "n_samples": int(cls_mask.sum()),
            }
    
    # Confidence calibration (accuracy at different uncertainty thresholds)
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
        confident_mask = uncertainty_scores < threshold
        if confident_mask.any():
            report["confidence_calibration"][f"uncertainty<{threshold}"] = {
                "n_samples": int(confident_mask.sum()),
                "accuracy": float(correct[confident_mask].mean()),
                "fraction": float(confident_mask.mean()),
            }
    
    return report


def print_uncertainty_report(report: Dict):
    """Print formatted uncertainty report."""
    print("\n  ── Uncertainty Analysis ──────────────────────────────────")
    
    overall = report["overall"]
    print(f"  Overall uncertainty:  {overall['mean_uncertainty']:.3f} ± {overall['std_uncertainty']:.3f}")
    print(f"  Correct predictions:  {overall['correct_mean_uncertainty']:.3f} (should be LOW)")
    print(f"  Incorrect predictions: {overall['incorrect_mean_uncertainty']:.3f} (should be HIGH)")
    
    gap = overall['incorrect_mean_uncertainty'] - overall['correct_mean_uncertainty']
    if gap > 0.05:
        print(f"  ✓ Good calibration: incorrect predictions have higher uncertainty (gap={gap:.3f})")
    else:
        print(f"  ⚠ Poor calibration: uncertainty doesn't distinguish correct/incorrect well")
    
    print("\n  Per-class uncertainty:")
    for cls_name, cls_data in report.get("per_class", {}).items():
        print(f"    {cls_name}: uncertainty={cls_data['mean_uncertainty']:.3f}, "
              f"accuracy={cls_data['accuracy']:.3f}, n={cls_data['n_samples']}")
    
    print("\n  Confidence calibration:")
    for threshold, data in report.get("confidence_calibration", {}).items():
        print(f"    {threshold}: {data['n_samples']} samples "
              f"({data['fraction']:.1%}), accuracy={data['accuracy']:.3f}")
    
    print("  ─────────────────────────────────────────────────────────")
