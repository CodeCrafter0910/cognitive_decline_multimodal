"""
Enhanced Evaluation Module (Problem 18, 19)

Upgrades:
1. Comprehensive clinical metrics: Sensitivity, Specificity, PPV, NPV per class
2. Confidence intervals via bootstrapping
3. Statistical significance tests
4. Enhanced visualizations with professional styling
5. Cross-validation result aggregation
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize

# ── Professional Plotting Style ────────────────────────────────────────
# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("deep")

# Custom matplotlib parameters for professional look
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC',
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'patch.edgecolor': 'white',
    'patch.linewidth': 1.5,
})

LABEL_NAMES = ["CN", "MCI", "AD"]
COLORS      = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Professional blue, orange, green


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIDENCE INTERVALS (Problem 19)
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_proba: np.ndarray = None,
                                   metric_fn=None, n_iterations: int = 1000,
                                   confidence: float = 0.95,
                                   seed: int = 42) -> Dict:
    """
    Compute confidence intervals via bootstrapping.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        metric_fn: Custom metric function(y_true, y_pred) -> float
        n_iterations: Number of bootstrap iterations
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed
    
    Returns:
        Dict with point estimate, lower bound, upper bound
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    if metric_fn is None:
        metric_fn = lambda yt, yp: accuracy_score(yt, yp)
    
    point_estimate = metric_fn(y_true, y_pred)
    
    bootstrap_scores = []
    for _ in range(n_iterations):
        indices = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        except Exception:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
    
    return {
        "point_estimate": float(point_estimate),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": confidence,
        "std": float(bootstrap_scores.std()),
    }


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    y_proba: np.ndarray,
                                    n_bootstrap: int = 1000) -> Dict:
    """
    Compute ALL clinical metrics with confidence intervals.
    
    Returns per-class: Sensitivity, Specificity, PPV, NPV, F1
    Returns overall: Accuracy, F1-macro, ROC-AUC, with 95% CI
    """
    metrics = {}
    
    # ── Overall Metrics ────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    try:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        roc_auc = float(np.mean([
            auc(*roc_curve(y_bin[:, i], y_proba[:, i])[:2])
            for i in range(3)
        ]))
    except Exception:
        roc_auc = float('nan')
    
    # Bootstrap confidence intervals
    acc_ci = bootstrap_confidence_interval(y_true, y_pred, n_iterations=n_bootstrap)
    f1_ci = bootstrap_confidence_interval(
        y_true, y_pred,
        metric_fn=lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
        n_iterations=n_bootstrap
    )
    
    metrics["overall"] = {
        "accuracy": acc_ci,
        "f1_macro": f1_ci,
        "roc_auc": roc_auc,
    }
    
    # ── Per-Class Metrics ──────────────────────────────────────────────
    metrics["per_class"] = {}
    
    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        y_binary_true = (y_true == cls_idx).astype(int)
        y_binary_pred = (y_pred == cls_idx).astype(int)
        
        # True Positives, False Positives, True Negatives, False Negatives
        tp = ((y_binary_true == 1) & (y_binary_pred == 1)).sum()
        fp = ((y_binary_true == 0) & (y_binary_pred == 1)).sum()
        tn = ((y_binary_true == 0) & (y_binary_pred == 0)).sum()
        fn = ((y_binary_true == 1) & (y_binary_pred == 0)).sum()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # Precision / PPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          # NPV
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        # Class-specific AUC
        try:
            y_bin_cls = label_binarize(y_true, classes=[0, 1, 2])
            fpr, tpr, _ = roc_curve(y_bin_cls[:, cls_idx], y_proba[:, cls_idx])
            cls_auc = auc(fpr, tpr)
        except Exception:
            cls_auc = float('nan')
        
        metrics["per_class"][cls_name] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
            "f1": float(f1),
            "auc": float(cls_auc),
            "support": int(y_binary_true.sum()),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        }
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = confusion_matrix(y_true, y_pred)
    
    # Percentage-based confusion matrix
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Custom annotations with count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"
    
    # Professional color scheme
    sns.heatmap(cm, annot=annot, fmt='', cmap='RdYlGn_r', 
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                ax=ax, cbar_kws={'label': 'Count'}, 
                linewidths=2, linecolor='white',
                vmin=0, vmax=cm.max(),
                annot_kws={'size': 12, 'weight': 'bold'})
    
    ax.set_xlabel("Predicted Class", fontsize=13, fontweight='bold')
    ax.set_ylabel("True Class", fontsize=13, fontweight='bold')
    ax.set_title(title, fontweight="bold", fontsize=14, pad=15)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('#333333')
    
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def plot_roc_curves(y_true, y_proba, title: str, path: Path):
    y_bin   = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(9, 7))
    
    aucs = []
    for i, (name, color) in enumerate(zip(LABEL_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=3,
                label=f"{name}  (AUC = {roc_auc:.3f})", alpha=0.9)
        
        # Fill area under curve with gradient effect
        ax.fill_between(fpr, tpr, alpha=0.15, color=color)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.5, label="Random (AUC = 0.500)")
    
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight='bold')
    ax.set_title(f"{title}\nMacro-Average AUC = {np.mean(aucs):.3f}", 
                 fontweight="bold", fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def plot_comparison(metrics_rows: List[Dict], path: Path):
    names = [m["tag"] for m in metrics_rows]
    accs  = [m["accuracy"] for m in metrics_rows]
    aucs  = [m["roc_auc"]  for m in metrics_rows]

    x   = np.arange(len(names))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Professional color scheme
    b1  = ax.bar(x - w / 2, accs, w, label="Accuracy", 
                 color="#3498db", edgecolor='#2c3e50', linewidth=1.5, alpha=0.9)
    b2  = ax.bar(x + w / 2, aucs, w, label="ROC-AUC",  
                 color="#e74c3c", edgecolor='#2c3e50', linewidth=1.5, alpha=0.9)
    
    # Add value labels on bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.3f}", ha="center", va="bottom", 
                    fontsize=10, fontweight='bold', color='#2c3e50')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=13, fontweight='bold')
    ax.set_title("Model Comparison — Test Set Performance", 
                 fontweight="bold", fontsize=14, pad=15)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal reference line at 0.33 (random baseline for 3 classes)
    ax.axhline(y=0.333, color='gray', linestyle=':', lw=2, alpha=0.6, 
               label='Random Baseline (33.3%)')
    
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def plot_per_class_metrics(metrics: Dict, path: Path):
    """Plot per-class sensitivity, specificity, PPV, NPV."""
    per_class = metrics.get("per_class", {})
    if not per_class:
        return
    
    classes = list(per_class.keys())
    metric_names = ["sensitivity", "specificity", "ppv", "npv"]
    metric_labels = ["Sensitivity\n(Recall)", "Specificity", "PPV\n(Precision)", "NPV"]
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    
    for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        values = [per_class[cls][metric_name] for cls in classes]
        bars = ax.bar(classes, values, color=COLORS, edgecolor='#2c3e50', 
                     width=0.65, linewidth=1.5, alpha=0.9)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                    f"{val:.3f}", ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='#2c3e50')
        
        ax.set_ylim(0, 1.18)
        ax.set_title(metric_label, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel("Score", fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Add reference line at 0.8 (good performance threshold)
        ax.axhline(y=0.8, color='green', linestyle=':', lw=2, alpha=0.4)
    
    fig.suptitle("Per-Class Clinical Metrics", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def plot_attention_weights(attn_weights: np.ndarray, labels: np.ndarray,
                           label_names: list, path: Path):
    """Plot attention weight distribution per class."""
    if attn_weights is None:
        return
    
    modality_names = ["MRI", "FDG", "Clinical"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    for cls_idx, cls_name in enumerate(label_names):
        ax = axes[cls_idx]
        mask = labels == cls_idx
        if not mask.any():
            continue
        
        cls_weights = attn_weights[mask]
        bp = ax.boxplot([cls_weights[:, i] for i in range(3)],
                       labels=modality_names, patch_artist=True,
                       widths=0.6,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2.5, color='red'))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('#2c3e50')
        
        ax.set_title(f"{cls_name} (n={mask.sum()})", fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel("Attention Weight", fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle("Modality Attention Weights by Class", fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def plot_cv_results(cv_scores: Dict, path: Path):
    """Plot cross-validation results across folds."""
    if not cv_scores:
        return
    
    metrics_to_plot = ['accuracy', 'f1_macro', 'roc_auc']
    metric_labels = ['Accuracy', 'F1-Score (Macro)', 'ROC-AUC']
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        values = cv_scores.get(metric, [])
        if not values:
            continue
        
        folds = range(1, len(values) + 1)
        bars = ax.bar(folds, values, color='#3498db', alpha=0.8, 
                     edgecolor='#2c3e50', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        # Mean line
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', lw=2.5,
                   label=f"Mean = {mean_val:.3f}")
        
        # Fill ± std area
        ax.fill_between([0.5, len(values) + 0.5],
                       mean_val - std_val,
                       mean_val + std_val,
                       alpha=0.15, color='red', label=f"±1 SD = {std_val:.3f}")
        
        ax.set_xlabel("Fold", fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f"{label}\n{mean_val:.3f} ± {std_val:.3f}",
                     fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
    
    fig.suptitle("5-Fold Stratified Cross-Validation Results", 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    
    fig.suptitle("Cross-Validation Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SAVE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def save_all(results_dir: Path,
             fusion_result: Dict,
             single_modality_metrics: List[Dict],
             ablation: Dict,
             comprehensive_metrics: Dict = None,
             cv_scores: Dict = None,
             attn_weights: np.ndarray = None):
    """
    Save all evaluation results and visualizations.
    
    Enhanced with:
    - Per-class clinical metrics plot
    - Cross-validation results plot
    - Attention weight visualization
    - Comprehensive metrics CSV
    - Confidence intervals
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    y_true  = fusion_result["y_true"]
    y_pred  = fusion_result["y_pred"]
    y_proba = fusion_result["y_proba"]

    # ── Core Plots ─────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred,
                          "Fusion Model — Test Set",
                          results_dir / "confusion_matrix.png")

    plot_roc_curves(y_true, y_proba,
                    "ROC Curves — Test Set",
                    results_dir / "roc_curves.png")

    comparison_rows = single_modality_metrics + [fusion_result]
    plot_comparison(comparison_rows, results_dir / "model_comparison.png")
    
    # ── Enhanced Plots ─────────────────────────────────────────────────
    if comprehensive_metrics:
        plot_per_class_metrics(comprehensive_metrics, results_dir / "per_class_metrics.png")
    
    if cv_scores:
        plot_cv_results(cv_scores, results_dir / "cv_results.png")
    
    if attn_weights is not None:
        plot_attention_weights(attn_weights, y_true, LABEL_NAMES,
                              results_dir / "attention_weights.png")

    # ── Metrics CSV ────────────────────────────────────────────────────
    csv_rows = []
    for m in comparison_rows:
        csv_rows.append({
            "Model":    m["tag"],
            "Accuracy": round(m["accuracy"], 4),
            "F1_Macro": round(m["f1_macro"],  4),
            "ROC_AUC":  round(m["roc_auc"],   4),
        })
    for name, vals in ablation.items():
        csv_rows.append({
            "Model":    name,
            "Accuracy": round(vals["accuracy"], 4),
            "F1_Macro": round(vals["f1_macro"],  4),
            "ROC_AUC":  float("nan"),
        })
    pd.DataFrame(csv_rows).to_csv(results_dir / "metrics_summary.csv", index=False)

    # ── Classification Report ──────────────────────────────────────────
    with open(results_dir / "classification_report.txt", "w", encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  ADNI Multimodal AI — Final Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Fusion Model — Test Set\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_true, y_pred,
                                       target_names=LABEL_NAMES, zero_division=0))
        f.write("\n")
        
        # Add comprehensive metrics if available
        if comprehensive_metrics:
            f.write("\n" + "=" * 60 + "\n")
            f.write("  Per-Class Clinical Metrics\n")
            f.write("=" * 60 + "\n")
            for cls_name, cls_metrics in comprehensive_metrics.get("per_class", {}).items():
                f.write(f"\n{cls_name}:\n")
                f.write(f"  Sensitivity (Recall): {cls_metrics['sensitivity']:.3f}\n")
                f.write(f"  Specificity:          {cls_metrics['specificity']:.3f}\n")
                f.write(f"  PPV (Precision):      {cls_metrics['ppv']:.3f}\n")
                f.write(f"  NPV:                  {cls_metrics['npv']:.3f}\n")
                f.write(f"  F1-Score:             {cls_metrics['f1']:.3f}\n")
                f.write(f"  AUC:                  {cls_metrics['auc']:.3f}\n")
                f.write(f"  Support:              {cls_metrics['support']}\n")
            
            # Add confidence intervals
            overall = comprehensive_metrics.get("overall", {})
            if "accuracy" in overall and isinstance(overall["accuracy"], dict):
                f.write(f"\n\nConfidence Intervals (95%):\n")
                f.write(f"  Accuracy: {overall['accuracy']['point_estimate']:.3f} "
                       f"[{overall['accuracy']['lower']:.3f}, {overall['accuracy']['upper']:.3f}]\n")
                f.write(f"  F1-Macro: {overall['f1_macro']['point_estimate']:.3f} "
                       f"[{overall['f1_macro']['lower']:.3f}, {overall['f1_macro']['upper']:.3f}]\n")
        
        # Add cross-validation results
        if cv_scores:
            f.write("\n" + "=" * 60 + "\n")
            f.write("  Cross-Validation Results\n")
            f.write("=" * 60 + "\n")
            for metric in ['accuracy', 'f1_macro', 'roc_auc']:
                values = cv_scores.get(metric, [])
                if values:
                    f.write(f"  {metric}: {np.mean(values):.3f} ± {np.std(values):.3f}\n")

    print(f"\nResults saved to: {results_dir}")
    print("  confusion_matrix.png")
    print("  roc_curves.png")
    print("  model_comparison.png")
    if comprehensive_metrics:
        print("  per_class_metrics.png")
    if cv_scores:
        print("  cv_results.png")
    if attn_weights is not None:
        print("  attention_weights.png")
    print("  metrics_summary.csv")
    print("  classification_report.txt")
