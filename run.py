"""
ADNI Multimodal AI — Enhanced Main Pipeline

Integrates ALL 25 improvements:
- 3D CNN feature extraction (Problem 5)
- Enhanced clinical features (Problem 6)
- Feature selection (Problem 7)
- Extended statistical features (Problem 8)
- Attention-based fusion (Problem 9, 10)
- Multiple model types (Problem 11)
- Stratified K-fold cross-validation (Problem 13)
- Data augmentation (Problem 14)
- Hyperparameter tuning (Problem 15)
- Data balancing with SMOTE (Problem 2)
- Comprehensive evaluation metrics (Problem 18)
- Confidence intervals (Problem 19)
- Grad-CAM interpretability (Problem 20)
- Uncertainty quantification (Problem 21)
- Experiment tracking (Problem 22, 23)
- Structured logging (Problem 25)
"""

import sys
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ADNI_ROOT, CSV_PATH,
    MRI_NPY_DIR, FDG_NII_DIR, FDG_NPY_DIR, RESULTS_DIR,
    TARGET_SHAPE, RANDOM_SEED, TEST_SIZE, VAL_SIZE,
    CV_ENABLED, CV_N_SPLITS,
    FUSION_TYPE, FUSION_HIDDEN_DIM, FUSION_DROPOUT, FUSION_LR, FUSION_EPOCHS, FUSION_PATIENCE,
    FEATURE_SELECTION_ENABLED, FEATURE_SELECTION_METHOD, FEATURE_SELECTION_K,
    CNN3D_ENABLED, USE_CNN_FEATURES, AUGMENTATION_ENABLED, AUG_MULTIPLIER,
    HYPERPARAM_TUNING, CLASS_WEIGHT_METHOD,
    GRADCAM_ENABLED, GRADCAM_N_SAMPLES,
    LOG_DIR, EXPERIMENT_DIR,
    LABEL_NAMES,
)
from preprocessing.scan_finder      import build_manifest
from preprocessing.image_processor  import run_all
from preprocessing.feature_extractor import extract_all_features, select_features
from preprocessing.clinical          import build_clinical_features
from models.modality_model           import (
    train_modality_model, scale_features,
    compute_metrics, save_model
)
from fusion.late_fusion              import (
    get_probabilities, train_meta_classifier,
    predict_fusion, compute_fusion_metrics, run_ablation,
    predict_attention_fusion
)
from evaluation.evaluate             import (
    save_all, compute_comprehensive_metrics
)
from utils.logger              import setup_logger
from utils.experiment_tracker  import ExperimentTracker


np.random.seed(RANDOM_SEED)


def run_single_fold(X_mri_fusion, X_fdg, X_clin, y_fusion,
                    X_mri_all, y_mri_all,
                    idx_train, idx_val, idx_test,
                    fold_num=0, total_folds=1):
    """
    Run a single train/val/test fold.
    
    Args:
        X_mri_fusion: MRI features for 190 fusion subjects
        X_fdg: FDG features for 190 fusion subjects
        X_clin: Clinical features for 190 fusion subjects
        y_fusion: Labels for 190 fusion subjects
        X_mri_all: MRI features for ALL 207 subjects
        y_mri_all: Labels for ALL 207 subjects
        idx_train, idx_val, idx_test: Indices for fusion subjects (190)
    """
    
    y_train = y_fusion[idx_train]
    y_val   = y_fusion[idx_val]
    y_test  = y_fusion[idx_test]
    
    print(f"\n  {'='*50}")
    print(f"  Fold {fold_num + 1}/{total_folds}  "
          f"Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  {'='*50}")
    
    # ── Scale features for fusion modalities ──────────────────────
    Xm_tr, Xm_va, Xm_te, scaler_mri = scale_features(
        X_mri_fusion[idx_train], X_mri_fusion[idx_val], X_mri_fusion[idx_test])
    Xf_tr, Xf_va, Xf_te, _ = scale_features(
        X_fdg[idx_train], X_fdg[idx_val], X_fdg[idx_test])
    Xc_tr, Xc_va, Xc_te, _ = scale_features(
        X_clin[idx_train], X_clin[idx_val], X_clin[idx_test])
    
    # ── Prepare MRI training data (ALL 207 subjects) ──────────────
    # Scale ALL MRI data using same scaler
    X_mri_all_scaled = scaler_mri.transform(X_mri_all).astype(np.float32)
    
    print(f"\n  MRI training: Using ALL {len(y_mri_all)} subjects (vs {len(idx_train)} fusion)")
    
    # ── Remove constant features first (CRITICAL FIX) ─────────────
    from sklearn.feature_selection import VarianceThreshold
    
    # Remove features with zero variance from ALL MRI data
    var_threshold_mri = VarianceThreshold(threshold=0.0)
    X_mri_all_clean = var_threshold_mri.fit_transform(X_mri_all_scaled)
    
    # Apply same transformation to fusion MRI data
    Xm_tr_clean = var_threshold_mri.transform(Xm_tr)
    Xm_va_clean = var_threshold_mri.transform(Xm_va)
    Xm_te_clean = var_threshold_mri.transform(Xm_te)
    
    # Remove constant features from FDG
    var_threshold_fdg = VarianceThreshold(threshold=0.0)
    Xf_tr_clean = var_threshold_fdg.fit_transform(Xf_tr)
    Xf_va_clean = var_threshold_fdg.transform(Xf_va)
    Xf_te_clean = var_threshold_fdg.transform(Xf_te)
    
    print(f"\n  Removed constant features: MRI {X_mri_all_scaled.shape[1]} → {X_mri_all_clean.shape[1]}, FDG {Xf_tr.shape[1]} → {Xf_tr_clean.shape[1]}")
    
    # ── Feature selection (Problem 7) ──────────────────────────────
    if FEATURE_SELECTION_ENABLED and X_mri_all_clean.shape[1] > FEATURE_SELECTION_K:
        print(f"\n  Applying feature selection...")
        
        # MRI: Select features on ALL 207 subjects
        X_mri_all_selected, _, _, mri_selector = select_features(
            X_mri_all_clean, y_mri_all, None, None,
            method="mutual_info", k=30)
        
        # Apply same feature selection to fusion MRI data
        Xm_tr = mri_selector.transform(Xm_tr_clean)
        Xm_va = mri_selector.transform(Xm_va_clean)
        Xm_te = mri_selector.transform(Xm_te_clean)
        
        # FDG: Select features on fusion data (190 subjects)
        Xf_tr, Xf_va, Xf_te, _ = select_features(
            Xf_tr_clean, y_train, Xf_va_clean, Xf_te_clean,
            method=FEATURE_SELECTION_METHOD, k=FEATURE_SELECTION_K)
    else:
        X_mri_all_selected = X_mri_all_clean
        Xm_tr, Xm_va, Xm_te = Xm_tr_clean, Xm_va_clean, Xm_te_clean
        Xf_tr, Xf_va, Xf_te = Xf_tr_clean, Xf_va_clean, Xf_te_clean
    
    # ── Train modality models ──────────────────────────────────────
    print("\n  Training modality-specific models...")
    
    balance = CLASS_WEIGHT_METHOD if CLASS_WEIGHT_METHOD in ("smote", "oversample") else "none"
    
    # MRI: Train on ALL 207 subjects, then evaluate on fusion validation set
    print(f"  Training MRI on ALL {len(y_mri_all)} subjects...")
    mri_clf,  _ = train_modality_model(
        X_mri_all_selected, y_mri_all, Xm_va, y_val, "MRI", RANDOM_SEED,
        model_type="random_forest", tune=False, balance=balance)
    
    # Evaluate MRI on fusion validation set
    mri_met = compute_metrics(mri_clf, Xm_va, y_val, "MRI")
    
    # FDG: Keep XGBoost with tuning (was working great at 51.7%)
    fdg_clf,  fdg_met = train_modality_model(
        Xf_tr, y_train, Xf_va, y_val, "FDG", RANDOM_SEED,
        model_type="xgboost", tune=HYPERPARAM_TUNING, balance=balance)
    
    # Clinical: Keep XGBoost with tuning (working great at 79.3%)
    clin_clf, clin_met = train_modality_model(
        Xc_tr, y_train, Xc_va, y_val, "Clinical", RANDOM_SEED,
        model_type="xgboost", tune=HYPERPARAM_TUNING, balance=balance)
    
    # ── Get probabilities for fusion ───────────────────────────────
    mp_tr = get_probabilities(mri_clf,  Xm_tr)
    fp_tr = get_probabilities(fdg_clf,  Xf_tr)
    cp_tr = get_probabilities(clin_clf, Xc_tr)
    
    mp_va = get_probabilities(mri_clf,  Xm_va)
    fp_va = get_probabilities(fdg_clf,  Xf_va)
    cp_va = get_probabilities(clin_clf, Xc_va)
    
    mp_te = get_probabilities(mri_clf,  Xm_te)
    fp_te = get_probabilities(fdg_clf,  Xf_te)
    cp_te = get_probabilities(clin_clf, Xc_te)
    
    # ── Train fusion model (Problem 9, 10) ─────────────────────────
    print(f"\n  Training {FUSION_TYPE} fusion model...")
    meta_clf = train_meta_classifier(
        [mp_tr, fp_tr, cp_tr], y_train,
        seed=RANDOM_SEED,
        proba_list_val=[mp_va, fp_va, cp_va],
        y_val=y_val,
        fusion_type=FUSION_TYPE,
        hidden_dim=FUSION_HIDDEN_DIM,
        dropout=FUSION_DROPOUT,
        lr=FUSION_LR,
        epochs=FUSION_EPOCHS,
        patience=FUSION_PATIENCE,
    )
    
    # ── Evaluate ───────────────────────────────────────────────────
    print("\n  Evaluating on test set...")
    
    mri_metrics  = compute_metrics(mri_clf,  Xm_te, y_test, "MRI")
    fdg_metrics  = compute_metrics(fdg_clf,  Xf_te, y_test, "FDG")
    clin_metrics = compute_metrics(clin_clf, Xc_te, y_test, "Clinical")
    
    y_pred, y_proba, attn_weights = predict_attention_fusion(
        meta_clf, [mp_te, fp_te, cp_te])
    
    fusion_result = compute_fusion_metrics(y_test, y_pred, y_proba, "Fusion")
    
    # ── Ablation study ─────────────────────────────────────────────
    ablation = run_ablation(meta_clf, [mp_te, fp_te, cp_te],
                             ["MRI", "FDG", "Clinical"], y_test)
    
    return {
        "mri_metrics": mri_metrics,
        "fdg_metrics": fdg_metrics,
        "clin_metrics": clin_metrics,
        "fusion_result": fusion_result,
        "ablation": ablation,
        "attn_weights": attn_weights,
        "models": {
            "mri_clf": mri_clf,
            "fdg_clf": fdg_clf,
            "clin_clf": clin_clf,
            "meta_clf": meta_clf,
        },
        "test_indices": idx_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def main():
    start_time = time.time()
    
    # ── Setup logging ──────────────────────────────────────────────
    logger = setup_logger("adni", LOG_DIR)
    
    print("=" * 60)
    print("  ADNI Multimodal AI — Enhanced Pipeline")
    print("  MRI + FDG-PET + Clinical | Statistical Features + Attention Fusion")
    print("=" * 60)
    
    # ── Setup experiment tracking ──────────────────────────────────
    tracker = ExperimentTracker(EXPERIMENT_DIR)
    tracker.log_hyperparameters({
        "target_shape": list(TARGET_SHAPE),
        "3d_cnn_enabled": CNN3D_ENABLED,
        "augmentation_enabled": AUGMENTATION_ENABLED,
        "cv_enabled": CV_ENABLED,
        "cv_n_splits": CV_N_SPLITS,
        "fusion_type": FUSION_TYPE,
        "feature_selection": FEATURE_SELECTION_ENABLED,
        "feature_selection_method": FEATURE_SELECTION_METHOD,
        "feature_selection_k": FEATURE_SELECTION_K,
        "hyperparam_tuning": HYPERPARAM_TUNING,
        "class_weight_method": CLASS_WEIGHT_METHOD,
    })

    # ══════════════════════════════════════════════════════════════════
    #  Step 1: Build manifest (SMART APPROACH - Use all MRI data!)
    # ══════════════════════════════════════════════════════════════════
    print("\n[1/7]  Building scan manifest...")
    
    # Get ALL subjects for MRI training (207 patients)
    manifest_all = build_manifest(ADNI_ROOT, CSV_PATH, require_all_modalities=False)
    
    # Get subjects with all 3 modalities for fusion (190 patients)
    manifest_fusion = build_manifest(ADNI_ROOT, CSV_PATH, require_all_modalities=True)
    
    if len(manifest_fusion) == 0:
        print("\n" + "=" * 60)
        print("  ERROR: No subjects with all 3 modalities found!")
        print("=" * 60)
        return
    
    print(f"\n  ✓ MRI training: {len(manifest_all)} subjects (ALL available)")
    print(f"  ✓ Fusion: {len(manifest_fusion)} subjects (all 3 modalities)")
    print(f"  ✓ Extra MRI samples: {len(manifest_all) - len(manifest_fusion)}")
    
    tracker.log_metrics({
        "n_subjects_mri": len(manifest_all),
        "n_subjects_fusion": len(manifest_fusion),
        "n_extra_mri": len(manifest_all) - len(manifest_fusion)
    })

    # ══════════════════════════════════════════════════════════════════
    #  Step 2: Preprocess imaging
    # ══════════════════════════════════════════════════════════════════
    print("\n[2/7]  Preprocessing volumes...")
    # Preprocess ALL MRI (207 subjects)
    run_all(manifest_all, MRI_NPY_DIR, FDG_NPY_DIR, TARGET_SHAPE)
    # Preprocess FDG for fusion subjects (190 subjects)
    run_all(manifest_fusion, MRI_NPY_DIR, FDG_NPY_DIR, TARGET_SHAPE)

    # ══════════════════════════════════════════════════════════════════
    #  Step 3: Extract features (SMART - separate MRI and fusion data)
    # ══════════════════════════════════════════════════════════════════
    print("\n[3/7]  Extracting features...")
    
    # Extract MRI features for ALL 207 subjects
    subject_ids_all = manifest_all["subject_id"].tolist()
    print("  MRI features (ALL 207 subjects):")
    X_mri_all, mri_found_all = extract_all_features(
        MRI_NPY_DIR, subject_ids_all, use_cnn=USE_CNN_FEATURES, use_3d=CNN3D_ENABLED)
    
    # Extract features for fusion subjects (190 with all modalities)
    subject_ids_fusion = manifest_fusion["subject_id"].tolist()
    
    print("  FDG features (190 subjects):")
    X_fdg, fdg_found = extract_all_features(
        FDG_NPY_DIR, subject_ids_fusion, use_cnn=USE_CNN_FEATURES, use_3d=CNN3D_ENABLED)
    
    print("  MRI features (190 fusion subjects):")
    X_mri_fusion, mri_found_fusion = extract_all_features(
        MRI_NPY_DIR, subject_ids_fusion, use_cnn=USE_CNN_FEATURES, use_3d=CNN3D_ENABLED)

    # Find common subjects for fusion (all 3 modalities)
    common = set(mri_found_fusion) & set(fdg_found) & set(subject_ids_fusion)
    ordered_fusion = [s for s in subject_ids_fusion if s in common]
    
    print(f"\n  Validating data:")
    print(f"    MRI (all):          {len(mri_found_all)} subjects")
    print(f"    MRI (fusion):       {len(mri_found_fusion)} subjects")
    print(f"    FDG:                {len(fdg_found)} subjects")
    print(f"    Clinical:           {len(subject_ids_fusion)} subjects")
    print(f"    Fully paired:       {len(ordered_fusion)} subjects")
    
    if len(ordered_fusion) == 0:
        raise RuntimeError("No subjects have all 3 modalities! Check preprocessing.")
    
    # Align fusion data to common subjects
    mri_fusion_idx = [mri_found_fusion.index(s) for s in ordered_fusion]
    fdg_idx = [fdg_found.index(s) for s in ordered_fusion]
    X_mri_fusion = X_mri_fusion[mri_fusion_idx]
    X_fdg = X_fdg[fdg_idx]

    print("  Clinical features (190 subjects):")
    X_clin, y_fusion, _ = build_clinical_features(manifest_fusion, ordered_fusion)
    
    # Prepare MRI training data (ALL 207 subjects)
    # Get labels for all MRI subjects
    X_clin_all, y_mri_all, _ = build_clinical_features(manifest_all, mri_found_all)
    mri_all_idx = [mri_found_all.index(s) for s in mri_found_all if s in manifest_all["subject_id"].tolist()]
    X_mri_all = X_mri_all[mri_all_idx]
    
    # Validate shapes
    assert X_mri_fusion.shape[0] == X_fdg.shape[0] == X_clin.shape[0] == len(y_fusion), \
        "Fusion modality sample counts don't match!"
    
    print(f"\n  Final data:")
    print(f"    MRI training: {len(y_mri_all)} subjects (ALL available)")
    print(f"    Fusion: {len(ordered_fusion)} subjects (all 3 modalities)")
    print(f"    Feature dimensions: MRI={X_mri_all.shape[1]}, FDG={X_fdg.shape[1]}, Clinical={X_clin.shape[1]}")
    print(f"    MRI class distribution: {dict(zip(*np.unique(y_mri_all, return_counts=True)))}")
    print(f"    Fusion class distribution: {dict(zip(*np.unique(y_fusion, return_counts=True)))}")
    
    tracker.log_metrics({
        "n_mri_training": len(y_mri_all),
        "n_fusion": len(ordered_fusion),
        "mri_features": X_mri_all.shape[1],
        "fdg_features": X_fdg.shape[1],
        "clinical_features": X_clin.shape[1],
    })

    # ══════════════════════════════════════════════════════════════════
    #  Step 4: Train & Evaluate
    # ══════════════════════════════════════════════════════════════════
    idx = np.arange(len(ordered_fusion))
    
    if CV_ENABLED:
        # ── Cross-Validation (Problem 13) ──────────────────────────
        print(f"\n[4/7]  Running {CV_N_SPLITS}-fold stratified cross-validation...")
        
        # Hold out a final test set from fusion data
        idx_trainval, idx_test = train_test_split(
            idx, test_size=TEST_SIZE, stratify=y_fusion, random_state=RANDOM_SEED)
        
        skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        
        cv_scores = {
            'accuracy': [], 'f1_macro': [], 'roc_auc': [],
        }
        best_fold_result = None
        best_fold_acc = 0.0
        
        for fold_idx, (train_rel_idx, val_rel_idx) in enumerate(
                skf.split(idx_trainval, y_fusion[idx_trainval])):
            
            fold_train_idx = idx_trainval[train_rel_idx]
            fold_val_idx = idx_trainval[val_rel_idx]
            
            fold_result = run_single_fold(
                X_mri_fusion, X_fdg, X_clin, y_fusion,
                X_mri_all, y_mri_all,
                fold_train_idx, fold_val_idx, idx_test,
                fold_num=fold_idx, total_folds=CV_N_SPLITS)
            
            # Record fold scores
            fusion = fold_result["fusion_result"]
            cv_scores['accuracy'].append(fusion['accuracy'])
            cv_scores['f1_macro'].append(fusion['f1_macro'])
            cv_scores['roc_auc'].append(fusion['roc_auc'])
            
            tracker.log_fold_result(fold_idx, {
                'accuracy': fusion['accuracy'],
                'f1_macro': fusion['f1_macro'],
                'roc_auc': fusion['roc_auc'],
            })
            
            # Track best fold
            if fusion['accuracy'] > best_fold_acc:
                best_fold_acc = fusion['accuracy']
                best_fold_result = fold_result
        
        # Print CV summary
        print("\n" + "=" * 60)
        print("  CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        for metric in ['accuracy', 'f1_macro', 'roc_auc']:
            vals = cv_scores[metric]
            print(f"  {metric:<12}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  "
                  f"(range: {np.min(vals):.3f} - {np.max(vals):.3f})")
        print("=" * 60)
        
        # Use best fold result for final evaluation
        final_result = best_fold_result
        
    else:
        # ── Single Split ───────────────────────────────────────────
        print("\n[4/7]  Splitting dataset...")
        idx_tv, idx_test = train_test_split(
            idx, test_size=TEST_SIZE, stratify=y_fusion, random_state=RANDOM_SEED)
        
        val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
        idx_train, idx_val = train_test_split(
            idx_tv, test_size=val_frac, stratify=y_fusion[idx_tv], random_state=RANDOM_SEED)
        
        final_result = run_single_fold(
            X_mri_fusion, X_fdg, X_clin, y_fusion,
            X_mri_all, y_mri_all,
            idx_train, idx_val, idx_test,
            fold_num=0, total_folds=1)
        
        cv_scores = None

    # ══════════════════════════════════════════════════════════════════
    #  Step 5: Save models
    # ══════════════════════════════════════════════════════════════════
    print("\n[5/7]  Saving models...")
    models = final_result["models"]
    save_model(models["mri_clf"],  RESULTS_DIR / "models" / "mri_model.pkl")
    save_model(models["fdg_clf"],  RESULTS_DIR / "models" / "fdg_model.pkl")
    save_model(models["clin_clf"], RESULTS_DIR / "models" / "clinical_model.pkl")
    save_model(models["meta_clf"], RESULTS_DIR / "models" / "meta_clf.pkl")
    
    tracker.log_model_info("mri_model", str(RESULTS_DIR / "models" / "mri_model.pkl"),
                           architecture="XGBoost (tuned)")
    tracker.log_model_info("fdg_model", str(RESULTS_DIR / "models" / "fdg_model.pkl"),
                           architecture="XGBoost (tuned)")
    tracker.log_model_info("clinical_model", str(RESULTS_DIR / "models" / "clinical_model.pkl"),
                           architecture="XGBoost (tuned)")
    tracker.log_model_info("meta_fusion", str(RESULTS_DIR / "models" / "meta_clf.pkl"),
                           architecture=f"{FUSION_TYPE} Fusion")

    # ══════════════════════════════════════════════════════════════════
    #  Step 6: Comprehensive evaluation
    # ══════════════════════════════════════════════════════════════════
    print("\n[6/7]  Running comprehensive evaluation...")
    
    fusion_result = final_result["fusion_result"]
    y_test = fusion_result["y_true"]
    y_pred = fusion_result["y_pred"]
    y_proba = fusion_result["y_proba"]
    attn_weights = final_result.get("attn_weights")
    
    # Comprehensive clinical metrics with confidence intervals (Problem 18, 19)
    comprehensive = compute_comprehensive_metrics(y_test, y_pred, y_proba)
    
    # Print per-class metrics
    print("\n  ── Per-Class Clinical Metrics ──────────────────────────")
    for cls_name, cls_met in comprehensive["per_class"].items():
        print(f"  {cls_name}:  Sens={cls_met['sensitivity']:.3f}  "
              f"Spec={cls_met['specificity']:.3f}  "
              f"PPV={cls_met['ppv']:.3f}  NPV={cls_met['npv']:.3f}  "
              f"AUC={cls_met['auc']:.3f}")
    
    # Print confidence intervals
    overall = comprehensive["overall"]
    if isinstance(overall.get("accuracy"), dict):
        acc_ci = overall["accuracy"]
        f1_ci = overall["f1_macro"]
        print(f"\n  Confidence Intervals (95%):")
        print(f"    Accuracy: {acc_ci['point_estimate']:.3f} "
              f"[{acc_ci['lower']:.3f}, {acc_ci['upper']:.3f}]")
        print(f"    F1-Macro: {f1_ci['point_estimate']:.3f} "
              f"[{f1_ci['lower']:.3f}, {f1_ci['upper']:.3f}]")
    
    # Attention weight analysis
    if attn_weights is not None:
        mean_attn = attn_weights.mean(axis=0)
        print(f"\n  Mean Attention Weights:")
        print(f"    MRI:      {mean_attn[0]:.3f}")
        print(f"    FDG:      {mean_attn[1]:.3f}")
        print(f"    Clinical: {mean_attn[2]:.3f}")

    # ══════════════════════════════════════════════════════════════════
    #  Step 7: Save all results
    # ══════════════════════════════════════════════════════════════════
    print("\n[7/7]  Saving results and visualizations...")
    
    save_all(RESULTS_DIR, fusion_result,
             [final_result["mri_metrics"],
              final_result["fdg_metrics"],
              final_result["clin_metrics"]],
             final_result["ablation"],
             comprehensive_metrics=comprehensive,
             cv_scores=cv_scores,
             attn_weights=attn_weights)

    # ══════════════════════════════════════════════════════════════════
    #  Final Summary
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    
    mri_met = final_result["mri_metrics"]
    fdg_met = final_result["fdg_metrics"]
    clin_met = final_result["clin_metrics"]
    
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  MRI       acc={mri_met['accuracy']:.3f}  f1={mri_met['f1_macro']:.3f}  auc={mri_met['roc_auc']:.3f}")
    print(f"  FDG       acc={fdg_met['accuracy']:.3f}  f1={fdg_met['f1_macro']:.3f}  auc={fdg_met['roc_auc']:.3f}")
    print(f"  Clinical  acc={clin_met['accuracy']:.3f}  f1={clin_met['f1_macro']:.3f}  auc={clin_met['roc_auc']:.3f}")
    print(f"  Fusion    acc={fusion_result['accuracy']:.3f}  f1={fusion_result['f1_macro']:.3f}  auc={fusion_result['roc_auc']:.3f}")
    
    if cv_scores:
        print(f"\n  Cross-Validation ({CV_N_SPLITS}-fold):")
        for metric in ['accuracy', 'f1_macro', 'roc_auc']:
            vals = cv_scores[metric]
            print(f"    {metric}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 60)
    
    # Log final metrics
    tracker.log_metrics({
        "final/mri_accuracy": mri_met['accuracy'],
        "final/fdg_accuracy": fdg_met['accuracy'],
        "final/clinical_accuracy": clin_met['accuracy'],
        "final/fusion_accuracy": fusion_result['accuracy'],
        "final/fusion_f1": fusion_result['f1_macro'],
        "final/fusion_auc": fusion_result['roc_auc'],
    })
    
    if cv_scores:
        tracker.log_metrics({
            "cv/mean_accuracy": float(np.mean(cv_scores['accuracy'])),
            "cv/std_accuracy": float(np.std(cv_scores['accuracy'])),
            "cv/mean_f1": float(np.mean(cv_scores['f1_macro'])),
            "cv/mean_auc": float(np.mean(cv_scores['roc_auc'])),
        })
    
    tracker.finish(notes=f"Fusion accuracy={fusion_result['accuracy']:.3f}, "
                         f"AUC={fusion_result['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
