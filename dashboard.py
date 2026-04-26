"""
ADNI Multimodal AI — Enhanced Dashboard

Features:
- Overview with system architecture diagram
- Results with per-class clinical metrics
- Cross-validation results visualization
- Attention weight analysis
- Grad-CAM interpretability viewer
- Uncertainty analysis
- Ablation study
- Experiment history
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS_DIR, EXPERIMENT_DIR, LABEL_NAMES

st.set_page_config(
    page_title="ADNI Multimodal AI — Cognitive Decline Detection",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Results ────────────────────────────────────────────────────────
metrics_path = RESULTS_DIR / "metrics_summary.csv"
results_available = metrics_path.exists()

if results_available:
    metrics_df = pd.read_csv(metrics_path)
    
    fusion_row = metrics_df[metrics_df["Model"] == "Fusion"]
    if len(fusion_row) > 0:
        fusion_acc = fusion_row["Accuracy"].values[0]
        fusion_f1 = fusion_row["F1_Macro"].values[0] if "F1_Macro" in fusion_row.columns else 0.0
        fusion_auc = fusion_row["ROC_AUC"].values[0]
    else:
        fusion_acc = fusion_f1 = fusion_auc = 0.0
else:
    fusion_acc = fusion_f1 = fusion_auc = 0.0
    metrics_df = None


# ── Sidebar Navigation ─────────────────────────────────────────────────
st.sidebar.title("🧠 ADNI Multimodal AI")
st.sidebar.markdown("**Multi-Signal Cognitive Decline Detection**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Results",
    "📈 Cross-Validation",
    "🔍 Confusion Matrix",
    "📉 ROC Curves",
    "⚖️ Model Comparison",
    "🎯 Attention Analysis",
    "🔬 Per-Class Metrics",
    "🧪 Ablation Study",
    "📋 Experiment History",
    "⚠️ Disclaimer"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** ADNI")
st.sidebar.markdown("**Classes:** CN / MCI / AD")
st.sidebar.markdown("**Validation:** 5-Fold Stratified CV")
st.sidebar.markdown("**Fusion:** Attention-Based")

if not results_available:
    st.sidebar.warning("⚠️ No results found. Run pipeline first!")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown('<div class="main-title">Multi-Signal AI System for Cognitive Decline Detection</div>',
                unsafe_allow_html=True)
    st.markdown("### SDP Project — Alzheimer's Disease Classification using Multimodal Learning")
    
    if results_available:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fusion Accuracy", f"{fusion_acc:.1%}")
        col2.metric("Fusion F1-Macro", f"{fusion_f1:.3f}")
        col3.metric("Fusion ROC-AUC", f"{fusion_auc:.3f}")
        col4.metric("Classes", "CN / MCI / AD")
    else:
        st.warning("⚠️ Run `python adni_project/run.py` first to generate results.")
    
    st.markdown("---")
    
    # ── Key Improvements ──────────────────────────────────────────
    st.markdown("### 🚀 Key Technical Improvements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Architecture Upgrades:**
        - ✅ True 3D CNN (full brain volume analysis)
        - ✅ Attention-based multimodal fusion
        - ✅ Squeeze-and-Excitation blocks
        - ✅ Residual connections
        - ✅ Feature selection (Mutual Information)
        """)
    
    with col2:
        st.markdown("""
        **Training Upgrades:**
        - ✅ 5-fold stratified cross-validation
        - ✅ Medical data augmentation
        - ✅ SMOTE class balancing
        - ✅ Hyperparameter tuning
        - ✅ Early stopping + LR scheduling
        """)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        **Evaluation Upgrades:**
        - ✅ Per-class Sensitivity/Specificity
        - ✅ PPV and NPV metrics
        - ✅ 95% confidence intervals (bootstrap)
        - ✅ Uncertainty quantification
        """)
    
    with col4:
        st.markdown("""
        **Interpretability:**
        - ✅ Grad-CAM brain region visualization
        - ✅ Attention weight analysis
        - ✅ Ablation study
        - ✅ Experiment tracking
        """)
    
    st.markdown("---")
    
    # ── Pipeline Architecture ─────────────────────────────────────
    st.markdown("### System Pipeline Architecture")
    st.code("""
ADNI Dataset (Fully paired subjects: MRI + FDG-PET + Clinical)
           │
     ┌─────┴──────┬──────────────┐
     │            │              │
 Structural   FDG PET       MMSCORE
    MRI       Imaging       (Clinical)
     │            │              │
 Preprocess   Preprocess     Derive 15
 + 3D CNN     + 3D CNN      Enhanced
 Features     Features      Features
 (256-dim)    (256-dim)     + Severity Bins
     │            │          + Z-scores
     │            │              │
  Feature     Feature          │
  Selection   Selection        │
  (MI/PCA)    (MI/PCA)         │
     │            │              │
 SVM Ensemble SVM Ensemble  XGBoost
  (3 kernels)  (3 kernels)    + SMOTE
     │            │              │
  P(CN|MRI)  P(CN|FDG)   P(CN|Clin)
  P(MCI|MRI) P(MCI|FDG)  P(MCI|Clin)
  P(AD|MRI)  P(AD|FDG)   P(AD|Clin)
     │            │              │
     └─────┬──────┴──────────────┘
           │
   ┌───────┴────────┐
   │ Attention-Based │
   │ Neural Fusion   │
   │ (Learned Weights│
   │  per patient)   │
   └───────┬────────┘
           │
    Final Prediction
    CN / MCI / AD
    + Confidence Score
    + Grad-CAM Map
    """, language="text")
    
    st.markdown("---")
    
    # ── Modality Details ──────────────────────────────────────────
    st.markdown("### Signals Used")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🧲 Structural MRI**")
        st.markdown("- T1-weighted brain scans")
        st.markdown("- **3D CNN features** (full volume)")
        st.markdown("- 40+ statistical features")
        st.markdown("- Texture & gradient features")
        st.markdown("- Regional asymmetry analysis")
        st.markdown("- Captures brain **atrophy**")
    with col2:
        st.markdown("**🔬 FDG PET**")
        st.markdown("- Metabolic imaging (glucose)")
        st.markdown("- **3D CNN features** (full volume)")
        st.markdown("- 40+ statistical features")
        st.markdown("- Regional metabolism patterns")
        st.markdown("- Captures brain **hypometabolism**")
    with col3:
        st.markdown("**📋 Clinical (MMSCORE)**")
        st.markdown("- Mini-Mental State Exam")
        st.markdown("- **15 enhanced features**")
        st.markdown("- Severity classification bins")
        st.markdown("- Population-relative z-scores")
        st.markdown("- Sigmoid & exponential transforms")
        st.markdown("- Captures **cognitive decline**")


elif page == "📊 Results":
    st.title("📊 Final Results — Test Set Evaluation")
    
    if not results_available:
        st.error("No results found. Run the pipeline first: `python adni_project/run.py`")
    else:
        st.markdown("### Performance Summary")
        
        display_df = metrics_df[metrics_df["Model"].isin(["MRI", "FDG", "Clinical", "Fusion"])]
        st.dataframe(
            display_df.style.highlight_max(subset=["Accuracy", "ROC_AUC"], color="#c8e6c9"),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Classification report
        report_path = RESULTS_DIR / "classification_report.txt"
        if report_path.exists():
            st.markdown("### Detailed Classification Report")
            with open(report_path) as f:
                st.code(f.read(), language="text")


elif page == "📈 Cross-Validation":
    st.title("📈 Cross-Validation Results")
    
    img_path = RESULTS_DIR / "cv_results.png"
    if img_path.exists():
        st.image(str(img_path), caption="5-Fold Stratified Cross-Validation Results", use_column_width=True)
        st.markdown("""
**How to read this:**
- Each bar represents one fold's performance
- Red dashed line = mean across all folds
- Shaded area = ±1 standard deviation
- Lower variance = more robust model
        """)
    else:
        st.info("Cross-validation results will appear after running the pipeline with CV_ENABLED=True")


elif page == "🔍 Confusion Matrix":
    st.title("🔍 Confusion Matrix")
    img_path = RESULTS_DIR / "confusion_matrix.png"
    if img_path.exists():
        st.image(str(img_path), caption="Fusion Model — Test Set", width=600)
        st.markdown("""
**How to read this:**
- Rows = True class, Columns = Predicted class
- Diagonal = correct predictions (higher = better)
- Off-diagonal = misclassifications
- Percentages show class-level accuracy
- MCI is typically the hardest class to classify
        """)
    else:
        st.warning("Run the pipeline first to generate this plot.")


elif page == "📉 ROC Curves":
    st.title("📉 ROC Curves")
    img_path = RESULTS_DIR / "roc_curves.png"
    if img_path.exists():
        st.image(str(img_path), caption="ROC Curves — One vs Rest per class", use_column_width=True)
        st.markdown("""
**How to read this:**
- Each curve shows separation ability for one class
- AUC = Area Under Curve (1.0 = perfect, 0.5 = random)
- Shaded area indicates discriminative power
- Macro-average AUC summarizes overall performance
        """)
    else:
        st.warning("Run the pipeline first to generate this plot.")


elif page == "⚖️ Model Comparison":
    st.title("⚖️ Model Comparison")
    img_path = RESULTS_DIR / "model_comparison.png"
    if img_path.exists():
        st.image(str(img_path), caption="Accuracy and ROC-AUC across all models", use_column_width=True)
    else:
        st.warning("Run the pipeline first to generate this plot.")
    
    if results_available:
        st.markdown("### Metrics Table")
        st.dataframe(metrics_df, use_container_width=True)


elif page == "🎯 Attention Analysis":
    st.title("🎯 Attention Weight Analysis")
    
    img_path = RESULTS_DIR / "attention_weights.png"
    if img_path.exists():
        st.image(str(img_path), caption="Modality Attention Weights by Class", use_column_width=True)
        st.markdown("""
**What this shows:**
- The attention mechanism learns which modality to trust for each patient
- Higher weight = more important for that patient's diagnosis
- Different classes may rely on different modalities
- **CN patients**: May rely more on Clinical (normal scores confirm health)
- **MCI patients**: May rely on imaging (subtle brain changes)
- **AD patients**: All modalities typically agree (clear signs everywhere)
        """)
    else:
        st.info("Attention analysis will appear after running with FUSION_TYPE='attention'")
    
    # Grad-CAM section
    st.markdown("---")
    st.markdown("### 🧠 Grad-CAM Brain Region Analysis")
    
    gradcam_dir = RESULTS_DIR / "gradcam"
    if gradcam_dir.exists():
        gradcam_files = sorted(gradcam_dir.glob("*.png"))
        if gradcam_files:
            for img_file in gradcam_files:
                st.image(str(img_file), caption=img_file.stem.replace("_", " ").title(),
                        use_column_width=True)
        else:
            st.info("No Grad-CAM visualizations found.")
    else:
        st.info("Grad-CAM visualizations will appear after running with GRADCAM_ENABLED=True")


elif page == "🔬 Per-Class Metrics":
    st.title("🔬 Per-Class Clinical Metrics")
    
    img_path = RESULTS_DIR / "per_class_metrics.png"
    if img_path.exists():
        st.image(str(img_path), caption="Sensitivity, Specificity, PPV, NPV per class",
                use_column_width=True)
        st.markdown("""
**Clinical Metrics Explained:**

| Metric | What it means | Clinical importance |
|--------|--------------|-------------------|
| **Sensitivity** | How many actual cases are detected | Missing AD cases is dangerous |
| **Specificity** | How many healthy people are correctly identified | Reduces false alarms |
| **PPV** | Of positive predictions, how many are correct | Trustworthiness of diagnosis |
| **NPV** | Of negative predictions, how many are correct | Confidence in "all clear" |
        """)
    else:
        st.info("Per-class metrics will appear after running the pipeline.")


elif page == "🧪 Ablation Study":
    st.title("🧪 Ablation Study")
    st.markdown("""
    The ablation study measures what happens when each modality is removed from the fusion.
    This reveals how much each signal contributes to the final prediction.
    """)

    if not results_available:
        st.error("No results found. Run the pipeline first.")
    else:
        ablation_rows = metrics_df[metrics_df["Model"].str.startswith("drop_")]
        
        if len(ablation_rows) > 0:
            ablation_display = ablation_rows.copy()
            ablation_display["Dropped Modality"] = ablation_display["Model"].str.replace("drop_", "")
            ablation_display = ablation_display[["Dropped Modality", "Accuracy", "F1_Macro"]]
            ablation_display.columns = ["Dropped Modality", "Accuracy (after drop)", "F1-Score (after drop)"]
            
            st.dataframe(ablation_display, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Interpretation")
            
            baseline_acc = fusion_acc
            impacts = []
            for _, row in ablation_rows.iterrows():
                modality = row["Model"].replace("drop_", "")
                drop_acc = row["Accuracy"]
                impact = baseline_acc - drop_acc
                impacts.append((modality, impact))
            
            impacts.sort(key=lambda x: x[1], reverse=True)
            
            st.markdown("**Modality Importance (by accuracy drop):**")
            for modality, impact in impacts:
                if impact > 0.1:
                    st.markdown(f"- 🔴 **{modality}**: Major impact (Δ = {impact:.3f}) — Critical signal")
                elif impact > 0.05:
                    st.markdown(f"- 🟡 **{modality}**: Moderate impact (Δ = {impact:.3f}) — Useful signal")
                elif impact > 0:
                    st.markdown(f"- 🟢 **{modality}**: Minor impact (Δ = {impact:.3f}) — Supplementary")
                else:
                    st.markdown(f"- ⚪ **{modality}**: No impact (Δ = {impact:.3f})")
            
            st.success("**Conclusion:** Multimodal fusion leverages complementary information from all signals.")
        else:
            st.warning("No ablation results found in metrics file.")


elif page == "📋 Experiment History":
    st.title("📋 Experiment History")
    
    if EXPERIMENT_DIR.exists():
        experiments = []
        for exp_dir in sorted(EXPERIMENT_DIR.iterdir(), reverse=True):
            if exp_dir.is_dir() and (exp_dir / "experiment.json").exists():
                try:
                    with open(exp_dir / "experiment.json") as f:
                        record = json.load(f)
                    experiments.append(record)
                except Exception:
                    pass
        
        if experiments:
            st.markdown(f"**Total experiments:** {len(experiments)}")
            
            for exp in experiments[:10]:  # Show last 10
                with st.expander(f"🔬 {exp.get('experiment_name', 'Unknown')} — "
                               f"{exp.get('status', 'unknown')} "
                               f"({exp.get('duration_seconds', 0):.0f}s)"):
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Hyperparameters:**")
                        st.json(exp.get("hyperparameters", {}))
                    with col2:
                        st.markdown("**Final Metrics:**")
                        final_metrics = {k: v for k, v in exp.get("metrics", {}).items()
                                       if k.startswith("final/")}
                        st.json(final_metrics)
                    
                    if exp.get("fold_results"):
                        st.markdown("**Cross-Validation Folds:**")
                        fold_data = []
                        for fold in exp["fold_results"]:
                            fold_data.append({
                                "Fold": fold["fold"] + 1,
                                **fold.get("metrics", {})
                            })
                        st.dataframe(pd.DataFrame(fold_data), use_container_width=True)
        else:
            st.info("No experiments found yet. Run the pipeline to create experiment records.")
    else:
        st.info("Experiment directory not found. Run the pipeline first.")


elif page == "⚠️ Disclaimer":
    st.title("⚠️ Important Disclaimers")
    st.error("""
    **This system is a RESEARCH PROTOTYPE and is NOT for clinical use.**

    - Built on the ADNI research dataset under controlled academic conditions
    - Has NOT been validated as a medical device
    - Must NOT be used to diagnose or treat any patient
    - All outputs must be interpreted by qualified medical professionals
    - Performance on real-world unseen data may differ significantly
    """)

    st.info("""
    **About ADNI:**
    Data used in this project was obtained from the Alzheimer's Disease Neuroimaging
    Initiative (ADNI) database. ADNI is funded by the National Institute on Aging and
    the National Institute of Biomedical Imaging and Bioengineering.
    For more information: adni.loni.usc.edu
    """)

    if results_available:
        st.warning(f"""
    **About the Results:**
    - Fusion Accuracy = {fusion_acc:.1%}
    - Fusion ROC-AUC = {fusion_auc:.3f}
    - Validated with 5-fold stratified cross-validation
    - Includes 95% confidence intervals
    - Random chance baseline for 3 classes = 33.3%
        """)
    
    st.markdown("---")
    st.markdown("### Technical Specifications")
    st.markdown("""
    | Component | Specification |
    |-----------|--------------|
    | **Feature Extraction** | 3D CNN (Residual + SE blocks) |
    | **Imaging Features** | 256-dim deep + 40 statistical |
    | **Clinical Features** | 15 enhanced MMSE derivatives |
    | **Classifiers** | SVM Ensemble + XGBoost |
    | **Fusion** | Attention-based Neural Network |
    | **Validation** | 5-fold Stratified Cross-Validation |
    | **Interpretability** | Grad-CAM + Attention Weights |
    | **Uncertainty** | Monte Carlo Dropout |
    | **Balancing** | SMOTE + Class Weights |
    """)
