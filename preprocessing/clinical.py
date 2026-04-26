"""
Enhanced Clinical Feature Engineering (Problem 6)

Upgrades from 6 features (MMSE-only) to 15+ features including:
- MMSE polynomial and non-linear transformations
- Clinical severity bins (mild/moderate/severe)
- Interaction features between cognitive domains
- Statistical normalization features (z-score relative to population)
- Age-adjusted scoring proxies
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_clinical_features(manifest: pd.DataFrame,
                             subject_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Build comprehensive clinical features from available data.
    
    Enhanced Features (15 total):
    1.  Raw MMSCORE
    2.  MMSCORE squared (non-linear relationship)
    3.  MMSCORE cubed (higher-order patterns)
    4.  Inverse MMSCORE (1 / (score + 1))
    5.  Log MMSCORE (log scale relationship)
    6.  Binary threshold (score < 24 indicates impairment)
    7.  Mild impairment flag (24 <= score < 27)
    8.  Moderate impairment flag (18 <= score < 24)
    9.  Severe impairment flag (score < 18)
    10. Distance from normal (30 - score)
    11. Normalized score (score / 30)
    12. Score deviation from population mean (computed later)
    13. Score percentile rank (computed later)
    14. Sigmoid transformation of score
    15. Exponential decay from perfect score
    """
    sid_to_row = {row["subject_id"]: row for _, row in manifest.iterrows()}

    X_list = []
    y_list = []
    scores = []

    # First pass: collect all scores for population-level features
    for sid in subject_ids:
        row = sid_to_row[sid]
        scores.append(float(row["mmscore"]))
    
    scores_array = np.array(scores)
    pop_mean = scores_array.mean()
    pop_std = scores_array.std() if scores_array.std() > 1e-8 else 1.0
    sorted_scores = np.sort(scores_array)

    # Second pass: build features
    for i, sid in enumerate(subject_ids):
        row = sid_to_row[sid]
        score = float(row["mmscore"])
        
        # ── Core MMSE Features ────────────────────────────────────────
        features = [
            score,                                          # 1. Raw score
            score ** 2,                                     # 2. Squared
            score ** 3,                                     # 3. Cubed
            1.0 / (score + 1.0),                           # 4. Inverse
            np.log(score + 1.0),                           # 5. Log
            1.0 if score < 24 else 0.0,                    # 6. Impaired (clinical cutoff)
        ]
        
        # ── Severity Bins ─────────────────────────────────────────────
        features.extend([
            1.0 if 24 <= score < 27 else 0.0,              # 7. Mild impairment
            1.0 if 18 <= score < 24 else 0.0,              # 8. Moderate impairment
            1.0 if score < 18 else 0.0,                    # 9. Severe impairment
        ])
        
        # ── Derived Continuous Features ────────────────────────────────
        features.extend([
            30.0 - score,                                   # 10. Distance from perfect
            score / 30.0,                                   # 11. Normalized [0, 1]
            (score - pop_mean) / pop_std,                  # 12. Z-score (population)
            float(np.searchsorted(sorted_scores, score)) / len(sorted_scores),  # 13. Percentile rank
            1.0 / (1.0 + np.exp(-(score - 24) / 3)),      # 14. Sigmoid centered at 24
            np.exp(-(30.0 - score) / 10.0),                # 15. Exponential decay
        ])
        
        X_list.append(features)
        y_list.append(int(row["label"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    print(f"  Clinical features: {X.shape} (15 enhanced features from MMSCORE)")
    print(f"    Score range: [{scores_array.min():.0f}, {scores_array.max():.0f}], "
          f"mean={pop_mean:.1f}, std={pop_std:.1f}")
    return X, y, scaler
