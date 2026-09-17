"""
run_imbalance.py
-----------------
Roadmap step 1: class-imbalance handling for the RF-Behavioral model
(45 extended features). Compares:
  1. Baseline            -- unweighted RF, threshold = 0.5 (what we had so far)
  2. class_weight        -- RF(class_weight='balanced'), threshold = 0.5
  3. Threshold tuning     -- unweighted RF, threshold chosen to maximize F1
                             from out-of-fold (OOF) probabilities
  4. SMOTE                -- oversample the minority (bot) class inside each
                             training fold only (no leakage into test fold),
                             threshold = 0.5
  5. Combined (recommended) -- class_weight='balanced' + tuned threshold

All experiments reuse the exact same 45-feature matrix and 5-fold CV split
(same random_state) as run_analysis_custom.py, so results are directly
comparable to Table 2 in the thesis-progress report.

Usage:
    python3 run_imbalance.py /path/to/users_with_retweets.xlsx
"""
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_recall_curve, f1_score,
                             precision_score, recall_score, accuracy_score)
from imblearn.over_sampling import SMOTE

from features import engineer_all, engineer_behavioral_features
from load_custom_dataset_v2 import load_custom_v2 as load_custom
from extra_features import engineer_extra_features, EXTRA_FEATURE_NAMES

RNG = 42
np.random.seed(RNG)
N_SPLITS = 5

XLSX_PATH = sys.argv[1] if len(sys.argv) > 1 else "1000user_sheet.xlsx"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_custom")
os.makedirs(OUT, exist_ok=True)
from datetime import datetime, timezone
REF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def get_oof_probas(X, y, model_builder, use_smote=False):
    """5-fold CV, returns out-of-fold predicted probabilities for the
    positive (bot) class. model_builder() must return a fresh, unfit model."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        if use_smote:
            sm = SMOTE(random_state=RNG)
            Xtr, ytr = sm.fit_resample(Xtr, ytr)
        m = model_builder()
        m.fit(Xtr, ytr)
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def metrics_at_threshold(y, probas, thr):
    pred = (probas >= thr).astype(int)
    return {
        'threshold': thr,
        'accuracy': accuracy_score(y, pred),
        'f1': f1_score(y, pred, zero_division=0),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'roc_auc': roc_auc_score(y, probas),
    }


def best_f1_threshold(y, probas):
    prec, rec, thr = precision_recall_curve(y, probas)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.nanargmax(f1s[:-1])  # last point has no matching threshold
    return thr[best_idx], f1s[best_idx]


print("=" * 70)
print("LOADING DATA")
print("=" * 70)
df_raw = load_custom(XLSX_PATH)
df_labeled = df_raw.dropna(subset=["bot"]).copy()
df_labeled["bot"] = df_labeled["bot"].astype(int)

X_all, behav_cols, content_cols = engineer_all(df_labeled)
X_all = engineer_behavioral_features(df_labeled, ref_date=REF_DATE).join(X_all[content_cols])
X_extra = engineer_extra_features(df_labeled)
X_all = X_all.join(X_extra)
behav_cols = behav_cols + EXTRA_FEATURE_NAMES
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_all.median(numeric_only=True))

X = X_all[behav_cols].values
y = df_labeled["bot"].values
print(f"n={len(y)}, bot fraction={y.mean():.3f}, n_features={X.shape[1]}")

# ============================================================
# 1. Baseline -- unweighted, threshold 0.5
# ============================================================
oof_base = get_oof_probas(X, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1))
row_baseline = {'config': 'Baseline (unweighted, thr=0.5)', **metrics_at_threshold(y, oof_base, 0.5)}

# ============================================================
# 2. class_weight='balanced' -- threshold 0.5
# ============================================================
oof_cw = get_oof_probas(X, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1, class_weight='balanced'))
row_cw = {'config': "class_weight='balanced' (thr=0.5)", **metrics_at_threshold(y, oof_cw, 0.5)}

# ============================================================
# 3. Threshold tuning on the UNWEIGHTED model's OOF probabilities
# ============================================================
tuned_thr, _ = best_f1_threshold(y, oof_base)
row_thr = {'config': f'Threshold tuning (unweighted, thr={tuned_thr:.3f})',
           **metrics_at_threshold(y, oof_base, tuned_thr)}

# ============================================================
# 4. SMOTE oversampling inside each training fold -- threshold 0.5
# ============================================================
oof_smote = get_oof_probas(X, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1), use_smote=True)
row_smote = {'config': 'SMOTE oversampling (thr=0.5)', **metrics_at_threshold(y, oof_smote, 0.5)}

# ============================================================
# 5. Combined (recommended): class_weight='balanced' + tuned threshold
# ============================================================
tuned_thr_cw, _ = best_f1_threshold(y, oof_cw)
row_combined = {'config': f"class_weight='balanced' + tuned thr ({tuned_thr_cw:.3f})",
                **metrics_at_threshold(y, oof_cw, tuned_thr_cw)}

table3 = pd.DataFrame([row_baseline, row_cw, row_thr, row_smote, row_combined])
table3 = table3[['config', 'threshold', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc']]
best_row = table3.loc[table3['f1'].idxmax()].copy()
table3['config'] = table3['config'].str.replace(' [RECOMMENDED]', '', regex=False)
table3.loc[table3['f1'].idxmax(), 'config'] += '  <-- BEST F1'
table3.to_csv(f'{OUT}/table3_imbalance_handling.csv', index=False)
print("\nTable 3 -- class imbalance handling comparison:")
print(table3.to_string(index=False))

# ============================================================
# Figure: Precision-Recall curves (baseline vs class-weighted)
# ============================================================
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(7, 6))
for probs, name in [(oof_base, 'Unweighted'), (oof_cw, "class_weight='balanced'"), (oof_smote, 'SMOTE')]:
    prec, rec, _ = precision_recall_curve(y, probs)
    ax.plot(rec, prec, lw=2, label=name)
ax.axhline(y.mean(), color='gray', ls='--', alpha=0.5, label=f'Chance (bot rate={y.mean():.2f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall curves -- imbalance handling comparison', fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nBest F1 config: {best_row['config']}")
print(f"  threshold={best_row['threshold']:.3f}  F1={best_row['f1']:.3f}  "
      f"precision={best_row['precision']:.3f}  recall={best_row['recall']:.3f}")
print(f"  -> F1 improves from {row_baseline['f1']:.3f} (baseline) to {best_row['f1']:.3f}")
print(f"  -> Recall improves from {row_baseline['recall']:.3f} to {best_row['recall']:.3f}")

summary_path = f'{OUT}/summary.json'
if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
else:
    summary = {}
summary['imbalance_handling'] = {
    'best_config': str(best_row['config']),
    'best_threshold': float(best_row['threshold']),
    'baseline_f1': float(row_baseline['f1']),
    'best_f1': float(best_row['f1']),
    'baseline_recall': float(row_baseline['recall']),
    'best_recall': float(best_row['recall']),
}
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs written to: {OUT}")
