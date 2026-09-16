"""
run_analysis_dataset2.py
--------------------------
Runs the paper-1-style behavioral pipeline on the SECOND dataset
(all_users.xlsx, ~19,510 users, ~2,028 human-labeled). Results are kept
completely separate from dataset 1 (users_with_retweets.xlsx) -- different
file names, different results folder -- per explicit request.

Compares:
  - RF-Behavioral (24 features, same as the original paper)
  - RF-Behavioral (24 + 19 extra = 43 features, dataset-2 equivalent of
    the dataset-1 innovation)
  - RF-Behavioral+Botometer (43 features + the independent Botometer score
    as one extra feature -- NOT used to derive labels, so no circularity)
  - A standalone check: how well does Botometer Score ALONE (no ML, just
    thresholding the external tool's own score) agree with our labels?

Usage:
    python3 run_analysis_dataset2.py /path/to/all_users.xlsx
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

from datetime import datetime, timezone
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, precision_score,
                             recall_score, accuracy_score)

from features import engineer_behavioral_features
from load_dataset2 import load_dataset2
from extra_features_v2 import engineer_extra_features_v2, EXTRA_FEATURE_NAMES_V2

RNG = 42
np.random.seed(RNG)
N_SPLITS = 5

XLSX_PATH = sys.argv[1] if len(sys.argv) > 1 else "all_users.xlsx"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_dataset2")   # SEPARATE output folder
os.makedirs(OUT, exist_ok=True)
REF_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def cv_evaluate(X, y, model_builder, name, n_splits=N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    accs, f1s, precs, recs, aucs = [], [], [], [], []
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = model_builder()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        pp = m.predict_proba(X[te])[:, 1]
        oof[te] = pp
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, zero_division=0))
        precs.append(precision_score(y[te], p, zero_division=0))
        recs.append(recall_score(y[te], p))
        aucs.append(roc_auc_score(y[te], pp))
    return {
        'model': name, 'n': len(y),
        'accuracy_mean': np.mean(accs), 'f1_mean': np.mean(f1s),
        'precision_mean': np.mean(precs), 'recall_mean': np.mean(recs),
        'roc_auc_mean': np.mean(aucs), 'roc_auc_std': np.std(aucs),
    }, oof


print("=" * 70)
print("LOADING DATASET 2 (all_users.xlsx)")
print("=" * 70)
df_raw, extra_cols_present = load_dataset2(XLSX_PATH)
df_labeled = df_raw.dropna(subset=["bot"]).copy()
df_labeled["bot"] = df_labeled["bot"].astype(int)
print(f"Total users in file: {len(df_raw)}")
print(f"Human/suspension-labeled users: {len(df_labeled)}  (bot fraction: {df_labeled['bot'].mean():.3f})")
print(f"Label source breakdown:\n{df_labeled['label_source'].value_counts()}")

# --- base 24 behavioral features (same engineering as paper 1) ---
X_behav24 = engineer_behavioral_features(df_labeled, ref_date=REF_DATE)
behav24_cols = list(X_behav24.columns)

# --- extra (dataset-2-adapted) behavioral features ---
X_extra = engineer_extra_features_v2(df_labeled)
X_all = X_behav24.join(X_extra)
behav_all_cols = behav24_cols + EXTRA_FEATURE_NAMES_V2
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_all.median(numeric_only=True))

y = df_labeled["bot"].values
print(f"\n24-feature (paper-1) matrix: {X_behav24.shape}")
print(f"43-feature (extended) matrix: {X_all[behav_all_cols].shape}")

# ============================================================
# Table D2-1: model comparison
# ============================================================
results = []
oof_probas = {}

res, oof = cv_evaluate(X_behav24.values, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1), f'RF-Behavioral ({len(behav24_cols)} feats, paper-1 style)')
results.append(res); oof_probas['RF-Behavioral (24)'] = oof

res, oof = cv_evaluate(X_all[behav_all_cols].values, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1), f'RF-Behavioral ({len(behav_all_cols)} feats, extended)')
results.append(res); oof_probas['RF-Behavioral (extended)'] = oof
rf_ext_full = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1).fit(X_all[behav_all_cols].values, y)

# --- + Botometer score, on the subset that actually has one ---
has_bm = df_labeled["botometer_score"].notna()
print(f"\nUsers with both label AND Botometer score: {has_bm.sum()} / {len(y)}")
if has_bm.sum() >= 50:
    X_bm_cols = behav_all_cols + ['botometer_score']
    X_with_bm = X_all.loc[has_bm, behav_all_cols].copy()
    X_with_bm['botometer_score'] = df_labeled.loc[has_bm, 'botometer_score'].values
    y_bm = y[has_bm.values]

    res, oof_bm = cv_evaluate(X_with_bm.values, y_bm, lambda: RandomForestClassifier(
        n_estimators=300, random_state=RNG, n_jobs=-1), f'RF-Behavioral+Botometer ({len(X_bm_cols)} feats, subset)', n_splits=5)
    results.append(res)

    # same-subset behavioral-only baseline, for a fair apples-to-apples comparison
    res_base_subset, _ = cv_evaluate(X_all.loc[has_bm, behav_all_cols].values, y_bm, lambda: RandomForestClassifier(
        n_estimators=300, random_state=RNG, n_jobs=-1), f'RF-Behavioral (extended, SAME subset as above)', n_splits=5)
    results.append(res_base_subset)

    # standalone: does Botometer's own score (no ML) already separate the classes?
    bm_auc = roc_auc_score(y_bm, df_labeled.loc[has_bm, 'botometer_score'].values)
    print(f"\nStandalone Botometer Score AUC (no ML, just their score vs our labels): {bm_auc:.3f}")
else:
    bm_auc = None

table_d2 = pd.DataFrame(results)
table_d2.to_csv(f'{OUT}/table_d2_model_comparison.csv', index=False)
print("\nTable D2-1 -- dataset 2 model comparison:")
print(table_d2.to_string(index=False))

# ============================================================
# Feature importance (extended model, full labeled set)
# ============================================================
imp = pd.DataFrame({'feature': behav_all_cols, 'importance': rf_ext_full.feature_importances_}) \
    .sort_values('importance', ascending=False).reset_index(drop=True)
imp.to_csv(f'{OUT}/table_d2_feature_importance.csv', index=False)
print("\nTop 15 features (dataset 2, extended model):")
print(imp.head(15).to_string(index=False))

# ============================================================
# Figures
# ============================================================
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(7, 6))
for name in oof_probas:
    fpr, tpr, _ = roc_curve(y, oof_probas[name])
    auc = roc_auc_score(y, oof_probas[name])
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Dataset 2 (19.5k users): behavioral model ROC', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_d2_roc.png', dpi=150, bbox_inches='tight')
plt.close()

top15 = imp.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.barh(top15['feature'], top15['importance'], color='#2b8cbe')
ax.set_xlabel('Importance (Gini)')
ax.set_title('Dataset 2: top 15 behavioral features', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_d2_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Summary
# ============================================================
summary = {
    'dataset': 'dataset2_all_users_19510',
    'total_users': int(len(df_raw)),
    'labeled_users': int(len(df_labeled)),
    'bot_fraction': float(y.mean()),
    'label_source_breakdown': df_labeled['label_source'].value_counts().to_dict(),
    'RF_24feat_AUC': float(table_d2.iloc[0]['roc_auc_mean']),
    'RF_extended_AUC': float(table_d2.iloc[1]['roc_auc_mean']),
    'botometer_standalone_AUC': float(bm_auc) if bm_auc is not None else None,
    'n_users_with_botometer_and_label': int(has_bm.sum()),
}
with open(f'{OUT}/summary_dataset2.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print("SUMMARY (dataset 2)")
print("=" * 70)
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nAll outputs written to: {OUT}")
