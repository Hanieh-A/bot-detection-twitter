"""
run_fusion_paper2.py
---------------------
Roadmap: integrating "paper 2" (Trokhymovych et al., 2026) into the
project. Since (a) their pretrained HF model is unreachable from this
environment and (b) their exact LFC pipeline (lftk+spaCy) does not
support Persian -- see persian_lfc.py for the documented adaptation --
this script:

  1. Extracts the 17 Persian-adapted linguistic features (persian_lfc.py)
     from each user's sampled tweet text (~162/1045 users).
  2. Trains a standalone "RF-Linguistic" model (bot vs human) on just
     these 17 features -- this is our stand-in for paper 2's content
     classifier.
  3. Builds a FUSION model: 45 behavioral features + 17 linguistic
     features = 62 features total.
  4. Compares RF-Behavioral / RF-Linguistic / RF-Fusion on the same
     text-bearing subset (5-fold CV), and reports whether adding the
     linguistic layer beats behavioral-only.

Usage:
    python3 run_fusion_paper2.py /path/to/users_with_retweets.xlsx
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

from features import engineer_all, engineer_behavioral_features
from load_custom_dataset import load_custom
from extra_features import engineer_extra_features, EXTRA_FEATURE_NAMES
from persian_lfc import extract_persian_ling_features, LFC_FEATURE_NAMES
from coordination_features import compute_coordination_features, COORD_FEATURE_NAMES

RNG = 42
np.random.seed(RNG)
N_SPLITS = 5

XLSX_PATH = sys.argv[1] if len(sys.argv) > 1 else "users_with_retweets.xlsx"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_custom")
os.makedirs(OUT, exist_ok=True)
REF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)


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
print("LOADING DATA")
print("=" * 70)
df_raw = load_custom(XLSX_PATH)
df_labeled = df_raw.dropna(subset=["bot"]).copy()
df_labeled["bot"] = df_labeled["bot"].astype(int)

X_all, behav_cols, content_cols_orig = engineer_all(df_labeled)
X_all = engineer_behavioral_features(df_labeled, ref_date=REF_DATE).join(X_all[content_cols_orig])
X_extra = engineer_extra_features(df_labeled)
X_all = X_all.join(X_extra)
behav_cols = behav_cols + EXTRA_FEATURE_NAMES  # 45 behavioral features
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_all.median(numeric_only=True))

# --- restrict to users with sampled tweet text ---
has_text = df_labeled["status"].str.len() > 0
df_text = df_labeled.loc[has_text].reset_index(drop=True)
Xb_text = X_all.loc[has_text].reset_index(drop=True)
y = df_text["bot"].values
print(f"Users with tweet text: {len(df_text)} (bots: {y.sum()})")

# --- extract Persian linguistic ("paper 2 stand-in") features ---
print("\nExtracting Persian linguistic features (hazm-based)...")
X_ling = extract_persian_ling_features(df_text["status"].tolist())
X_ling = X_ling.replace([np.inf, -np.inf], np.nan).fillna(X_ling.median(numeric_only=True))
print(f"Linguistic features: {len(LFC_FEATURE_NAMES)} -> {LFC_FEATURE_NAMES}")

# --- extract cross-account duplicate/coordination features ---
print("\nExtracting coordination (cross-account duplicate content) features...")
coord = compute_coordination_features(XLSX_PATH)
coord_aligned = df_text[["id_str"]].merge(coord, left_on="id_str", right_index=True, how="left")
X_coord = coord_aligned[COORD_FEATURE_NAMES].fillna(0.0).reset_index(drop=True)
print(f"Coordination features: {COORD_FEATURE_NAMES}")
print(f"Users with any coordination signal in this subset: {(X_coord['n_coordination_partners'] > 0).sum()}")

LFC_FEATURE_NAMES = LFC_FEATURE_NAMES + COORD_FEATURE_NAMES
X_ling = pd.concat([X_ling.reset_index(drop=True), X_coord], axis=1)

# --- assemble fusion matrix ---
X_fusion_df = pd.concat([Xb_text.reset_index(drop=True), X_ling.reset_index(drop=True)], axis=1)
fusion_cols = behav_cols + LFC_FEATURE_NAMES

X_behav_only = X_fusion_df[behav_cols].values
X_ling_only = X_fusion_df[LFC_FEATURE_NAMES].values
X_fusion = X_fusion_df[fusion_cols].values

print(f"\nMatrix shapes -- behavioral only: {X_behav_only.shape}, "
      f"linguistic only: {X_ling_only.shape}, fusion: {X_fusion.shape}")

# ============================================================
# 5-fold CV comparison
# ============================================================
results = []
oof_probas = {}

res, oof = cv_evaluate(X_behav_only, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1), 'RF-Behavioral (45 feats, text subset)')
results.append(res); oof_probas['RF-Behavioral'] = oof

res, oof = cv_evaluate(X_ling_only, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1),
    f'RF-Linguistic ({len(LFC_FEATURE_NAMES)} feats: 17 Persian-LFC + 2 coordination)')
results.append(res); oof_probas['RF-Linguistic'] = oof

res, oof = cv_evaluate(X_fusion, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1),
    f'RF-Fusion (45+{len(LFC_FEATURE_NAMES)}={45+len(LFC_FEATURE_NAMES)} feats)')
results.append(res); oof_probas['RF-Fusion'] = oof

table7 = pd.DataFrame(results)
table7.to_csv(f'{OUT}/table7_paper2_fusion.csv', index=False)
print("\nTable 7 -- paper-2 (Persian LFC) fusion comparison:")
print(table7.to_string(index=False))

# ============================================================
# Feature importance in the fusion model (full fit)
# ============================================================
rf_fusion_full = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1).fit(X_fusion, y)
imp = pd.DataFrame({'feature': fusion_cols, 'importance': rf_fusion_full.feature_importances_}) \
    .sort_values('importance', ascending=False).reset_index(drop=True)
imp['source'] = imp['feature'].apply(lambda f: 'linguistic (paper 2)' if f in LFC_FEATURE_NAMES else 'behavioral (paper 1)')
imp.to_csv(f'{OUT}/table8_fusion_feature_importance.csv', index=False)
print("\nTable 8 -- top 15 features in the fusion model:")
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
ax.set_title('Paper-1 (behavioral) vs Paper-2 stand-in (linguistic) vs Fusion', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT}/fig11_paper2_fusion_roc.png', dpi=150, bbox_inches='tight')
plt.close()

top15 = imp.head(15).iloc[::-1]
colors = ['#e34a33' if s.startswith('linguistic') else '#2b8cbe' for s in top15['source']]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.barh(top15['feature'], top15['importance'], color=colors)
ax.set_xlabel('Importance (Gini)')
ax.set_title('Top 15 features -- fusion model (blue=behavioral, red=linguistic)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig12_fusion_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Summary
# ============================================================
summary_path = f'{OUT}/summary.json'
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
summary['paper2_fusion'] = {
    'n_users_with_text': int(len(y)), 'n_bots': int(y.sum()),
    'n_linguistic_features': len(LFC_FEATURE_NAMES),
    'RF_Behavioral_AUC': float(table7.loc[table7['model'].str.startswith('RF-Behavioral'), 'roc_auc_mean'].iloc[0]),
    'RF_Linguistic_AUC': float(table7.loc[table7['model'].str.startswith('RF-Linguistic'), 'roc_auc_mean'].iloc[0]),
    'RF_Fusion_AUC': float(table7.loc[table7['model'].str.startswith('RF-Fusion'), 'roc_auc_mean'].iloc[0]),
    'note': ("Linguistic features are a Persian-language adaptation (hazm-based) of the "
             "Trokhymovych et al. LFC idea, not their original lftk/spaCy pipeline or "
             "pretrained mBERT model, both of which are unavailable for Persian / this "
             "sandboxed environment respectively."),
}
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs written to: {OUT}")
