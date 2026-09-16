"""
run_fusion_paper2_dataset2.py
-------------------------------
Re-runs the "paper 2 stand-in" fusion (Persian linguistic features +
cross-account coordination features) from run_fusion_paper2.py, but on
DATASET 2, which has a much larger text-labeled subset (689 users vs
162 in dataset 1). Tests whether the earlier "content signal barely
helps" finding was a data-size artifact or a real property of the
Persian bot population.

Usage:
    python3 run_fusion_paper2_dataset2.py /path/to/all_users.xlsx
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
from persian_lfc import extract_persian_ling_features, LFC_FEATURE_NAMES
from dataset2_text_utils import load_all_tweets, aggregate_text_per_user, compute_coordination_features_d2

RNG = 42
np.random.seed(RNG)
N_SPLITS = 5

XLSX_PATH = sys.argv[1] if len(sys.argv) > 1 else "all_users.xlsx"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_dataset2")
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
    return {'model': name, 'n': len(y), 'accuracy_mean': np.mean(accs), 'f1_mean': np.mean(f1s),
            'precision_mean': np.mean(precs), 'recall_mean': np.mean(recs),
            'roc_auc_mean': np.mean(aucs), 'roc_auc_std': np.std(aucs)}, oof


print("=" * 70)
print("LOADING DATASET 2 + TWEET TEXT")
print("=" * 70)
df_raw, extra_cols = load_dataset2(XLSX_PATH)
df_labeled = df_raw.dropna(subset=["bot"]).copy()
df_labeled["bot"] = df_labeled["bot"].astype(int)
df_labeled["id_str"] = df_labeled["id_str"].astype(str)

df_labeled["screen_name_norm"] = df_labeled["screen_name"].astype(str).str.strip().str.lower()

tw = load_all_tweets(XLSX_PATH, cache_pickle=os.path.join(OUT, "_tweet_cache.pkl"))
text_per_user = aggregate_text_per_user(tw)
df_labeled = df_labeled.set_index("screen_name_norm")
df_labeled["status"] = text_per_user.reindex(df_labeled.index).fillna("")
df_labeled = df_labeled.reset_index()

has_text = df_labeled["status"].str.len() > 0
df_text = df_labeled.loc[has_text].reset_index(drop=True)
print(f"Labeled users with tweet text: {len(df_text)} (bots: {df_text['bot'].sum()})")

# --- behavioral features (43: 24 base + 19 extra) ---
X_behav24 = engineer_behavioral_features(df_text, ref_date=REF_DATE)
behav24_cols = list(X_behav24.columns)
X_extra = engineer_extra_features_v2(df_text)
X_behav = X_behav24.join(X_extra)
behav_cols = behav24_cols + EXTRA_FEATURE_NAMES_V2
X_behav = X_behav.replace([np.inf, -np.inf], np.nan).fillna(X_behav.median(numeric_only=True))
y = df_text["bot"].values

# --- Persian linguistic features ---
print("Extracting Persian linguistic features...")
X_ling = extract_persian_ling_features(df_text["status"].tolist())

# --- coordination features (on this dataset's own tweet corpus) ---
print("Extracting coordination features...")
coord = compute_coordination_features_d2(tw)
coord_aligned = df_text[["screen_name_norm"]].merge(coord, left_on="screen_name_norm", right_index=True, how="left")
X_coord = coord_aligned[["duplicate_tweet_ratio", "n_coordination_partners"]].fillna(0.0).reset_index(drop=True)
print(f"Users with any coordination signal: {(X_coord['n_coordination_partners'] > 0).sum()} / {len(df_text)}")

ling_cols = LFC_FEATURE_NAMES + ["duplicate_tweet_ratio", "n_coordination_partners"]
X_ling_full = pd.concat([X_ling.reset_index(drop=True), X_coord], axis=1)
X_ling_full = X_ling_full.replace([np.inf, -np.inf], np.nan).fillna(X_ling_full.median(numeric_only=True))

X_fusion_df = pd.concat([X_behav.reset_index(drop=True), X_ling_full], axis=1)
fusion_cols = behav_cols + ling_cols

# ============================================================
# CV comparison
# ============================================================
results = []
oof_probas = {}

res, oof = cv_evaluate(X_behav[behav_cols].values, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1), f'RF-Behavioral ({len(behav_cols)} feats, text subset)')
results.append(res); oof_probas['RF-Behavioral'] = oof

res, oof = cv_evaluate(X_ling_full[ling_cols].values, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1), f'RF-Linguistic ({len(ling_cols)} feats)')
results.append(res); oof_probas['RF-Linguistic'] = oof

res, oof = cv_evaluate(X_fusion_df[fusion_cols].values, y, lambda: RandomForestClassifier(
    n_estimators=300, random_state=RNG, n_jobs=-1), f'RF-Fusion ({len(fusion_cols)} feats)')
results.append(res); oof_probas['RF-Fusion'] = oof

table = pd.DataFrame(results)
table.to_csv(f'{OUT}/table_d2_paper2_fusion.csv', index=False)
print("\nDataset-2 paper-2 fusion comparison (n={}):".format(len(y)))
print(table.to_string(index=False))

# feature importance
rf_fusion_full = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1).fit(X_fusion_df[fusion_cols].values, y)
imp = pd.DataFrame({'feature': fusion_cols, 'importance': rf_fusion_full.feature_importances_}) \
    .sort_values('importance', ascending=False).reset_index(drop=True)
imp['source'] = imp['feature'].apply(lambda f: 'linguistic/coordination (paper 2)' if f in ling_cols else 'behavioral (paper 1)')
imp.to_csv(f'{OUT}/table_d2_fusion_feature_importance.csv', index=False)
print("\nTop 15 fusion features (dataset 2):")
print(imp.head(15).to_string(index=False))

# ROC figure
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(7, 6))
for name in oof_probas:
    fpr, tpr, _ = roc_curve(y, oof_probas[name])
    auc = roc_auc_score(y, oof_probas[name])
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title(f'Dataset 2 (n={len(y)}): behavioral vs linguistic vs fusion', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_d2_paper2_fusion_roc.png', dpi=150, bbox_inches='tight')
plt.close()

summary_path = f'{OUT}/summary_dataset2.json'
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
summary['paper2_fusion_dataset2'] = {
    'n_users_with_text': int(len(y)), 'n_bots': int(y.sum()),
    'RF_Behavioral_AUC': float(table.iloc[0]['roc_auc_mean']),
    'RF_Linguistic_AUC': float(table.iloc[1]['roc_auc_mean']),
    'RF_Fusion_AUC': float(table.iloc[2]['roc_auc_mean']),
}
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nAll outputs written to: {OUT}")
