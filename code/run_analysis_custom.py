"""
run_analysis_custom.py
-----------------------
Same pipeline as run_analysis.py (Katyal 2026), but pointed at the
student's own labeled dataset (users_with_retweets.xlsx) instead of the
jubins/MachineLearning-Detecting-Twitter-Bots CSV.

Usage:
    python3 run_analysis_custom.py /path/to/users_with_retweets.xlsx

Produces (in ../results_custom/):
  - table1_descriptive_stats.csv
  - table2_classifier_performance.csv   <- behavioral vs content vs fusion
  - table4_feature_importance.csv
  - fig1_feature_distributions.png
  - fig2_roc_curves.png
  - fig4_feature_importance.png
  - summary.json

Notes on adaptation from the original paper:
  * Labels are the majority vote across the dataset's 3 human taggers
    (see load_custom_dataset.py). Rows with a tie / no vote are dropped
    (~19% of users) -> 895 labeled users remain.
  * Only ~123/1104 users have any sampled tweet text (tweetsMetaData
    sheet), so the CONTENT and FUSION models are trained/evaluated on
    that smaller subset only; the BEHAVIORAL model uses the full
    labeled set (895 users) since it never needs tweet text.
  * created_at reference date is set to 2025-01-01 (after the latest
    account-creation date in the data) instead of the original paper's
    2018-01-01.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, precision_score,
                             recall_score, accuracy_score)

from features import engineer_all, engineer_behavioral_features
from load_custom_dataset_v2 import load_custom_v2 as load_custom
from extra_features import engineer_extra_features, EXTRA_FEATURE_NAMES

RNG = 42
np.random.seed(RNG)

XLSX_PATH = sys.argv[1] if len(sys.argv) > 1 else "1000user_sheet.xlsx"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_custom")
os.makedirs(OUT, exist_ok=True)
REF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def cv_evaluate(X, y, model, name, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    accs, f1s, precs, recs, aucs = [], [], [], [], []
    probas_oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        pp = m.predict_proba(X[te])[:, 1]
        probas_oof[te] = pp
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p))
        precs.append(precision_score(y[te], p, zero_division=0))
        recs.append(recall_score(y[te], p))
        aucs.append(roc_auc_score(y[te], pp))
    return {
        'model': name,
        'n': len(y),
        'accuracy_mean': np.mean(accs), 'accuracy_std': np.std(accs),
        'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
        'precision_mean': np.mean(precs), 'precision_std': np.std(precs),
        'recall_mean': np.mean(recs), 'recall_std': np.std(recs),
        'roc_auc_mean': np.mean(aucs), 'roc_auc_std': np.std(aucs),
    }, probas_oof


print("=" * 70)
print("LOADING CUSTOM DATASET:", XLSX_PATH)
print("=" * 70)

df_raw = load_custom(XLSX_PATH)
df_labeled = df_raw.dropna(subset=["bot"]).copy()
df_labeled["bot"] = df_labeled["bot"].astype(int)
print(f"Labeled users (majority vote across 3 taggers): {len(df_labeled)}")
print(f"Bot fraction: {df_labeled['bot'].mean():.3f}")

# --- Engineer behavioral + content features on the labeled set ---
X_all, behav_cols, content_cols = engineer_all(df_labeled)
X_all = engineer_behavioral_features(df_labeled, ref_date=REF_DATE).join(
    X_all[content_cols]
)
original_behav_cols = list(behav_cols)  # the 24 features from the paper, kept for A/B comparison
X_extra = engineer_extra_features(df_labeled)
X_all = X_all.join(X_extra)
behav_cols = behav_cols + EXTRA_FEATURE_NAMES  # extended behavioral feature set
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_all.median(numeric_only=True))
y = df_labeled["bot"].values

print(f"\nBehavioral features ({len(behav_cols)}): {behav_cols}")
print(f"Content features ({len(content_cols)}): {content_cols}")

# ============================================================
# Table 1: descriptive stats (behavioral features, full labeled set)
# ============================================================
desc_features = [c for c in [
    'account_age_days', 'statuses_per_day', 'followers_per_day',
    'friends_per_day', 'friends_to_followers', 'log_followers',
    'log_friends', 'screen_name_digit_ratio', 'screen_name_entropy',
    'has_description', 'default_profile',
] if c in X_all.columns]

rows = []
for f in desc_features:
    h_mean, h_std = X_all.loc[y == 0, f].mean(), X_all.loc[y == 0, f].std()
    b_mean, b_std = X_all.loc[y == 1, f].mean(), X_all.loc[y == 1, f].std()
    pooled = np.sqrt((h_std**2 + b_std**2) / 2)
    d = (b_mean - h_mean) / pooled if pooled > 0 else 0.0
    rows.append({'feature': f, 'nonbot_mean': h_mean, 'nonbot_std': h_std,
                 'bot_mean': b_mean, 'bot_std': b_std, 'cohen_d': d})
table1 = pd.DataFrame(rows)
table1.to_csv(f'{OUT}/table1_descriptive_stats.csv', index=False)
print("\nTable 1 (behavioral, n={}):".format(len(y)))
print(table1.to_string(index=False))

# ============================================================
# Table 2: BEHAVIORAL model on the FULL labeled set (895 users)
# ============================================================
X_behav = X_all[behav_cols].values
scaler_b = StandardScaler().fit(X_behav)
Xb_s = scaler_b.transform(X_behav)

results = []
oof_probas = {}

# --- A/B check: original 24 paper features vs extended feature set ---
X_orig = X_all[original_behav_cols].values
res, _ = cv_evaluate(X_orig, y, RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1),
                      f'RF-Behavioral (paper, {len(original_behav_cols)} feats)')
results.append(res)

res, probas = cv_evaluate(X_behav, y, RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1),
                           f'RF-Behavioral (extended, {len(behav_cols)} feats)')
results.append(res); oof_probas['RF-Behavioral'] = probas

res, probas = cv_evaluate(Xb_s, y, LogisticRegression(max_iter=2000, random_state=RNG), 'LogReg-Behavioral')
results.append(res); oof_probas['LogReg-Behavioral'] = probas

rf_behav_full = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1).fit(X_behav, y)

res, probas = cv_evaluate(X_behav, y, GradientBoostingClassifier(n_estimators=200, random_state=RNG), 'GB-Behavioral')
results.append(res); oof_probas['GB-Behavioral'] = probas

# ============================================================
# CONTENT + FUSION: only on users that actually have tweet text
# ============================================================
has_text = df_labeled["status"].str.len() > 0
print(f"\nUsers with sampled tweet text: {has_text.sum()} / {len(df_labeled)}")

if has_text.sum() >= 30 and y[has_text.values].sum() >= 5 and (~y[has_text.values].astype(bool)).sum() >= 5:
    idx = has_text.values
    X_content_sub = X_all.loc[idx, content_cols].values
    X_full_sub = X_all.loc[idx].values
    y_sub = y[idx]

    res, probas = cv_evaluate(X_content_sub, y_sub,
                               RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1),
                               'RF-Content (text subset)', n_splits=3)
    results.append(res)

    res, probas = cv_evaluate(X_full_sub, y_sub,
                               RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1),
                               'RF-Fusion (text subset)', n_splits=3)
    results.append(res)

    rf_fusion_clf = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1).fit(X_full_sub, y_sub)
    feat_names_fusion = behav_cols + content_cols
else:
    print("Not enough labeled users with tweet text for a reliable content/fusion "
          "model (need >=30 users with >=5 per class) -- skipping those rows.")
    rf_fusion_clf = None
    feat_names_fusion = None

table2 = pd.DataFrame(results)
table2.to_csv(f'{OUT}/table2_classifier_performance.csv', index=False)
print("\nTable 2:")
print(table2.to_string(index=False))

# ============================================================
# Table 4: feature importance (behavioral-only RF, full set)
# ============================================================
importances = rf_behav_full.feature_importances_
table4 = pd.DataFrame({'feature': behav_cols, 'importance': importances}) \
    .sort_values('importance', ascending=False).reset_index(drop=True)
table4.to_csv(f'{OUT}/table4_feature_importance.csv', index=False)
print("\nTable 4 (top 15, behavioral-only RF):")
print(table4.head(15).to_string(index=False))

# ============================================================
# Figures
# ============================================================
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})

key_feats = [f for f in ['statuses_per_day', 'friends_to_followers', 'log_followers',
                          'screen_name_digit_ratio', 'has_description', 'account_age_days']
             if f in X_all.columns]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, f in zip(axes.flatten(), key_feats):
    h = X_all.loc[y == 0, f].values
    b = X_all.loc[y == 1, f].values
    if f in ('statuses_per_day', 'account_age_days', 'friends_to_followers'):
        h, b = np.log1p(h), np.log1p(b)
        xl = f'log(1 + {f})'
    else:
        xl = f
    bins = np.linspace(min(h.min(), b.min()), max(h.max(), b.max()), 40)
    ax.hist(h, bins=bins, alpha=0.55, label='Non-bot', density=True, color='#2b8cbe')
    ax.hist(b, bins=bins, alpha=0.55, label='Bot', density=True, color='#e34a33')
    ax.set_xlabel(xl); ax.set_ylabel('Density'); ax.legend(fontsize=8)
fig.suptitle('Feature distributions by class -- custom dataset', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
for name in oof_probas:
    fpr, tpr, _ = roc_curve(y, oof_probas[name])
    auc = roc_auc_score(y, oof_probas[name])
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Chance')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curves -- 5-fold CV, custom dataset', fontweight='bold')
ax.legend(loc='lower right'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

top = table4.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.barh(top['feature'], top['importance'], color='#2b8cbe')
ax.set_xlabel('Importance (RF-Behavioral, Gini)')
ax.set_title('Top 15 behavioral features -- custom dataset', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Summary
# ============================================================
summary = {
    'total_users_in_file': int(len(df_raw)),
    'labeled_users_majority_vote': int(len(df_labeled)),
    'bot_fraction': float(y.mean()),
    'users_with_tweet_text': int(has_text.sum()),
    'n_behavioral_features': len(behav_cols),
    'n_content_features': len(content_cols),
    'RF_Behavioral_AUC': float(table2.loc[table2['model'] == f'RF-Behavioral (extended, {len(behav_cols)} feats)', 'roc_auc_mean'].iloc[0]),
    'RF_Behavioral_paper_only_AUC': float(table2.loc[table2['model'] == f'RF-Behavioral (paper, {len(original_behav_cols)} feats)', 'roc_auc_mean'].iloc[0]),
    'GB_Behavioral_AUC': float(table2.loc[table2['model'] == 'GB-Behavioral', 'roc_auc_mean'].iloc[0]),
}
if rf_fusion_clf is not None:
    summary['RF_Content_subset_AUC'] = float(table2.loc[table2['model'] == 'RF-Content (text subset)', 'roc_auc_mean'].iloc[0])
    summary['RF_Fusion_subset_AUC'] = float(table2.loc[table2['model'] == 'RF-Fusion (text subset)', 'roc_auc_mean'].iloc[0])

with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(json.dumps(summary, indent=2))
print(f"\nAll outputs written to: {OUT}")
