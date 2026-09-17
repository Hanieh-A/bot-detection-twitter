"""
Main analysis pipeline for the behavioral-fingerprinting paper.

Produces:
  - results/table1_descriptive_stats.csv
  - results/table2_classifier_performance.csv
  - results/table3_adversarial_robustness.csv
  - results/table4_feature_importance.csv
  - results/fig1_feature_distributions.png
  - results/fig2_roc_curves.png
  - results/fig3_adversarial_degradation.png
  - results/fig4_feature_importance.png
  - results/twibot20_generalization.csv
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             classification_report)

from features import (engineer_all, engineer_behavioral_features,
                      OUT, ensure_dataset1, ensure_twibot20)

RNG = 42
np.random.seed(RNG)

os.makedirs(OUT, exist_ok=True)


# ============================================================
# 1. LOAD DATASET 1 (jubins/kaggle, labeled)
# ============================================================

print("=" * 70)
print("LOADING DATASET 1: Twitter Bot Accounts (labeled)")
print("=" * 70)

df1 = pd.read_csv(ensure_dataset1(),
                  encoding='utf-8', on_bad_lines='skip', low_memory=False)

# Clean labels & dedupe
df1 = df1.dropna(subset=['bot']).copy()
df1['bot'] = df1['bot'].astype(int)
df1 = df1.drop_duplicates(subset=['id_str']).reset_index(drop=True)

print(f"Records: {len(df1)}  |  Bots: {df1['bot'].sum()}  |  Humans: {(df1['bot']==0).sum()}")
print(f"Class balance: {df1['bot'].mean():.3f} bots")

# Engineer features
X_all, behav_cols, content_cols = engineer_all(df1)
y = df1['bot'].values

# Clean infs / NaNs
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_all.median(numeric_only=True))

print(f"\nBehavioral features ({len(behav_cols)}): {behav_cols}")
print(f"Content features ({len(content_cols)}): {content_cols}")


# ============================================================
# 2. DESCRIPTIVE STATISTICS (Table 1)
# ============================================================

print("\n" + "=" * 70)
print("Computing descriptive stats by class (Table 1)")
print("=" * 70)

# Pick a curated subset of behavioral features for the descriptive table
desc_features = [
    'account_age_days', 'statuses_per_day', 'followers_per_day',
    'friends_per_day', 'friends_to_followers', 'log_followers',
    'log_friends', 'screen_name_digit_ratio', 'screen_name_entropy',
    'has_description', 'default_profile', 'verified',
]

rows = []
for f in desc_features:
    h_mean = X_all.loc[y == 0, f].mean()
    h_std = X_all.loc[y == 0, f].std()
    b_mean = X_all.loc[y == 1, f].mean()
    b_std = X_all.loc[y == 1, f].std()
    # Effect size (Cohen's d)
    pooled = np.sqrt((h_std**2 + b_std**2) / 2)
    d = (b_mean - h_mean) / pooled if pooled > 0 else 0.0
    rows.append({
        'feature': f,
        'human_mean': h_mean, 'human_std': h_std,
        'bot_mean': b_mean, 'bot_std': b_std,
        'cohen_d': d
    })
table1 = pd.DataFrame(rows)
table1.to_csv(f'{OUT}/table1_descriptive_stats.csv', index=False)
print(table1.to_string(index=False))


# ============================================================
# 3. CLASSIFIER COMPARISON (Table 2)
# ============================================================

print("\n" + "=" * 70)
print("Training classifiers — 5-fold stratified CV (Table 2)")
print("=" * 70)

def cv_evaluate(X, y, model, name, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    accs, f1s, precs, recs, aucs = [], [], [], [], []
    preds_oof = np.zeros(len(y))
    probas_oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        pp = m.predict_proba(X[te])[:, 1]
        preds_oof[te] = p
        probas_oof[te] = pp
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p))
        precs.append(precision_score(y[te], p))
        recs.append(recall_score(y[te], p))
        aucs.append(roc_auc_score(y[te], pp))
    return {
        'model': name,
        'accuracy_mean': np.mean(accs), 'accuracy_std': np.std(accs),
        'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
        'precision_mean': np.mean(precs), 'precision_std': np.std(precs),
        'recall_mean': np.mean(recs), 'recall_std': np.std(recs),
        'roc_auc_mean': np.mean(aucs), 'roc_auc_std': np.std(aucs),
    }, preds_oof, probas_oof

# Feature subsets
X_behav = X_all[behav_cols].values
X_content = X_all[content_cols].values
X_full = X_all.values

# Standardize for logistic regression
scaler_b = StandardScaler().fit(X_behav)
scaler_c = StandardScaler().fit(X_content)
scaler_a = StandardScaler().fit(X_full)
Xb_s = scaler_b.transform(X_behav)
Xc_s = scaler_c.transform(X_content)
Xa_s = scaler_a.transform(X_full)

results = []
oof_probas = {}

experiments = [
    ('LogReg-Content',     LogisticRegression(max_iter=2000, random_state=RNG), Xc_s),
    ('LogReg-Behavioral',  LogisticRegression(max_iter=2000, random_state=RNG), Xb_s),
    ('LogReg-Fusion',      LogisticRegression(max_iter=2000, random_state=RNG), Xa_s),
    ('RF-Content',         RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1), X_content),
    ('RF-Behavioral',      RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1), X_behav),
    ('RF-Fusion',          RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1), X_full),
    ('GB-Behavioral',      GradientBoostingClassifier(n_estimators=200, random_state=RNG), X_behav),
    ('GB-Fusion',          GradientBoostingClassifier(n_estimators=200, random_state=RNG), X_full),
]

for name, model, X in experiments:
    res, _, probas = cv_evaluate(X, y, model, name)
    results.append(res)
    oof_probas[name] = probas
    print(f"{name:20s}  acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
          f"f1={res['f1_mean']:.4f}  AUC={res['roc_auc_mean']:.4f}")

table2 = pd.DataFrame(results)
table2.to_csv(f'{OUT}/table2_classifier_performance.csv', index=False)


# ============================================================
# 4. ADVERSARIAL ROBUSTNESS (Table 3) — the headline contribution
# ============================================================

print("\n" + "=" * 70)
print("Adversarial robustness: LLM-laundering simulation (Table 3)")
print("=" * 70)
print("Threat model: an adversary uses an LLM to launder all text content")
print("so it is statistically indistinguishable from human-written text.")
print("We simulate this by progressively perturbing CONTENT features only.")

def llm_launder(X_df, content_cols, severity, human_pool_arr, rng_seed=RNG):
    """
    Simulate LLM text laundering at increasing severity:
      0.0 — no change
      0.5 — content features replaced 50% with values sampled from human dist
      1.0 — content features fully replaced with human-distribution samples
    Behavioral features are UNTOUCHED (the LLM cannot rewrite account history).

    human_pool_arr: 2D array of shape (n_humans, n_content_features) drawn
    from the *training-set* human distribution. Using the training pool
    prevents test-set leakage.
    """
    rng = np.random.default_rng(rng_seed)
    X_new = X_df.copy()
    n = len(X_new)
    for col_idx, c in enumerate(content_cols):
        sampled = human_pool_arr[rng.integers(0, len(human_pool_arr), size=n), col_idx]
        mask = rng.random(n) < severity
        X_new.loc[mask, c] = sampled[mask]
    return X_new

severities = [0.0, 0.25, 0.5, 0.75, 1.0]
adv_rows = []

# Use train/test split for adversarial eval (we'll attack the test set)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.3, random_state=RNG, stratify=y
)

# Train each classifier ONCE on clean training data
def fit_and_get(X_train_df, X_test_df, cols, model_cls, **kw):
    m = model_cls(**kw)
    m.fit(X_train_df[cols].values, y_train)
    return m

rf_content_clf = fit_and_get(X_train, X_test, content_cols,
                              RandomForestClassifier, n_estimators=300,
                              random_state=RNG, n_jobs=-1)
rf_behav_clf = fit_and_get(X_train, X_test, behav_cols,
                            RandomForestClassifier, n_estimators=300,
                            random_state=RNG, n_jobs=-1)
rf_fusion_clf = fit_and_get(X_train, X_test, behav_cols + content_cols,
                             RandomForestClassifier, n_estimators=300,
                             random_state=RNG, n_jobs=-1)

# Build laundering pool from training-set humans only (avoids test leakage)
human_pool_train = X_train.loc[y_train == 0, content_cols].values

for sev in severities:
    X_test_adv = llm_launder(X_test, content_cols, sev,
                              human_pool_train, rng_seed=RNG + int(sev*100))
    for name, clf, cols in [
        ('RF-Content', rf_content_clf, content_cols),
        ('RF-Behavioral', rf_behav_clf, behav_cols),
        ('RF-Fusion', rf_fusion_clf, behav_cols + content_cols),
    ]:
        p = clf.predict(X_test_adv[cols].values)
        pp = clf.predict_proba(X_test_adv[cols].values)[:, 1]
        adv_rows.append({
            'severity': sev, 'model': name,
            'accuracy': accuracy_score(y_test, p),
            'f1': f1_score(y_test, p),
            'precision': precision_score(y_test, p),
            'recall': recall_score(y_test, p),
            'roc_auc': roc_auc_score(y_test, pp),
        })

table3 = pd.DataFrame(adv_rows)
table3.to_csv(f'{OUT}/table3_adversarial_robustness.csv', index=False)
print(table3.to_string(index=False))


# ============================================================
# 5. FEATURE IMPORTANCE (Table 4)
# ============================================================

print("\n" + "=" * 70)
print("Feature importance from RF-Fusion (Table 4)")
print("=" * 70)

importances = rf_fusion_clf.feature_importances_
feat_names = behav_cols + content_cols
table4 = pd.DataFrame({
    'feature': feat_names,
    'importance': importances,
    'feature_family': ['behavioral'] * len(behav_cols) + ['content'] * len(content_cols),
}).sort_values('importance', ascending=False).reset_index(drop=True)
table4.to_csv(f'{OUT}/table4_feature_importance.csv', index=False)
print(table4.head(15).to_string(index=False))


# ============================================================
# 6. FIGURES
# ============================================================

print("\n" + "=" * 70)
print("Generating figures")
print("=" * 70)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})

# --- Fig 1: Feature distributions by class ---
key_feats = ['statuses_per_day', 'friends_to_followers', 'log_followers',
             'screen_name_digit_ratio', 'has_description', 'account_age_days']
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, f in zip(axes.flatten(), key_feats):
    h = X_all.loc[y == 0, f].values
    b = X_all.loc[y == 1, f].values
    if f in ('statuses_per_day', 'account_age_days'):
        h = np.log1p(h); b = np.log1p(b)
        xl = f'log(1 + {f})'
    elif f == 'friends_to_followers':
        h = np.log1p(h); b = np.log1p(b)
        xl = 'log(1 + friends/followers)'
    else:
        xl = f
    bins = np.linspace(min(h.min(), b.min()), max(h.max(), b.max()), 40)
    ax.hist(h, bins=bins, alpha=0.55, label='Human', density=True, color='#2b8cbe')
    ax.hist(b, bins=bins, alpha=0.55, label='Bot', density=True, color='#e34a33')
    ax.set_xlabel(xl); ax.set_ylabel('Density')
    ax.legend(fontsize=8)
fig.suptitle('Figure 1. Distributional separation between human and bot accounts '
             'on selected content-agnostic behavioral features',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_feature_distributions.png")

# --- Fig 2: ROC curves ---
fig, ax = plt.subplots(figsize=(7, 6))
for name in ['RF-Content', 'RF-Behavioral', 'RF-Fusion']:
    fpr, tpr, _ = roc_curve(y, oof_probas[name])
    auc = roc_auc_score(y, oof_probas[name])
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Chance')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Figure 2. ROC curves — 5-fold cross-validation', fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_roc_curves.png")

# --- Fig 3: Adversarial degradation ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric in zip(axes, ['f1', 'roc_auc']):
    for name in ['RF-Content', 'RF-Behavioral', 'RF-Fusion']:
        sub = table3[table3['model'] == name]
        ax.plot(sub['severity'], sub[metric], 'o-', lw=2, markersize=8, label=name)
    ax.set_xlabel('LLM-laundering severity\n(0 = clean, 1 = full text laundering)')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} vs. adversarial severity')
    ax.legend(); ax.set_ylim(0, 1.02)
fig.suptitle('Figure 3. Behavioral fingerprinting degrades gracefully under '
             'adversarial text laundering', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_adversarial_degradation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_adversarial_degradation.png")

# --- Fig 4: Feature importance ---
top_n = 15
top = table4.head(top_n).iloc[::-1]
colors = ['#2b8cbe' if fam == 'behavioral' else '#e34a33'
          for fam in top['feature_family']]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.barh(top['feature'], top['importance'], color=colors)
ax.set_xlabel('Importance (RF-Fusion, Gini)')
ax.set_title(f'Figure 4. Top {top_n} features by importance\n'
             '(blue = behavioral, red = content)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_feature_importance.png")


# ============================================================
# 7. GENERALIZATION ON TWIBOT-20 SAMPLE (unsupervised)
# ============================================================

print("\n" + "=" * 70)
print("Cross-dataset generalization: TwiBot-20 sample (unlabeled)")
print("=" * 70)

with open(ensure_twibot20()) as f:
    tb20 = json.load(f)

# Convert TwiBot-20 profile dicts to a DataFrame matching dataset-1 columns
rows = []
for u in tb20:
    p = u.get('profile', {}) or {}
    rows.append({
        'id_str': str(p.get('id_str', u.get('ID', ''))).strip(),
        'screen_name': str(p.get('screen_name', '') or '').strip(),
        'name': str(p.get('name', '') or '').strip(),
        'description': p.get('description'),
        'location': p.get('location'),
        'url': p.get('url'),
        'followers_count': int(p.get('followers_count') or 0),
        'friends_count': int(p.get('friends_count') or 0),
        'listed_count': int(p.get('listed_count') or 0),
        'statuses_count': int(p.get('statuses_count') or 0),
        'favourites_count': int(p.get('favourites_count') or 0),
        'created_at': p.get('created_at'),
        'verified': p.get('verified', False),
        'default_profile': p.get('default_profile', False),
        'default_profile_image': p.get('default_profile_image', False),
    })
df2 = pd.DataFrame(rows)
print(f"TwiBot-20 sample size: {len(df2)}")

# Engineer behavioral features only (no labels, so unsupervised)
X2 = engineer_behavioral_features(df2)
X2 = X2.replace([np.inf, -np.inf], np.nan).fillna(X2.median(numeric_only=True))

# Score each user with the RF-Behavioral classifier trained on Dataset 1
rf_behav_full = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
rf_behav_full.fit(X_all[behav_cols].values, y)
tb20_proba = rf_behav_full.predict_proba(X2[behav_cols].values)[:, 1]

generalization = pd.DataFrame({
    'screen_name': df2['screen_name'],
    'followers_count': df2['followers_count'],
    'statuses_count': df2['statuses_count'],
    'bot_probability': tb20_proba,
}).sort_values('bot_probability', ascending=False)
generalization.to_csv(f'{OUT}/twibot20_generalization.csv', index=False)
print(f"Median bot-probability on TwiBot-20 sample: {np.median(tb20_proba):.3f}")
print(f"Fraction flagged at >0.5: {(tb20_proba > 0.5).mean():.3f}")
print(f"Fraction flagged at >0.7: {(tb20_proba > 0.7).mean():.3f}")
print("\nTop 10 highest-scored accounts on TwiBot-20 sample:")
print(generalization.head(10).to_string(index=False))


# ============================================================
# 8. SUMMARY SAVE
# ============================================================

summary = {
    'dataset_1_size': len(df1),
    'dataset_1_bot_fraction': float(y.mean()),
    'dataset_2_size': len(df2),
    'n_behavioral_features': len(behav_cols),
    'n_content_features': len(content_cols),
    'best_clean_auc_RF_Fusion': float(table2.loc[table2['model'] == 'RF-Fusion', 'roc_auc_mean'].iloc[0]),
    'best_clean_auc_RF_Behavioral': float(table2.loc[table2['model'] == 'RF-Behavioral', 'roc_auc_mean'].iloc[0]),
    'best_clean_auc_RF_Content': float(table2.loc[table2['model'] == 'RF-Content', 'roc_auc_mean'].iloc[0]),
    'adv_full_laundering_AUC_content': float(table3[(table3['severity']==1.0) & (table3['model']=='RF-Content')]['roc_auc'].iloc[0]),
    'adv_full_laundering_AUC_behavioral': float(table3[(table3['severity']==1.0) & (table3['model']=='RF-Behavioral')]['roc_auc'].iloc[0]),
    'adv_full_laundering_AUC_fusion': float(table3[(table3['severity']==1.0) & (table3['model']=='RF-Fusion')]['roc_auc'].iloc[0]),
}
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(json.dumps(summary, indent=2))
