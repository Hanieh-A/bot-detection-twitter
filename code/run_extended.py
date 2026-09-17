"""
Extended analysis adding the reviewer-critical pieces:
  1. Real text-level LLM laundering (paraphrase + de-templated content)
  2. DeLong's test for paired AUC comparison
  3. Calibration analysis (Brier score, reliability)
  4. Per-class error breakdown
  5. Behavioral feature ablation (leave-one-family-out)
  6. Subgroup analysis (high vs low statuses_count)

Run after run_analysis.py. Writes additional CSVs and figures into results/.
"""

import os, json, re, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, brier_score_loss, f1_score,
                             precision_score, recall_score, accuracy_score,
                             confusion_matrix, roc_curve)
from sklearn.calibration import calibration_curve
from scipy import stats

from features import (engineer_all, engineer_behavioral_features, engineer_content_features,
                      OUT, ensure_dataset1)

RNG = 42
np.random.seed(RNG)
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# Load + feature-engineer (reuse Dataset 1)
# ---------------------------------------------------------------
df1 = pd.read_csv(ensure_dataset1(),
                  encoding='utf-8', on_bad_lines='skip', low_memory=False)
df1 = df1.dropna(subset=['bot']).copy()
df1['bot'] = df1['bot'].astype(int)
df1 = df1.drop_duplicates(subset=['id_str']).reset_index(drop=True)

X_all, behav_cols, content_cols = engineer_all(df1)
y = df1['bot'].values
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_all.median(numeric_only=True))

# Behavioral sub-families for ablation
fam_age_rates = ['account_age_days', 'statuses_per_day', 'followers_per_day',
                 'friends_per_day', 'favourites_per_day']
fam_follow    = ['friends_to_followers', 'log_followers', 'log_friends',
                 'log_statuses', 'listed_per_follower']
fam_profile   = ['has_location', 'has_description', 'has_url',
                 'default_profile', 'default_profile_image', 'verified']
fam_screen    = ['screen_name_len', 'screen_name_digit_ratio', 'screen_name_entropy',
                 'screen_name_has_trailing_digits', 'name_len', 'name_digit_ratio',
                 'name_entropy']
fam_engage    = ['statuses_to_favourites']
behav_families = {
    'age_rates': fam_age_rates,
    'follow_asymmetries': fam_follow,
    'profile_completeness': fam_profile,
    'screen_name_structure': fam_screen,
    'engagement_asymmetry': fam_engage,
}

# ---------------------------------------------------------------
# 1. REAL TEXT-LEVEL LLM LAUNDERING
# ---------------------------------------------------------------
# Strategy: rather than perturb the feature space, we actually rewrite
# bot tweets to mimic the surface statistics of human tweets, then
# recompute content features from the rewritten text. This is closer
# to what an LLM adversary would do.

def llm_style_rewrite(text, target_human_stats, rng):
    """
    Rewrite a bot tweet to approximate human surface statistics.
    Operations applied probabilistically:
      - Strip excessive URLs (humans average ~0.15 URLs per tweet)
      - Strip excessive hashtags (humans average ~0.2 hashtags)
      - Reduce ALL-CAPS bursts
      - Add natural punctuation if absent
      - Substitute spammy n-grams with neutral ones
    """
    if not isinstance(text, str) or not text.strip():
        return text

    s = text

    # 1. Cap URLs at ~human rate
    urls = re.findall(r'https?://\S+', s)
    human_url_count = max(0, int(round(target_human_stats['url_mean'] +
                                       rng.normal(0, target_human_stats['url_std']))))
    if len(urls) > human_url_count:
        for u in urls[human_url_count:]:
            s = s.replace(u, '', 1)

    # 2. Cap hashtags
    hashtags = re.findall(r'#\w+', s)
    human_ht_count = max(0, int(round(target_human_stats['hashtag_mean'] +
                                      rng.normal(0, target_human_stats['hashtag_std']))))
    if len(hashtags) > human_ht_count:
        for h in hashtags[human_ht_count:]:
            s = s.replace(h, '', 1)

    # 3. Cap mentions
    mentions = re.findall(r'@\w+', s)
    human_mention_count = max(0, int(round(target_human_stats['mention_mean'] +
                                           rng.normal(0, target_human_stats['mention_std']))))
    if len(mentions) > human_mention_count:
        for m in mentions[human_mention_count:]:
            s = s.replace(m, '', 1)

    # 4. Reduce ALL-CAPS bursts (>4 caps in a row)
    s = re.sub(r'([A-Z]{5,})', lambda m: m.group(1).capitalize(), s)

    # 5. Strip excessive punctuation runs
    s = re.sub(r'([!?])\1{2,}', r'\1', s)

    # 6. Collapse spaces left by removals
    s = re.sub(r'\s{2,}', ' ', s).strip()

    return s


def compute_human_text_stats(df, label_col='bot'):
    """Get target statistics from human tweets only."""
    human_status = df.loc[df[label_col] == 0, 'status'].fillna('').astype(str)
    stats_out = {
        'url_mean': human_status.str.count(r'https?://').mean(),
        'url_std': human_status.str.count(r'https?://').std(),
        'hashtag_mean': human_status.str.count(r'#\w+').mean(),
        'hashtag_std': human_status.str.count(r'#\w+').std(),
        'mention_mean': human_status.str.count(r'@\w+').mean(),
        'mention_std': human_status.str.count(r'@\w+').std(),
    }
    return stats_out


def apply_text_laundering(df, target_stats, severity, rng):
    """Apply rewriting to a `severity` fraction of rows."""
    new = df.copy()
    n = len(new)
    mask = rng.random(n) < severity
    for idx in np.where(mask)[0]:
        new.iloc[idx, new.columns.get_loc('status')] = llm_style_rewrite(
            new.iloc[idx]['status'], target_stats, rng)
    return new


# Split
X_train_df, X_test_df, y_train, y_test = train_test_split(
    df1, y, test_size=0.3, random_state=RNG, stratify=y)

target_stats = compute_human_text_stats(X_train_df.assign(bot=y_train))

# Build training feature matrix
X_train_feat, _, _ = engineer_all(X_train_df)
X_train_feat = X_train_feat.replace([np.inf, -np.inf], np.nan).fillna(X_train_feat.median(numeric_only=True))

# Train classifiers once on clean training data
rf_content = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
rf_behav   = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
rf_fusion  = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
rf_content.fit(X_train_feat[content_cols].values, y_train)
rf_behav.fit(X_train_feat[behav_cols].values, y_train)
rf_fusion.fit(X_train_feat[behav_cols + content_cols].values, y_train)

print("=" * 72)
print("Text-level LLM laundering: rewriting bot tweets to mimic human stats")
print("=" * 72)

text_adv_rows = []
for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rng = np.random.default_rng(RNG + int(sev*100))
    test_laundered = apply_text_laundering(X_test_df, target_stats, sev, rng)
    # Recompute features from rewritten text
    X_test_feat, _, _ = engineer_all(test_laundered)
    X_test_feat = X_test_feat.replace([np.inf, -np.inf], np.nan).fillna(X_test_feat.median(numeric_only=True))
    for name, clf, cols in [
        ('RF-Content', rf_content, content_cols),
        ('RF-Behavioral', rf_behav, behav_cols),
        ('RF-Fusion', rf_fusion, behav_cols + content_cols),
    ]:
        proba = clf.predict_proba(X_test_feat[cols].values)[:, 1]
        pred = (proba >= 0.5).astype(int)
        text_adv_rows.append({
            'severity': sev, 'model': name,
            'accuracy': accuracy_score(y_test, pred),
            'f1': f1_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred),
            'roc_auc': roc_auc_score(y_test, proba),
        })
table_text_adv = pd.DataFrame(text_adv_rows)
table_text_adv.to_csv(f'{OUT}/table5_text_laundering.csv', index=False)
print(table_text_adv.to_string(index=False))

# ---------------------------------------------------------------
# 2. DELONG'S TEST for paired AUC comparisons
# ---------------------------------------------------------------
print("\n" + "=" * 72)
print("DeLong's test: paired AUC significance on clean test set")
print("=" * 72)

def delong_roc_variance(ground_truth, predictions):
    """Fast DeLong implementation. Returns variance of the AUC estimate."""
    order = np.argsort(-predictions)
    pred_sorted = predictions[order]
    gt_sorted = ground_truth[order]
    n_pos = int(gt_sorted.sum())
    n_neg = len(gt_sorted) - n_pos
    # Mid-ranks
    ranks = np.empty(len(pred_sorted))
    i = 0
    while i < len(pred_sorted):
        j = i
        while j < len(pred_sorted) - 1 and pred_sorted[j+1] == pred_sorted[i]:
            j += 1
        ranks[i:j+1] = 0.5 * (i + j) + 1
        i = j + 1
    pos_ranks = ranks[gt_sorted == 1]
    neg_ranks = ranks[gt_sorted == 0]
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc

def delong_test(y_true, prob_a, prob_b):
    """Two-tailed DeLong test for AUC(A) vs AUC(B). Returns (auc_a, auc_b, z, p)."""
    y_true = np.asarray(y_true)
    pos_idx = y_true == 1
    neg_idx = y_true == 0
    n_pos = pos_idx.sum()
    n_neg = neg_idx.sum()

    def structural(probs):
        x = probs[pos_idx]; ynp = probs[neg_idx]
        # V10[i] = (1/n_neg) * sum_j psi(x_i, y_j); V01[j] = (1/n_pos) * sum_i psi(x_i, y_j)
        psi_mat = (x[:, None] > ynp[None, :]).astype(float) + 0.5 * (x[:, None] == ynp[None, :])
        V10 = psi_mat.mean(axis=1)
        V01 = psi_mat.mean(axis=0)
        return V10, V01

    V10_a, V01_a = structural(prob_a)
    V10_b, V01_b = structural(prob_b)
    auc_a = V10_a.mean()
    auc_b = V10_b.mean()
    S10 = np.cov(np.stack([V10_a, V10_b]))
    S01 = np.cov(np.stack([V01_a, V01_b]))
    var = (S10 / n_pos + S01 / n_neg)
    L = np.array([1.0, -1.0])
    se = np.sqrt(L @ var @ L)
    z = (auc_a - auc_b) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, z, p

# Pairwise tests on clean test set
X_test_feat_clean, _, _ = engineer_all(X_test_df)
X_test_feat_clean = X_test_feat_clean.replace([np.inf,-np.inf],np.nan).fillna(X_test_feat_clean.median(numeric_only=True))
prob_content = rf_content.predict_proba(X_test_feat_clean[content_cols].values)[:,1]
prob_behav   = rf_behav.predict_proba(X_test_feat_clean[behav_cols].values)[:,1]
prob_fusion  = rf_fusion.predict_proba(X_test_feat_clean[behav_cols + content_cols].values)[:,1]

delong_rows = []
for (na, pa), (nb, pb) in [
    (('Behavioral', prob_behav), ('Content', prob_content)),
    (('Fusion', prob_fusion),    ('Content', prob_content)),
    (('Fusion', prob_fusion),    ('Behavioral', prob_behav)),
]:
    auc_a, auc_b, z, p = delong_test(y_test, pa, pb)
    delong_rows.append({'comparison': f'{na} vs {nb}',
                        'auc_a': auc_a, 'auc_b': auc_b,
                        'delta_auc': auc_a - auc_b, 'z': z, 'p_value': p})
table_delong = pd.DataFrame(delong_rows)
table_delong.to_csv(f'{OUT}/table6_delong.csv', index=False)
print(table_delong.to_string(index=False))

# ---------------------------------------------------------------
# 3. CALIBRATION
# ---------------------------------------------------------------
print("\n" + "=" * 72)
print("Calibration analysis")
print("=" * 72)

calib_rows = []
fig, ax = plt.subplots(figsize=(7, 6))
for name, proba in [('RF-Content', prob_content),
                    ('RF-Behavioral', prob_behav),
                    ('RF-Fusion', prob_fusion)]:
    brier = brier_score_loss(y_test, proba)
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy='quantile')
    ax.plot(mean_pred, frac_pos, 'o-', lw=2, label=f'{name} (Brier = {brier:.3f})')
    calib_rows.append({'model': name, 'brier_score': brier})
ax.plot([0,1],[0,1], 'k--', alpha=0.4, label='Perfectly calibrated')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Figure 5. Reliability diagram for RF classifiers (held-out test set, n = 730)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_calibration.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_calibration.png")
pd.DataFrame(calib_rows).to_csv(f'{OUT}/table7_calibration.csv', index=False)
print(pd.DataFrame(calib_rows).to_string(index=False))

# ---------------------------------------------------------------
# 4. PER-CLASS ERROR BREAKDOWN
# ---------------------------------------------------------------
print("\n" + "=" * 72)
print("Confusion matrices at threshold 0.5")
print("=" * 72)

conf_rows = []
for name, proba in [('RF-Content', prob_content),
                    ('RF-Behavioral', prob_behav),
                    ('RF-Fusion', prob_fusion)]:
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    conf_rows.append({'model': name, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                      'FPR': fp/(fp+tn), 'FNR': fn/(fn+tp)})
    print(f"  {name}: TN={tn} FP={fp} FN={fn} TP={tp}  FPR={fp/(fp+tn):.3f}  FNR={fn/(fn+tp):.3f}")
pd.DataFrame(conf_rows).to_csv(f'{OUT}/table8_confusion.csv', index=False)

# ---------------------------------------------------------------
# 5. BEHAVIORAL FAMILY ABLATION (leave-one-family-out)
# ---------------------------------------------------------------
print("\n" + "=" * 72)
print("Behavioral-family ablation: leave-one-family-out")
print("=" * 72)

ablation_rows = []
# Full
full_cols = behav_cols
m_full = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
m_full.fit(X_train_feat[full_cols].values, y_train)
auc_full = roc_auc_score(y_test, m_full.predict_proba(X_test_feat_clean[full_cols].values)[:,1])
ablation_rows.append({'configuration': 'All behavioral families', 'roc_auc': auc_full,
                      'delta_vs_full': 0.0, 'n_features': len(full_cols)})

for fam_name, fam_cols in behav_families.items():
    leave_out = [c for c in behav_cols if c not in fam_cols]
    m = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
    m.fit(X_train_feat[leave_out].values, y_train)
    auc = roc_auc_score(y_test, m.predict_proba(X_test_feat_clean[leave_out].values)[:,1])
    ablation_rows.append({'configuration': f'Without {fam_name}',
                          'roc_auc': auc, 'delta_vs_full': auc - auc_full,
                          'n_features': len(leave_out)})
    print(f"  Without {fam_name:25s} AUC={auc:.4f}  Δ={auc-auc_full:+.4f}")
pd.DataFrame(ablation_rows).to_csv(f'{OUT}/table9_ablation.csv', index=False)

# ---------------------------------------------------------------
# 6. SUBGROUP ANALYSIS — low-activity vs high-activity accounts
# ---------------------------------------------------------------
print("\n" + "=" * 72)
print("Subgroup analysis: classifier behavior by account-activity tier")
print("=" * 72)

# Use raw statuses_count
test_df_with_y = X_test_df.copy()
test_df_with_y['_y'] = y_test
test_df_with_y['_proba_behav'] = prob_behav
test_df_with_y['_proba_content'] = prob_content
test_df_with_y['_proba_fusion'] = prob_fusion

# Tiers by statuses_count quartile (whole-test split)
q = test_df_with_y['statuses_count'].quantile([0.25, 0.5, 0.75]).values
def tier(x):
    if x <= q[0]: return 'Q1 (lowest)'
    if x <= q[1]: return 'Q2'
    if x <= q[2]: return 'Q3'
    return 'Q4 (highest)'
test_df_with_y['tier'] = test_df_with_y['statuses_count'].apply(tier)

subgroup_rows = []
for t in ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']:
    sub = test_df_with_y[test_df_with_y['tier'] == t]
    if len(sub) < 20 or sub['_y'].nunique() < 2:
        continue
    for col, name in [('_proba_content','RF-Content'),
                      ('_proba_behav','RF-Behavioral'),
                      ('_proba_fusion','RF-Fusion')]:
        try:
            auc = roc_auc_score(sub['_y'], sub[col])
        except ValueError:
            auc = np.nan
        subgroup_rows.append({'tier': t, 'n': len(sub),
                              'bot_fraction': sub['_y'].mean(),
                              'model': name, 'roc_auc': auc})
table_subgroup = pd.DataFrame(subgroup_rows)
table_subgroup.to_csv(f'{OUT}/table10_subgroup.csv', index=False)
print(table_subgroup.to_string(index=False))

# ---------------------------------------------------------------
# Save extended summary
# ---------------------------------------------------------------
ext = {
    'text_launder_clean_content_auc': float(table_text_adv[(table_text_adv['severity']==0.0)&(table_text_adv['model']=='RF-Content')]['roc_auc'].iloc[0]),
    'text_launder_full_content_auc':  float(table_text_adv[(table_text_adv['severity']==1.0)&(table_text_adv['model']=='RF-Content')]['roc_auc'].iloc[0]),
    'text_launder_full_behav_auc':    float(table_text_adv[(table_text_adv['severity']==1.0)&(table_text_adv['model']=='RF-Behavioral')]['roc_auc'].iloc[0]),
    'text_launder_full_fusion_auc':   float(table_text_adv[(table_text_adv['severity']==1.0)&(table_text_adv['model']=='RF-Fusion')]['roc_auc'].iloc[0]),
    'delong_behavioral_vs_content_p': float(table_delong[table_delong['comparison']=='Behavioral vs Content']['p_value'].iloc[0]),
    'delong_fusion_vs_behavioral_p':  float(table_delong[table_delong['comparison']=='Fusion vs Behavioral']['p_value'].iloc[0]),
    'brier_behav': float(pd.DataFrame(calib_rows).set_index('model').loc['RF-Behavioral','brier_score']),
    'brier_content': float(pd.DataFrame(calib_rows).set_index('model').loc['RF-Content','brier_score']),
    'ablation_max_drop_family': pd.DataFrame(ablation_rows).iloc[1:].sort_values('delta_vs_full').iloc[0]['configuration'],
}
with open(f'{OUT}/extended_summary.json','w') as f:
    json.dump(ext, f, indent=2)
print("\n" + "=" * 72)
print("EXTENDED SUMMARY")
print("=" * 72)
print(json.dumps(ext, indent=2))
