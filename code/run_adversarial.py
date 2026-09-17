"""
run_adversarial.py
-------------------
Roadmap step 3: adversarial robustness test, adapted from the original
paper's run_extended.py ("real text-level LLM laundering").

Idea: simulate an adversary who uses an LLM to rewrite tweet text so it
looks more human (fewer URLs/hashtags/mentions, no ALL-CAPS bursts, no
excessive punctuation), at increasing severity (fraction of the test
population rewritten: 0%, 25%, 50%, 75%, 100%). Re-evaluate three models
that were trained ONCE on clean data:
    RF-Content    (12 content features -- text-derived)
    RF-Behavioral (45 behavioral features -- account/activity-derived)
    RF-Fusion     (57 = 45 + 12 combined)
Hypothesis (from Katyal 2026): behavioral features are laundering-proof
because they are not derived from tweet text, so RF-Behavioral's AUC
should stay flat as severity increases, while RF-Content's AUC should
degrade.

IMPORTANT CAVEAT (documented here and in the thesis report): in our
dataset the 12 content features are computed from a small SAMPLE of each
user's tweets (the `status` field built by load_custom_dataset.py,
<=20 concatenated tweets), while the 21 extra behavioral features come
from aggregate statistics the student pre-computed over each user's FULL
tweet history. Laundering only rewrites the small sample text, not the
underlying full-history aggregates -- so behavioral-feature invariance
here is partly a consequence of the data pipeline (they were never
derived from the rewritable field to begin with), not purely an emergent
empirical finding. This should be stated plainly when presenting results.

Because content features require tweet text, this experiment necessarily
runs on the ~162-174 user subset that has sampled text (same subset as
the earlier RF-Content / RF-Fusion experiments in run_analysis_custom.py).

Usage:
    python3 run_adversarial.py /path/to/users_with_retweets.xlsx
"""
import os
import sys
import re
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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

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


# ============================================================
# LLM-style rewrite (same operations as the original paper)
# ============================================================
def llm_style_rewrite(text, target_stats, rng):
    if not isinstance(text, str) or not text.strip():
        return text
    s = text
    urls = re.findall(r'https?://\S+', s)
    human_url_count = max(0, int(round(target_stats['url_mean'] + rng.normal(0, target_stats['url_std']))))
    if len(urls) > human_url_count:
        for u in urls[human_url_count:]:
            s = s.replace(u, '', 1)
    hashtags = re.findall(r'#\w+', s)
    human_ht_count = max(0, int(round(target_stats['hashtag_mean'] + rng.normal(0, target_stats['hashtag_std']))))
    if len(hashtags) > human_ht_count:
        for h in hashtags[human_ht_count:]:
            s = s.replace(h, '', 1)
    mentions = re.findall(r'@\w+', s)
    human_mention_count = max(0, int(round(target_stats['mention_mean'] + rng.normal(0, target_stats['mention_std']))))
    if len(mentions) > human_mention_count:
        for m in mentions[human_mention_count:]:
            s = s.replace(m, '', 1)
    s = re.sub(r'([A-Z]{5,})', lambda m: m.group(1).capitalize(), s)
    s = re.sub(r'([!?])\1{2,}', r'\1', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s


def compute_human_text_stats(df):
    human_status = df.loc[df['bot'] == 0, 'status'].fillna('').astype(str)
    return {
        'url_mean': human_status.str.count(r'https?://').mean(),
        'url_std': max(human_status.str.count(r'https?://').std(), 1e-6),
        'hashtag_mean': human_status.str.count(r'#\w+').mean(),
        'hashtag_std': max(human_status.str.count(r'#\w+').std(), 1e-6),
        'mention_mean': human_status.str.count(r'@\w+').mean(),
        'mention_std': max(human_status.str.count(r'@\w+').std(), 1e-6),
    }


def apply_text_laundering(df, target_stats, severity, rng):
    new = df.copy()
    mask = rng.random(len(new)) < severity
    for idx in np.where(mask)[0]:
        new.iloc[idx, new.columns.get_loc('status')] = llm_style_rewrite(
            new.iloc[idx]['status'], target_stats, rng)
    return new


print("=" * 70)
print("LOADING DATA")
print("=" * 70)
df_raw = load_custom(XLSX_PATH)
df_labeled = df_raw.dropna(subset=["bot"]).copy()
df_labeled["bot"] = df_labeled["bot"].astype(int)
df_text = df_labeled[df_labeled["status"].str.len() > 0].reset_index(drop=True)
print(f"Users with tweet text (this experiment's population): {len(df_text)}")
print(f"Bot fraction in this subset: {df_text['bot'].mean():.3f}")

X_train_df, X_test_df, y_train, y_test = train_test_split(
    df_text, df_text["bot"].values, test_size=0.3, random_state=RNG, stratify=df_text["bot"].values)

target_stats = compute_human_text_stats(X_train_df)


def build_features(df):
    X_all, behav_cols, content_cols = engineer_all(df)
    X_all = engineer_behavioral_features(df, ref_date=REF_DATE).join(X_all[content_cols])
    X_extra = engineer_extra_features(df)
    X_all = X_all.join(X_extra)
    behav_cols = behav_cols + EXTRA_FEATURE_NAMES
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_all.median(numeric_only=True))
    return X_all, behav_cols, content_cols


X_train_feat, behav_cols, content_cols = build_features(X_train_df)

rf_content = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
rf_behav = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
rf_fusion = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
rf_content.fit(X_train_feat[content_cols].values, y_train)
rf_behav.fit(X_train_feat[behav_cols].values, y_train)
rf_fusion.fit(X_train_feat[behav_cols + content_cols].values, y_train)

print("\n" + "=" * 70)
print("ADVERSARIAL TEXT LAUNDERING (severity = fraction of test set rewritten)")
print("=" * 70)

rows = []
for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rng = np.random.default_rng(RNG + int(sev * 100))
    test_laundered = apply_text_laundering(X_test_df, target_stats, sev, rng)
    X_test_feat, _, _ = build_features(test_laundered)
    for name, clf, cols in [
        ('RF-Content', rf_content, content_cols),
        ('RF-Behavioral', rf_behav, behav_cols),
        ('RF-Fusion', rf_fusion, behav_cols + content_cols),
    ]:
        proba = clf.predict_proba(X_test_feat[cols].values)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            'severity': sev, 'model': name,
            'accuracy': accuracy_score(y_test, pred),
            'f1': f1_score(y_test, pred, zero_division=0),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, proba),
        })

table6 = pd.DataFrame(rows)
table6.to_csv(f'{OUT}/table6_adversarial_laundering.csv', index=False)
print(table6.to_string(index=False))

# ============================================================
# Figure: AUC vs severity, one line per model
# ============================================================
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(7.5, 6))
colors = {'RF-Content': '#e34a33', 'RF-Behavioral': '#2b8cbe', 'RF-Fusion': '#31a354'}
for name in ['RF-Content', 'RF-Behavioral', 'RF-Fusion']:
    sub = table6[table6['model'] == name]
    ax.plot(sub['severity'], sub['roc_auc'], marker='o', lw=2.5, label=name, color=colors[name])
ax.set_xlabel('Laundering severity (fraction of test users rewritten)')
ax.set_ylabel('ROC-AUC')
ax.set_title('Adversarial robustness: AUC vs. LLM text-laundering severity', fontweight='bold')
ax.set_ylim(0.4, 1.0)
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(f'{OUT}/fig10_adversarial_robustness.png', dpi=150, bbox_inches='tight')
plt.close()

auc_drop = {}
for name in ['RF-Content', 'RF-Behavioral', 'RF-Fusion']:
    sub = table6[table6['model'] == name].sort_values('severity')
    auc_drop[name] = float(sub.iloc[0]['roc_auc'] - sub.iloc[-1]['roc_auc'])

print("\nAUC drop from severity=0.0 to severity=1.0:")
for k, v in auc_drop.items():
    print(f"  {k}: {v:+.3f}")

summary_path = f'{OUT}/summary.json'
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
summary['adversarial_robustness'] = {
    'population_n': len(df_text),
    'auc_drop_severity_0_to_1': auc_drop,
    'caveat': ('Behavioral features are computed upstream of the sampled tweet '
               'text used for laundering, so their invariance here is partly '
               'a pipeline property, not purely an emergent empirical result.'),
}
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs written to: {OUT}")
