"""
run_cross_dataset_generalization.py
--------------------------------------
Tests whether the behavioral model generalizes ACROSS the two independent
datasets, or whether it has just learned quirks specific to one bot
campaign/population:

  A) Train on Dataset 1 (1104 users)  -> Test on Dataset 2 (19510 users)
  B) Train on Dataset 2 (19510 users) -> Test on Dataset 1 (1104 users)

Only features computable identically in BOTH datasets are used, so the
comparison is apples-to-apples:
  - The original 24 behavioral features (paper 1), computed by the same
    features.py::engineer_behavioral_features in both cases.
  - A common subset of the "extra" engineered features that exist in both
    extra_features.py (dataset 1) and extra_features_v2.py (dataset 2):
    everything except dataset 1's 'tweeted_on_creation_day_or_week' flag
    (not available in dataset 2) and with retweet_ratio (dataset 1) /
    retweet_ratio_approx (dataset 2) aligned under one shared name.

As a within-dataset reference point, the usual 5-fold CV AUC for each
dataset (from earlier scripts) is also reported alongside.

Usage:
    python3 run_cross_dataset_generalization.py \
        /path/to/users_with_retweets.xlsx /path/to/all_users.xlsx
"""
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from features import engineer_behavioral_features
from load_custom_dataset import load_custom
from extra_features import engineer_extra_features
from load_dataset2 import load_dataset2
from extra_features_v2 import engineer_extra_features_v2

RNG = 42
np.random.seed(RNG)

XLSX1 = sys.argv[1] if len(sys.argv) > 1 else "users_with_retweets.xlsx"
XLSX2 = sys.argv[2] if len(sys.argv) > 2 else "all_users.xlsx"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT1 = os.path.join(ROOT, "results_custom")
OUT2 = os.path.join(ROOT, "results_dataset2")
os.makedirs(OUT1, exist_ok=True)
os.makedirs(OUT2, exist_ok=True)

# the common "extra" feature set, aligned by name across both datasets
COMMON_EXTRA_COLS = [
    'prop_original_tweet', 'prop_retweet_with_comment', 'prop_reply', 'prop_plain_retweet',
    'retweet_ratio',  # aligned name; dataset 2's retweet_ratio_approx is renamed to this
    'mean_no_hashtags', 'mean_no_words', 'mean_user_mentions_per_tweet',
    'unique_mention_rate_per_tweet', 'mean_no_media_per_tweet', 'no_languages',
    'log_tweet_frequency', 'log_time_between_tweets', 'log_mean_favourites_per_tweet',
    'log_mean_retweets_per_tweet', 'log_retweet_as_tweet_rate', 'log_max_tweets_per_hour',
    'log_max_tweets_per_day', 'log_max_occurence_of_same_gap', 'log_media_count',
]


def build_dataset1_matrix():
    df_raw = load_custom(XLSX1)
    df = df_raw.dropna(subset=["bot"]).copy()
    df["bot"] = df["bot"].astype(int)
    ref_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    X24 = engineer_behavioral_features(df, ref_date=ref_date)
    Xe = engineer_extra_features(df).rename(columns={'retweet_ratio': 'retweet_ratio'})
    # dataset 1's extra_features.py already outputs a column literally named
    # 'retweet_ratio' (from the raw column of the same name) -- no rename needed,
    # it's included here for symmetry/clarity with dataset 2's rename below.
    X = X24.join(Xe[COMMON_EXTRA_COLS])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    y = df["bot"].values
    return X, y


def build_dataset2_matrix():
    df_raw, _ = load_dataset2(XLSX2)
    df = df_raw.dropna(subset=["bot"]).copy()
    df["bot"] = df["bot"].astype(int)
    ref_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    X24 = engineer_behavioral_features(df, ref_date=ref_date)
    Xe = engineer_extra_features_v2(df).rename(columns={'retweet_ratio_approx': 'retweet_ratio'})
    X = X24.join(Xe[COMMON_EXTRA_COLS])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    y = df["bot"].values
    return X, y


def metrics(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        'accuracy': accuracy_score(y_true, pred),
        'f1': f1_score(y_true, pred, zero_division=0),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred),
        'roc_auc': roc_auc_score(y_true, proba),
    }


print("=" * 70)
print("BUILDING FEATURE MATRICES (43 common features: 24 base + 19 extra)")
print("=" * 70)
X1, y1 = build_dataset1_matrix()
X2, y2 = build_dataset2_matrix()
common_cols = list(X1.columns)
assert list(X2.columns) == common_cols, "Column mismatch between datasets!"
print(f"Dataset 1: {X1.shape}, bot rate={y1.mean():.3f}")
print(f"Dataset 2: {X2.shape}, bot rate={y2.mean():.3f}")
print(f"Common feature columns ({len(common_cols)}): {common_cols}")

results = []

# A) Train on dataset 1 -> test on dataset 2
clf_A = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1, class_weight='balanced')
clf_A.fit(X1.values, y1)
proba_A = clf_A.predict_proba(X2.values)[:, 1]
m = metrics(y2, proba_A)
m.update({'direction': 'Train=D1 -> Test=D2', 'n_train': len(y1), 'n_test': len(y2)})
results.append(m)

# B) Train on dataset 2 -> test on dataset 1
clf_B = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1, class_weight='balanced')
clf_B.fit(X2.values, y2)
proba_B = clf_B.predict_proba(X1.values)[:, 1]
m = metrics(y1, proba_B)
m.update({'direction': 'Train=D2 -> Test=D1', 'n_train': len(y2), 'n_test': len(y1)})
results.append(m)

table = pd.DataFrame(results)[['direction', 'n_train', 'n_test', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc']]
print("\nCross-dataset generalization results:")
print(table.to_string(index=False))

table.to_csv(f'{ROOT}/results_cross_dataset_generalization.csv', index=False)

# feature importance from each direction's model, for interpretation
imp_A = pd.DataFrame({'feature': common_cols, 'importance': clf_A.feature_importances_}) \
    .sort_values('importance', ascending=False).reset_index(drop=True)
imp_B = pd.DataFrame({'feature': common_cols, 'importance': clf_B.feature_importances_}) \
    .sort_values('importance', ascending=False).reset_index(drop=True)
print("\nTop 10 features -- model trained on Dataset 1:")
print(imp_A.head(10).to_string(index=False))
print("\nTop 10 features -- model trained on Dataset 2:")
print(imp_B.head(10).to_string(index=False))

summary = {
    'common_features_used': common_cols,
    'dataset1_n': len(y1), 'dataset1_bot_rate': float(y1.mean()),
    'dataset2_n': len(y2), 'dataset2_bot_rate': float(y2.mean()),
    'train_d1_test_d2_auc': float(results[0]['roc_auc']),
    'train_d2_test_d1_auc': float(results[1]['roc_auc']),
}
with open(f'{ROOT}/results_cross_dataset_generalization.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\nSaved to results_cross_dataset_generalization.csv/.json")
