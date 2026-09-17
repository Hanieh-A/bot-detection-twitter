"""
run_shap.py
-----------
Roadmap step 2: SHAP-based interpretability for the final recommended
model: RF-Behavioral, 45 features (24 from Katyal 2026 + 21 extra),
unweighted, decision threshold 0.39 (see run_imbalance.py).

Produces (in ../results_custom/):
  - fig6_shap_summary_beeswarm.png   -- global view: direction + magnitude
  - fig7_shap_bar_importance.png     -- mean |SHAP| ranking (top 15)
  - fig8_shap_dependence_<feat>.png  -- how top-2 features drive the prediction
  - fig9_shap_waterfall_bot_example.png / _human_example.png
        -- per-user explanation for one correctly-flagged bot and one
           correctly-cleared human (useful slides for the defense)
  - table5_shap_importance.csv

Usage:
    python3 run_shap.py /path/to/users_with_retweets.xlsx
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
import shap

from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

from features import engineer_all, engineer_behavioral_features
from load_custom_dataset_v2 import load_custom_v2 as load_custom
from extra_features import engineer_extra_features, EXTRA_FEATURE_NAMES

RNG = 42
np.random.seed(RNG)
THRESHOLD = 0.39  # chosen in run_imbalance.py (best F1)

XLSX_PATH = sys.argv[1] if len(sys.argv) > 1 else "1000user_sheet.xlsx"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_custom")
os.makedirs(OUT, exist_ok=True)
REF_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)

print("=" * 70)
print("LOADING DATA & TRAINING FINAL MODEL")
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

X_df = X_all[behav_cols].copy()
y = df_labeled["bot"].values
ids = df_labeled["id_str"].values
screen_names = df_labeled["screen_name"].values

model = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
model.fit(X_df.values, y)
print(f"Model trained on n={len(y)} users, {X_df.shape[1]} features.")

# ============================================================
# SHAP values (TreeExplainer -- exact & fast for tree ensembles)
# ============================================================
print("\nComputing SHAP values (this can take a minute)...")
explainer = shap.TreeExplainer(model)
sv = explainer(X_df)

# shap>=0.4x returns an Explanation with shape (n, n_features, n_classes) for
# binary RandomForestClassifier -- take the "bot" (class 1) slice.
if sv.values.ndim == 3:
    sv_bot = shap.Explanation(
        values=sv.values[:, :, 1],
        base_values=sv.base_values[:, 1] if np.ndim(sv.base_values) > 1 else sv.base_values,
        data=sv.data,
        feature_names=sv.feature_names,
    )
else:
    sv_bot = sv

# ============================================================
# Fig 6: beeswarm summary (global direction + magnitude)
# ============================================================
plt.figure(figsize=(9, 7))
shap.summary_plot(sv_bot, X_df, max_display=15, show=False)
plt.title("SHAP summary -- what drives the bot prediction", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Fig 7: bar plot of mean |SHAP| (top 15)
# ============================================================
mean_abs_shap = np.abs(sv_bot.values).mean(axis=0)
table5 = pd.DataFrame({'feature': behav_cols, 'mean_abs_shap': mean_abs_shap}) \
    .sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
table5.to_csv(f'{OUT}/table5_shap_importance.csv', index=False)
print("\nTable 5 -- top 15 features by mean |SHAP|:")
print(table5.head(15).to_string(index=False))

plt.figure(figsize=(9, 7))
shap.summary_plot(sv_bot, X_df, plot_type="bar", max_display=15, show=False)
plt.title("Mean |SHAP value| -- top 15 behavioral features", fontsize=12, fontweight='bold')
plt.xlabel("mean(|SHAP value|)")
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_shap_bar_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Fig 8: dependence plots for the top-2 features
# ============================================================
top2 = table5['feature'].head(2).tolist()
for feat in top2:
    plt.figure(figsize=(7.5, 6))
    shap.dependence_plot(feat, sv_bot.values, X_df, show=False, interaction_index=None)
    plt.title(f"SHAP dependence -- {feat}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    safe_name = feat.replace('/', '_')
    plt.savefig(f'{OUT}/fig8_shap_dependence_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# Fig 9: individual waterfall explanations (one bot, one human -- both
# correctly classified with high confidence, for the defense slides)
# ============================================================
proba_bot = model.predict_proba(X_df.values)[:, 1]
pred = (proba_bot >= THRESHOLD).astype(int)

bot_correct_idx = np.where((y == 1) & (pred == 1))[0]
human_correct_idx = np.where((y == 0) & (pred == 0))[0]

if len(bot_correct_idx) > 0:
    # pick the most confidently-correct bot example
    best_bot_i = bot_correct_idx[np.argmax(proba_bot[bot_correct_idx])]
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(sv_bot[best_bot_i], max_display=12, show=False)
    plt.title(f"Why flagged as bot: @{screen_names[best_bot_i]} (p={proba_bot[best_bot_i]:.2f})",
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig9_shap_waterfall_bot_example.png', dpi=150, bbox_inches='tight')
    plt.close()

if len(human_correct_idx) > 0:
    best_human_i = human_correct_idx[np.argmin(proba_bot[human_correct_idx])]
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(sv_bot[best_human_i], max_display=12, show=False)
    plt.title(f"Why cleared as human: @{screen_names[best_human_i]} (p={proba_bot[best_human_i]:.2f})",
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig9_shap_waterfall_human_example.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# Update summary.json
# ============================================================
summary_path = f'{OUT}/summary.json'
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
summary['shap_interpretability'] = {
    'top5_features_by_mean_abs_shap': table5['feature'].head(5).tolist(),
    'model': 'RF-Behavioral, 45 features, unweighted, threshold=0.39',
}
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAll SHAP outputs written to: {OUT}")
