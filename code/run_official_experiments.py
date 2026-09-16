"""Official leakage-safe experiments for the Persian Twitter bot project.

The scripts in ``run_analysis_*.py`` are retained as historical experiments.
This runner is the source of results for corrected claims because it uses
fold-local preprocessing and nested threshold selection.

Examples (from repository root)::

    python code/run_official_experiments.py --dataset d1
    python code/run_official_experiments.py --dataset d2
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from experiment_config import (DATASET1_PATH, DATASET2_PATH, INNER_SPLITS,
                               OFFICIAL_RESULTS, OUTER_SPLITS, RF_PARAMS, SEED,
                               THRESHOLD_GRID)
from evaluation import nested_cv_evaluate
from extra_features import EXTRA_FEATURE_NAMES, engineer_extra_features
from extra_features_v2 import EXTRA_FEATURE_NAMES_V2, engineer_extra_features_v2
from features import engineer_all, engineer_behavioral_features
from load_custom_dataset import load_custom
from load_dataset2 import load_dataset2
from validation import audit_dataset1, audit_dataset2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def base_rf():
    return RandomForestClassifier(**RF_PARAMS)


def clean_feature_frame(frame):
    # Do not impute here: SimpleImputer in the CV pipeline learns medians only
    # from the relevant training fold.  Replacing infinities is non-learned.
    return frame.replace([np.inf, -np.inf], np.nan)


def evaluate_set(name, X, y, models):
    records, predictions, folds = [], {}, []
    for label, estimator, scale in models:
        result = nested_cv_evaluate(
            X, y, estimator, label, scale=scale, seed=SEED,
            outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
            threshold_grid=THRESHOLD_GRID,
        )
        record = {"experiment": name, **result.summary, "n_features": int(X.shape[1])}
        records.append(record)
        predictions[label] = result.predictions
        fold_frame = result.fold_metrics.copy()
        fold_frame.insert(0, "experiment", name)
        fold_frame.insert(1, "model", label)
        folds.append(fold_frame)
    return records, predictions, pd.concat(folds, ignore_index=True)


def save_plots(predictions, y, stem):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, pred in predictions.items():
        RocCurveDisplay.from_predictions(y, pred.probability_bot, name=label, ax=ax)
    ax.set_title("Nested-CV out-of-fold ROC curves")
    fig.tight_layout(); fig.savefig(OFFICIAL_RESULTS / f"{stem}_roc.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, pred in predictions.items():
        PrecisionRecallDisplay.from_predictions(y, pred.probability_bot, name=label, ax=ax)
    ax.set_title("Nested-CV out-of-fold precision-recall curves")
    fig.tight_layout(); fig.savefig(OFFICIAL_RESULTS / f"{stem}_pr.png", dpi=160); plt.close(fig)


def run_d1():
    raw = load_custom(str(DATASET1_PATH))
    audit = audit_dataset1(raw)
    df = raw.dropna(subset=["bot"]).copy()
    df["bot"] = df.bot.astype(int)
    base = engineer_behavioral_features(df, ref_date=datetime(2025, 1, 1, tzinfo=timezone.utc))
    extended = base.join(engineer_extra_features(df))
    _, _, content_cols = engineer_all(df)
    content = engineer_all(df)[0][content_cols]
    models = [
        ("RF behavioral", base_rf(), False),
        ("LogReg behavioral", LogisticRegression(max_iter=2000, random_state=SEED), True),
    ]
    records, preds, folds = evaluate_set("D1-full-base", clean_feature_frame(base), df.bot.values, models)
    ext_records, ext_preds, ext_folds = evaluate_set(
        "D1-full-extended", clean_feature_frame(extended), df.bot.values,
        [("RF extended behavioral", base_rf(), False)],
    )
    records += ext_records; preds.update(ext_preds); folds = pd.concat([folds, ext_folds], ignore_index=True)

    # Fair behavioural/content/fusion comparison: identical text-bearing users.
    text_mask = df.status.fillna("").astype(str).str.len().gt(0)
    matched = df.loc[text_mask].reset_index(drop=True)
    matched_extended = extended.loc[text_mask].reset_index(drop=True)
    matched_content = content.loc[text_mask].reset_index(drop=True)
    fusion = matched_extended.join(matched_content)
    matched_records, matched_preds, matched_folds = [], {}, []
    for label, features in [
        ("RF behavioral (matched text subset)", matched_extended),
        ("RF content (matched text subset)", matched_content),
        ("RF fusion (matched text subset)", fusion),
    ]:
        r, p, f = evaluate_set("D1-matched-text", clean_feature_frame(features), matched.bot.values,
                               [(label, base_rf(), False)])
        matched_records += r; matched_preds.update(p); matched_folds.append(f)
    records += matched_records; folds = pd.concat([folds, *matched_folds], ignore_index=True)

    pd.DataFrame(records).to_csv(OFFICIAL_RESULTS / "d1_metrics.csv", index=False)
    folds.to_csv(OFFICIAL_RESULTS / "d1_fold_metrics.csv", index=False)
    for label, p in {**preds, **matched_preds}.items():
        p.to_csv(OFFICIAL_RESULTS / f"d1_predictions_{label.replace(' ', '_').replace('(', '').replace(')', '')}.csv", index=False)
    save_plots(matched_preds, matched.bot.values, "d1_matched_text")
    return audit, records


def run_d2():
    raw, _ = load_dataset2(str(DATASET2_PATH))
    audit = audit_dataset2(raw)
    df = raw.dropna(subset=["bot"]).copy()
    df["bot"] = df.bot.astype(int)
    base = engineer_behavioral_features(df, ref_date=datetime(2026, 1, 1, tzinfo=timezone.utc))
    extended = base.join(engineer_extra_features_v2(df))
    records, preds, folds = evaluate_set(
        "D2-full-base", clean_feature_frame(base), df.bot.values,
        [("RF behavioral", base_rf(), False)],
    )
    r, p, f = evaluate_set(
        "D2-full-extended", clean_feature_frame(extended), df.bot.values,
        [("RF extended behavioral", base_rf(), False)],
    )
    records += r; preds.update(p); folds = pd.concat([folds, f], ignore_index=True)
    # Botometer is a diagnostic only; it is not used to construct labels.
    bm = df.botometer_score.notna()
    if bm.sum() and df.loc[bm, "bot"].nunique() == 2:
        from sklearn.metrics import roc_auc_score
        records.append({"experiment": "D2-Botometer-diagnostic", "model": "Botometer raw score",
                        "n": int(bm.sum()), "n_features": 1,
                        "roc_auc_oof": float(roc_auc_score(df.loc[bm, "bot"], df.loc[bm, "botometer_score"])),
                        "evaluation": "diagnostic score; no model fitting"})
    pd.DataFrame(records).to_csv(OFFICIAL_RESULTS / "d2_metrics.csv", index=False)
    folds.to_csv(OFFICIAL_RESULTS / "d2_fold_metrics.csv", index=False)
    for label, frame in preds.items():
        frame.to_csv(OFFICIAL_RESULTS / f"d2_predictions_{label.replace(' ', '_')}.csv", index=False)
    save_plots(preds, df.bot.values, "d2_behavioral")
    return audit, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("d1", "d2", "both"), default="both")
    args = parser.parse_args()
    OFFICIAL_RESULTS.mkdir(exist_ok=True)
    audits, all_records = {}, []
    if args.dataset in ("d1", "both"):
        audits["dataset1"] , records = run_d1(); all_records += records
    if args.dataset in ("d2", "both"):
        audits["dataset2"], records = run_d2(); all_records += records
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "outer_splits": OUTER_SPLITS, "inner_splits": INNER_SPLITS,
        "threshold_grid": list(THRESHOLD_GRID), "rf_params": RF_PARAMS,
        "python": sys.version, "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "dataset_sha256": {"d1": sha256(DATASET1_PATH), "d2": sha256(DATASET2_PATH)},
        "audits": audits,
        "legacy_results_note": "results_custom and results_dataset2 predate fold-local preprocessing; retain as historical outputs, not official corrected results.",
    }
    (OFFICIAL_RESULTS / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(all_records).to_csv(OFFICIAL_RESULTS / "all_metrics.csv", index=False)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Official outputs: {OFFICIAL_RESULTS}")


if __name__ == "__main__":
    main()
