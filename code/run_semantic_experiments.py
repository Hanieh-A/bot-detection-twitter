"""Additive semantic-embedding experiments for the Persian Twitter bot project.

Run from the repository root:
    python code/run_semantic_experiments.py --dataset d1

This script writes only under results_semantic/ and leaves existing scripts/results
untouched. It reuses the project's existing dataset loader, feature engineering,
configuration, and leakage-safe nested-CV nested_cv utilities.
"""
from __future__ import annotations
from data_audit import audit_dataset1
from load_dataset1 import load_custom_v2 as load_custom
from semantic_features import DEFAULT_MODEL_NAME, engineer_semantic_features
from features import engineer_behavioral_features, engineer_content_features
from extra_features_d1 import engineer_extra_features
from nested_cv import nested_cv_evaluate
from experiment_config import (
    DATASET1_PATH, INNER_SPLITS, OUTER_SPLITS, RF_PARAMS, SEED, THRESHOLD_GRID
)
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")


ROOT_RESULTS = Path(__file__).resolve().parent.parent / "results_semantic"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def base_rf():
    return RandomForestClassifier(**RF_PARAMS)


def clean_feature_frame(frame):
    return frame.replace([np.inf, -np.inf], np.nan)


def evaluate_one(experiment, model_name, X, y):
    result = nested_cv_evaluate(
        clean_feature_frame(X),
        y,
        base_rf(),
        model_name,
        scale=False,
        seed=SEED,
        outer_splits=OUTER_SPLITS,
        inner_splits=INNER_SPLITS,
        threshold_grid=THRESHOLD_GRID,
    )
    summary = {
        "experiment": experiment,
        **result.summary,
        "n_features": int(X.shape[1]),
    }
    folds = result.fold_metrics.copy()
    folds.insert(0, "experiment", experiment)
    folds.insert(1, "model", model_name)
    return summary, result.predictions, folds


def save_plots(predictions, y, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, pred in predictions.items():
        RocCurveDisplay.from_predictions(
            y, pred["probability_bot"], name=label, ax=ax)
    ax.set_title("Nested-CV out-of-fold ROC curves — semantic experiments")
    fig.tight_layout()
    fig.savefig(output_dir / "semantic_roc.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, pred in predictions.items():
        PrecisionRecallDisplay.from_predictions(
            y, pred["probability_bot"], name=label, ax=ax)
    ax.set_title(
        "Nested-CV out-of-fold precision-recall curves — semantic experiments")
    fig.tight_layout()
    fig.savefig(output_dir / "semantic_pr.png", dpi=160)
    plt.close(fig)


def run_d1():
    output_dir = ROOT_RESULTS
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    raw = load_custom(str(DATASET1_PATH))
    audit = audit_dataset1(raw)
    df = raw.dropna(subset=["bot"]).copy()
    df["bot"] = df["bot"].astype(int)

    # Compare all models on exactly the same accounts with a non-empty sampled tweet.
    text_mask = df["status"].fillna("").astype(str).str.strip().ne("")
    matched = df.loc[text_mask].copy()
    if matched.empty:
        raise ValueError(
            "No labeled rows with a non-empty 'status' column were found. "
            "Check the dataset loader's output columns."
        )

    # Preserve the reference date used by the existing official D1 runner.
    matched_behavioral = engineer_behavioral_features(
        matched, ref_date=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    matched_extended = matched_behavioral.join(
        engineer_extra_features(matched))
    matched_content = engineer_content_features(matched)
    matched_semantic = engineer_semantic_features(matched, cache_dir=cache_dir)

    feature_sets = [
        ("RF behavioral extended (matched text subset)", matched_extended),
        ("RF simple content (matched text subset)", matched_content),
        ("RF semantic MiniLM (matched text subset)", matched_semantic),
        (
            "RF behavioral plus semantic MiniLM (matched text subset)",
            matched_extended.join(matched_semantic),
        ),
    ]

    records, predictions, fold_frames = [], {}, []
    y = matched["bot"].to_numpy(dtype=int)
    for model_name, X in feature_sets:
        print(f"\nEvaluating {model_name}: n={len(X)}, features={X.shape[1]}")
        summary, pred, folds = evaluate_one(
            "D1-matched-text-semantic", model_name, X.reset_index(drop=True), y
        )
        records.append(summary)
        predictions[model_name] = pred
        # Add account index within this matched subset for traceability.
        pred = pred.copy()
        pred["matched_row"] = np.arange(len(pred))
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "plus")
        )
        pred.to_csv(output_dir / f"predictions_{safe_name}.csv", index=False)
        fold_frames.append(folds)

    metrics = pd.DataFrame(records)
    metrics.to_csv(output_dir / "semantic_metrics.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        output_dir / "semantic_fold_metrics.csv", index=False
    )
    save_plots(predictions, y, output_dir)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET1_PATH),
        "dataset_sha256": sha256(DATASET1_PATH),
        "semantic_model": DEFAULT_MODEL_NAME,
        "semantic_dimension": 384,
        "n_labeled_rows": int(len(df)),
        "n_matched_text_rows": int(len(matched)),
        "seed": SEED,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "threshold_grid": list(THRESHOLD_GRID),
        "rf_params": RF_PARAMS,
        "audit": audit,
        "notes": [
            "This additive runner does not modify the existing official runner or nested_cv.py.",
            "All models are evaluated on the same matched subset.",
            "The pretrained semantic encoder is frozen; only the classifier is evaluated with nested CV.",
            "Embedding cache is keyed by model name and ordered text content.",
        ],
    }
    (output_dir / "semantic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\nSemantic experiment metrics:")
    print(metrics.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("d1",), default="d1",
                        help="This additive semantic runner currently supports D1.")
    parser.parse_args()
    run_d1()


if __name__ == "__main__":
    main()
