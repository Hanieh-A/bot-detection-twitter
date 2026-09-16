"""Leakage-safe evaluation utilities shared by official experiments.

All learnt preprocessing lives in an sklearn Pipeline.  The outer test fold is
never used for imputation, scaling, threshold selection, or model fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class EvaluationResult:
    summary: dict
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame


def make_pipeline(estimator, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(estimator)))
    return Pipeline(steps)


def _best_threshold(y, probabilities, grid):
    scores = [f1_score(y, probabilities >= threshold, zero_division=0)
              for threshold in grid]
    return float(grid[int(np.argmax(scores))])


def _inner_threshold(X, y, estimator, scale, seed, n_splits, grid):
    """Choose a threshold using only the current outer-training partition."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for train, valid in splitter.split(X, y):
        pipeline = make_pipeline(estimator, scale)
        pipeline.fit(X.iloc[train], y[train])
        oof[valid] = pipeline.predict_proba(X.iloc[valid])[:, 1]
    return _best_threshold(y, oof, grid)


def nested_cv_evaluate(X: pd.DataFrame, y, estimator, name: str, *, scale=False,
                       seed=42, outer_splits=5, inner_splits=4,
                       threshold_grid=None) -> EvaluationResult:
    """Nested-CV estimate with fold-local preprocessing and threshold tuning.

    ROC-AUC/AP use outer-fold probabilities. Threshold-dependent measures use a
    threshold selected from inner-CV predictions of each outer training fold.
    This makes the reported F1/precision/recall honest estimates, rather than
    reusing the same OOF predictions to choose and score a threshold.
    """
    if threshold_grid is None:
        threshold_grid = tuple(round(x / 100, 2) for x in range(5, 96, 5))
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Binary labels with both classes are required.")
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    probabilities = np.zeros(len(y), dtype=float)
    predicted = np.zeros(len(y), dtype=int)
    thresholds = np.zeros(len(y), dtype=float)
    rows = []
    for fold, (train, test) in enumerate(outer.split(X, y), start=1):
        threshold = _inner_threshold(
            X.iloc[train].reset_index(drop=True), y[train], estimator, scale,
            seed + fold, inner_splits, threshold_grid,
        )
        pipeline = make_pipeline(estimator, scale)
        pipeline.fit(X.iloc[train], y[train])
        p = pipeline.predict_proba(X.iloc[test])[:, 1]
        pred = (p >= threshold).astype(int)
        probabilities[test], predicted[test], thresholds[test] = p, pred, threshold
        rows.append({
            "fold": fold, "n_test": len(test), "threshold_from_inner_cv": threshold,
            "roc_auc": roc_auc_score(y[test], p),
            "average_precision": average_precision_score(y[test], p),
            "accuracy": accuracy_score(y[test], pred),
            "f1": f1_score(y[test], pred, zero_division=0),
            "precision": precision_score(y[test], pred, zero_division=0),
            "recall": recall_score(y[test], pred, zero_division=0),
        })
    prediction_frame = pd.DataFrame({
        "row": np.arange(len(y)), "y_true": y, "probability_bot": probabilities,
        "prediction": predicted, "threshold_from_inner_cv": thresholds,
    })
    fold_metrics = pd.DataFrame(rows)
    summary = {
        "model": name, "n": int(len(y)), "positive_rate": float(y.mean()),
        "roc_auc_oof": float(roc_auc_score(y, probabilities)),
        "average_precision_oof": float(average_precision_score(y, probabilities)),
        "accuracy_outer_mean": float(fold_metrics.accuracy.mean()),
        "f1_outer_mean": float(fold_metrics.f1.mean()),
        "precision_outer_mean": float(fold_metrics.precision.mean()),
        "recall_outer_mean": float(fold_metrics.recall.mean()),
        "roc_auc_outer_mean": float(fold_metrics.roc_auc.mean()),
        "roc_auc_outer_std": float(fold_metrics.roc_auc.std(ddof=1)),
        "selected_threshold_median": float(np.median(thresholds)),
        "evaluation": "nested stratified CV; preprocessing fit within each fold",
    }
    return EvaluationResult(summary, prediction_frame, fold_metrics)
