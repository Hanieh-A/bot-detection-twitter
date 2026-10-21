"""Leakage-safe, account-level TF-IDF + Logistic Regression bot detection baseline.

Only biography and tweet text enter the model. IDs/handles are used solely for joins
and reporting, never as model features. See README.md for dataset/label caveats.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

SEED = 42


def norm_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def find_column(df: pd.DataFrame, requested: str | None, candidates: Iterable[str], role: str) -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"{role} column {requested!r} not found. Available columns: {list(df.columns)}")
        return requested
    lower = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def read_sheet(path: str, sheet: str | int | None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workbook not found: {p.resolve()}")
    return pd.read_excel(p, sheet_name=sheet if sheet is not None else 0)


def canonical_label(value: object, positive_label: str, negative_label: str) -> float:
    """Return 1 for bot, 0 for human, NaN for ambiguous/unrecognized labels."""
    s = norm_text(value).casefold().strip()
    # Handles values like bot(1), human(2), Bot, Human, and numeric 1/2.
    bot_values = {positive_label.casefold(), "bot", "bot(1)", "1", "true"}
    human_values = {negative_label.casefold(), "human",
                    "human(2)", "2", "false"}
    if s in bot_values or re.search(r"\bbot\s*\(?\s*1\s*\)?", s):
        return 1.0
    if s in human_values or re.search(r"\bhuman\s*\(?\s*2\s*\)?", s):
        return 0.0
    return np.nan


def normalize_id_for_join(value: object) -> str | None:
    """Normalize a numeric account ID, ignoring its last five digits as requested.

    Use this only for account-ID columns, never for tweet IDs or screen names.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    # Excel may turn an integer-looking ID into e.g. "123456789.0".
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    if not s.isdigit() or len(s) <= 5:
        return None
    return s[:-5]


def normalize_handle_for_join(value: object) -> str | None:
    """Normalize screen names without changing their identity."""
    if pd.isna(value):
        return None
    s = str(value).strip().lstrip("@").casefold()
    if not s or s in {"nan", "none", "null", "<na>"}:
        return None
    return s


def make_user_documents(
    users: pd.DataFrame,
    tweets: pd.DataFrame | None,
    *,
    user_id_col: str | None,
    tweet_user_id_col: str | None,
    bio_col: str | None,
    tweet_text_col: str | None,
    timestamp_col: str | None,
    max_tweets: int,
) -> tuple[pd.DataFrame, dict]:
    """Create one document/account and automatically choose the best matching join keys."""
    users = users.copy()
    uid = find_column(users, user_id_col, [
                      "user_id", "id", "original.user_id"], "user ID")
    user_handle = find_column(
        users, None, ["screen_name", "username", "handle"], "user handle")
    if uid is None and user_handle is None:
        raise ValueError(
            "Could not find a user join key. Pass --user-id-column or ensure screen_name exists.")

    bio = find_column(
        users, bio_col, ["clean_description", "description", "bio"], "biography")
    if bio is None:
        warnings.warn(
            "No biography column found; documents will use tweet text only.")
    users["__bio_text"] = users[bio].map(norm_text) if bio else ""

    tweet_map: dict[str, str] = {}
    join_col = uid or user_handle
    tid = None
    join_type = "account_id" if uid else "screen_name"
    matched_tweet_rows = 0
    matching_unique_keys = 0

    if tweets is not None and not tweets.empty:
        t = tweets.copy()
        text_col = find_column(t, tweet_text_col, [
                               "text", "tweet_text", "full_text"], "tweet text")
        if text_col is None:
            raise ValueError(
                f"No tweet text column found. Available columns: {list(t.columns)}")

        # Build candidates for both sides. In some exports, `id` in the tweet
        # sheet is the tweet ID, not its author's ID. Compare actual overlap and
        # fall back to screen_name when the requested ID columns do not match.
        user_candidates: list[tuple[str, str]] = []
        for col in [uid, user_handle, find_column(users, None, ["user_id", "original.user_id", "id"], "user ID")]:
            if col and col in users.columns and all(col != existing[0] for existing in user_candidates):
                kind = "screen_name" if col == user_handle or str(col).casefold(
                ) in {"screen_name", "username", "handle"} else "account_id"
                user_candidates.append((col, kind))

        tweet_candidates: list[tuple[str, str]] = []
        requested_tid = find_column(t, tweet_user_id_col, [
                                    "user_id", "original.user_id", "user.id", "screen_name", "username"], "tweet user ID")
        for col in [requested_tid, find_column(t, None, ["user_id", "original.user_id", "user.id"], "tweet author ID"),
                    find_column(t, None, ["screen_name", "username", "handle"], "tweet author handle")]:
            if col and col in t.columns and all(col != existing[0] for existing in tweet_candidates):
                kind = "screen_name" if str(col).casefold() in {
                    "screen_name", "username", "handle"} else "account_id"
                tweet_candidates.append((col, kind))

        if not user_candidates or not tweet_candidates:
            raise ValueError(
                "Could not identify compatible user/tweet join columns. Check the workbook columns.")

        # Select the pair that matches the largest share of nonempty tweet rows.
        # Account IDs are normalized by dropping their last five digits; handles
        # are matched case-insensitively with a leading @ removed.
        best_pair = None
        for ucol, ukind in user_candidates:
            norm_u = normalize_handle_for_join if ukind == "screen_name" else normalize_id_for_join
            user_keys = pd.Series(users[ucol].map(norm_u)).dropna()
            user_key_set = set(user_keys.astype(str))
            if not user_key_set:
                continue
            for tcol, tkind in tweet_candidates:
                if ukind != tkind:
                    continue
                norm_t = normalize_handle_for_join if tkind == "screen_name" else normalize_id_for_join
                tweet_keys = t[tcol].map(norm_t)
                valid = tweet_keys.notna() & t[text_col].map(norm_text).ne("")
                valid_keys = tweet_keys[valid].astype(str)
                if valid_keys.empty:
                    continue
                matched_mask = valid_keys.isin(user_key_set)
                matched_rows = int(matched_mask.sum())
                match_rate = matched_rows / len(valid_keys)
                unique_matches = len(set(valid_keys[matched_mask]))
                score = (match_rate, matched_rows, unique_matches)
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, ucol, ukind, tcol, tkind,
                                 matched_rows, unique_matches)

        if best_pair is None or best_pair[5] == 0:
            raise ValueError(
                "No tweet rows matched any user join key. Verify the author identifier columns; "
                "the `id` column in a tweet sheet is often the tweet ID. "
                f"User columns: {list(users.columns)}; tweet columns: {list(t.columns)}"
            )

        _, join_col, join_type, tid, _, matched_tweet_rows, matching_unique_keys = best_pair
        norm_key = normalize_handle_for_join if join_type == "screen_name" else normalize_id_for_join
        users["__join_key"] = users[join_col].map(norm_key)
        t["__join_key"] = t[tid].map(norm_key)
        t["__text"] = t[text_col].map(norm_text)
        t = t[t["__text"].ne("") & t["__join_key"].notna()]
        t = t.drop_duplicates(subset=["__join_key", "__text"])
        ts = find_column(t, timestamp_col, [
                         "published_at", "created_at", "timestamp", "received_at"], "tweet timestamp")
        if ts:
            t["__sort_time"] = pd.to_datetime(t[ts], errors="coerce", utc=True)
            t = t.sort_values(["__join_key", "__sort_time", "__text"], ascending=[
                              True, False, True], na_position="last", kind="mergesort")
        else:
            t = t.sort_values(["__join_key", "__text"], kind="mergesort")
        for key, group in t.groupby("__join_key", sort=False):
            chosen = group["__text"].head(max_tweets).tolist()
            if chosen:
                tweet_map[str(key)] = " ".join(chosen)

    else:
        users["__join_key"] = users[join_col].astype("string").str.strip()
        ts = None

    users["__tweet_text"] = users["__join_key"].map(tweet_map).fillna("")
    users["text_document"] = [
        (f"[BIO] {b} " if b else "") + (f"[TWEETS] {tw}" if tw else "")
        for b, tw in zip(users["__bio_text"], users["__tweet_text"])
    ]
    users["text_document"] = users["text_document"].map(norm_text)
    summary = {
        "n_user_rows": int(len(users)),
        "n_nonempty_documents": int(users["text_document"].ne("").sum()),
        "n_bio_only": int((users["__bio_text"].ne("") & users["__tweet_text"].eq("")).sum()),
        "n_tweet_text": int(users["__tweet_text"].ne("").sum()),
        "n_bio_and_tweets": int((users["__bio_text"].ne("") & users["__tweet_text"].ne("")).sum()),
        "n_empty_documents": int(users["text_document"].eq("").sum()),
        "join_key_used": str(join_col),
        "tweet_join_key_used": str(tid) if tweets is not None and not tweets.empty else None,
        "join_key_type": join_type,
        "matched_tweet_rows_before_deduplication": int(matched_tweet_rows),
        "matching_unique_join_keys": int(matching_unique_keys),
        "bio_column_used": str(bio) if bio else None,
        "max_tweets_per_user": int(max_tweets),
        "tweet_selection": f"most recent by {ts}" if tweets is not None and not tweets.empty and ts else "deterministic lexical order (timestamp unavailable)",
    }
    return users, summary


def make_pipeline(level: str, min_df: int, max_features: int, c: float, class_weight: str | None, max_iter: int) -> Pipeline:
    if level == "word":
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df,
                                     max_features=max_features, norm="l2", sublinear_tf=True,
                                     token_pattern=r"(?u)\b\w+\b")
    else:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df,
                                     max_features=max_features, norm="l2", sublinear_tf=True)
    return Pipeline([("tfidf", vectorizer), ("clf", LogisticRegression(
        C=c, class_weight=class_weight, max_iter=max_iter, random_state=SEED,
        solver="liblinear"))])


def threshold_from_inner_oof(y: np.ndarray, scores: np.ndarray) -> float:
    """Choose threshold maximizing bot F1 using inner-training OOF predictions only."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / \
        np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == np.nanmax(f1))
    return float(thresholds[int(best[0])]) if len(best) else 0.5


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    out = {
        "n_users": int(len(y)), "positive_rate": float(np.mean(y)),
        "precision_bot": float(precision_score(y, pred, zero_division=0)),
        "recall_bot": float(recall_score(y, pred, zero_division=0)),
        "f1_bot": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(accuracy_score(y, pred)),
        "threshold": float(threshold),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
    }
    return out


def nested_oof(X: list[str], y: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    counts = np.bincount(y, minlength=2)
    outer_n = min(args.outer_splits, int(counts.min()))
    if outer_n < 2:
        raise ValueError(
            f"Not enough examples in each class for stratified CV: bot={counts[1]}, human={counts[0]}.")
    if outer_n != args.outer_splits:
        warnings.warn(
            f"Reducing outer folds from {args.outer_splits} to {outer_n} due to class counts.")
    outer = StratifiedKFold(n_splits=outer_n, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fold_rows: list[dict] = []
    configs = [(c, cw) for c in args.c_values for cw in [None, "balanced"]]
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), y), 1):
        y_train = y[train_idx]
        inner_counts = np.bincount(y_train, minlength=2)
        inner_n = min(args.inner_splits, int(inner_counts.min()))
        if inner_n < 2:
            raise ValueError(
                f"Outer fold {fold} has too few minority-class training records for inner CV.")
        inner = StratifiedKFold(
            n_splits=inner_n, shuffle=True, random_state=SEED + fold)
        best = None
        # Tune C and class weighting on inner OOF AP; all vectorizers fit only inside folds.
        for c, cw in configs:
            pipe = make_pipeline(args.level, args.min_df,
                                 args.max_features, c, cw, args.max_iter)
            inner_prob = cross_val_predict(pipe, [X[i] for i in train_idx], y_train,
                                           cv=inner, method="predict_proba", n_jobs=args.n_jobs)[:, 1]
            ap = average_precision_score(y_train, inner_prob)
            threshold = threshold_from_inner_oof(y_train, inner_prob)
            candidate = (ap, c, cw, threshold)
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        _, best_c, best_cw, threshold = best
        final = make_pipeline(args.level, args.min_df,
                              args.max_features, best_c, best_cw, args.max_iter)
        final.fit([X[i] for i in train_idx], y_train)
        prob = final.predict_proba([X[i] for i in test_idx])[:, 1]
        oof[test_idx] = prob
        row = {"fold": fold, "train_n": len(train_idx), "test_n": len(test_idx),
               "vocabulary_size": int(len(final.named_steps["tfidf"].vocabulary_)),
               "n_features": int(len(final.named_steps["tfidf"].vocabulary_)),
               "selected_C": best_c, "selected_class_weight": best_cw or "none",
               "selected_threshold": threshold, "inner_oof_average_precision": float(best[0]),
               **metrics(y[test_idx], prob, threshold)}
        fold_rows.append(row)
        print(
            f"[outer fold {fold}/{outer_n}] n={len(test_idx)} AP={row['average_precision']:.4f} F1={row['f1_bot']:.4f}")
    if np.isnan(oof).any():
        raise RuntimeError("Some users did not receive OOF predictions.")
    return oof, np.array([r["selected_threshold"] for r in fold_rows]), fold_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["d1", "d2"], required=True)
    p.add_argument("--user-file", required=True,
                   help="Excel workbook containing one row per user")
    p.add_argument("--user-sheet", default=None)
    p.add_argument("--tweet-file", default=None,
                   help="Workbook containing tweet text; may be same as user-file")
    p.add_argument(
        "--tweet-sheet", nargs="+", default=None,
        help="One or more tweet sheet names in the workbook (e.g. --tweet-sheet tweetMetaData_1 tweetMetaData_2). All listed sheets are concatenated before matching."
    )
    p.add_argument("--label-column", default=None,
                   help="D1 defaults to final-tag. D2 must be specified explicitly.")
    p.add_argument("--positive-label", default="bot")
    p.add_argument("--negative-label", default="human")
    p.add_argument("--user-id-column", default=None)
    p.add_argument("--tweet-user-id-column", default=None)
    p.add_argument("--bio-column", default=None)
    p.add_argument("--tweet-text-column", default=None)
    p.add_argument("--tweet-timestamp-column", default=None)
    p.add_argument("--max-tweets", type=int, default=20)
    p.add_argument("--level", choices=["word", "char"], default="word")
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--c-values", type=float,
                   nargs="+", default=[0.1, 1.0, 10.0])
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=4)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "d2" and not args.label_column:
        raise SystemExit(
            "Dataset 2 requires --label-column. Verify label provenance first; do not blindly use Predicted_Label if it came from the behavioral model.")
    label_col = args.label_column or "final-tag"
    users = read_sheet(args.user_file, args.user_sheet)
    if label_col not in users.columns:
        raise SystemExit(
            f"Label column {label_col!r} not found. Available columns: {list(users.columns)}")
    tweets = None
    if args.tweet_file:
        tweet_sheets = args.tweet_sheet or [None]
        tweet_frames = []
        for sheet_name in tweet_sheets:
            sheet_frame = read_sheet(args.tweet_file, sheet_name)
            # Keep a trace of the source sheet for diagnostics; it is not a model feature.
            sheet_frame["__source_tweet_sheet"] = str(
                sheet_name if sheet_name is not None else "default")
            tweet_frames.append(sheet_frame)
        tweets = pd.concat(tweet_frames, ignore_index=True, sort=False)
        print(
            "Tweet sheets loaded: "
            + json.dumps(
                {str(name if name is not None else "default"): int(len(frame_))
                 for name, frame_ in zip(tweet_sheets, tweet_frames)},
                ensure_ascii=False,
            )
        )
    frame, coverage = make_user_documents(users, tweets, user_id_col=args.user_id_column,
                                          tweet_user_id_col=args.tweet_user_id_column, bio_col=args.bio_column,
                                          tweet_text_col=args.tweet_text_column, timestamp_col=args.tweet_timestamp_column,
                                          max_tweets=args.max_tweets)
    frame["__y"] = frame[label_col].map(lambda v: canonical_label(
        v, args.positive_label, args.negative_label))
    # Exclude ambiguous labels and empty text. Never silently convert uncertain/News Agent to human.
    eligible = frame[frame["__y"].notna(
    ) & frame["text_document"].ne("")].copy()
    eligible["__y"] = eligible["__y"].astype(int)
    if eligible.empty or eligible["__y"].nunique() != 2:
        raise SystemExit(
            f"After filtering, need both bot and human labels. Counts: {eligible['__y'].value_counts().to_dict()}")
    y = eligible["__y"].to_numpy()
    X = eligible["text_document"].tolist()
    outdir = Path(
        args.output_dir or f"results_text_tfidf_{args.level}_{args.dataset}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset={args.dataset}; label={label_col}; eligible={len(eligible)}; bots={int(y.sum())}; humans={int((y == 0).sum())}")
    print(f"Text coverage: {json.dumps(coverage, ensure_ascii=False)}")
    oof, fold_thresholds, folds = nested_oof(X, y, args)
    # Each outer fold has its own threshold; apply each threshold only to that fold's OOF predictions.
    thresholds_by_row = np.empty(len(y), dtype=float)
    outer = StratifiedKFold(n_splits=len(
        folds), shuffle=True, random_state=SEED)
    # Recreate exact splits to align fold thresholds with each OOF test index.
    for threshold, (_, test_idx) in zip(fold_thresholds, outer.split(np.zeros(len(y)), y)):
        thresholds_by_row[test_idx] = threshold
    pred = (oof >= thresholds_by_row).astype(int)
    aggregate = metrics(y, oof, 0.5)
    aggregate["threshold_note"] = "Ranking metrics use OOF probabilities; precision/recall/F1/accuracy use each outer fold's threshold selected from inner OOF predictions."
    aggregate["threshold_median"] = float(np.median(fold_thresholds))
    aggregate["n_bot"] = int(y.sum())
    aggregate["n_human"] = int((y == 0).sum())
    aggregate["dataset"] = args.dataset
    aggregate["experiment"] = f"text_tfidf_{args.level}_{args.dataset}"
    aggregate["label_column"] = label_col
    # Use fold-specific thresholds for classification metrics.
    aggregate.update({"precision_bot_nested_threshold": float(precision_score(y, pred, zero_division=0)),
                      "recall_bot_nested_threshold": float(recall_score(y, pred, zero_division=0)),
                      "f1_bot_nested_threshold": float(f1_score(y, pred, zero_division=0)),
                      "accuracy_nested_threshold": float(accuracy_score(y, pred)),
                      "confusion_matrix_nested_threshold_tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()})
    # Reporting identifier only; do not export unnecessary profile information.
    id_col = find_column(eligible, args.user_id_column, [
                         "user_id", "id", "original.user_id"], "user ID")
    if id_col is None:
        id_col = find_column(
            eligible, None, ["screen_name", "username", "handle"], "handle")
    predictions = pd.DataFrame({"user_id": eligible[id_col].astype(str).to_numpy() if id_col else eligible.index.astype(str),
                                "true_label_bot_1_human_0": y, "oof_bot_probability": oof,
                                "selected_threshold": thresholds_by_row, "predicted_label_bot_1_human_0": pred})
    predictions.to_csv(outdir / "oof_predictions.csv", index=False)
    pd.DataFrame(folds).to_csv(outdir / "fold_metrics.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(aggregate,
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "dataset_summary.json").write_text(json.dumps({**coverage,
                                                             "label_column": label_col, "eligible_users": int(len(eligible)), "excluded_ambiguous_or_unlabeled": int(frame["__y"].isna().sum()),
                                                             "excluded_empty_text": int((frame["__y"].notna() & frame["text_document"].eq("")).sum()),
                                                             "bot_count": int(y.sum()), "human_count": int((y == 0).sum()),
                                                             "positive_class": "bot", "seed": SEED, "outer_splits_actual": len(folds), "inner_splits_requested": args.inner_splits,
                                                             "level": args.level, "max_features": args.max_features, "min_df": args.min_df,
                                                             "C_grid": args.c_values, "max_iter": args.max_iter}, indent=2, ensure_ascii=False), encoding="utf-8")
    # Curves from OOF probabilities, independent of classification threshold.
    fpr, tpr, _ = roc_curve(y, oof)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        outdir / "roc_curve.csv", index=False)
    precision, recall, _ = precision_recall_curve(y, oof)
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
        outdir / "pr_curve.csv", index=False)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
