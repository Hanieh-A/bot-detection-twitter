"""Feature-level fusion (behavioral + content/TF-IDF) bot detection, Random Forest.

Combines, for every account:
  - Behavioral/account-level numeric features (already computed, stored as
    columns in the behavioral Excel workbook).
  - Content features: a TF-IDF vector over the account's biography + up to
    N recent tweets, built with EXACTLY the same join/cleaning logic as
    run_text_tfidf2.py (imported from that file, not reimplemented).

Fusion strategy: feature concatenation, not score/probability stacking.
sklearn's ColumnTransformer does the concatenation; in every outer CV fold,
the TF-IDF vectorizer and the behavioral-feature imputer are fit ONLY on
that fold's training users (then applied to the held-out fold), so there is
no leakage -- this mirrors the nested, leakage-safe design already used in
run_text_tfidf2.py for the content-only model.

This script does not run anything by itself other than what you invoke from
the command line; it does not call run_text_tfidf2.py's own CLI/main().

--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
Put this file in the same directory as run_text_tfidf2.py (code/ in the
repo), since it imports functions from it directly.

--------------------------------------------------------------------------
USAGE (same workbook holds behavioral columns + bio + tweet sheets)
--------------------------------------------------------------------------
python fusion_model.py --dataset d1 --behavioral-file ../datasets/dataset1.xlsx --user-file ../datasets/dataset1.xlsx --tweet-file ../datasets/dataset1.xlsx --tweet-sheet tweets_meta_data --output-dir ../results_fusion_d1

python fusion_model.py --dataset d2 --behavioral-file ../datasets/dataset2.xlsx --user-file ../datasets/dataset2.xlsx --tweet-file ../datasets/dataset2.xlsx --tweet-sheet tweetsMetaData_1 tweetsMetaData_2 --output-dir ../results_fusion_d2 --label-column Predicted_Label

For dataset 2 you must also pass --label-column if it isn't "final-tag" in
that workbook (same requirement as run_text_tfidf2.py).

Run `python fusion_model.py --help` for the full option list -- most of it
mirrors run_text_tfidf2.py's flags so the two runs stay comparable.

--------------------------------------------------------------------------
ASSUMPTIONS I MADE (please check these against your actual thesis setup)
--------------------------------------------------------------------------
1. Join key: id with the last 5 digits dropped, via normalize_id_for_join()
   -- imported straight from run_text_tfidf2.py, exactly as you asked.

2. Default behavioral feature list = BEHAVIORAL_FEATURE_WHITELIST below.
   I could not see your behavioral-model script, only the column list you
   pasted, so I built this whitelist by hand. Left OUT by default:
     - identifiers/handles/names/URLs/images/raw created_at (not features)
     - the label/tagger/probe/reason columns and "cluster_id" (looked
       derived from, or correlated with, the labeling process itself)
     - "Botometer Score", "LLM(ChatGPT)", "LLM(Gemini)", "Grok 4" -- these
       are OTHER systems' bot-probability outputs, not your own behavioral
       signal; mixing them in would make the behavioral vs. content vs.
       fusion comparison unfair. Toggle back in with --include-botometer /
       --include-llm-scores if that's actually what you want.
     - the free-text manual-review flag columns (hashtag frequency, post
       similarity, burstiness, etc.) -- these look like they may have fed
       into how `final-tag` was assigned by the human taggers, so including
       them risks leakage. Toggle back in with
       --include-manual-review-flags, but please double check first how
       final-tag was actually produced.
   The script prints the exact list of behavioral columns it used at
   startup -- check it, and add/drop columns with
   --extra-behavioral-features / --exclude-behavioral-features.

3. Rows with an empty text_document (no bio, no matched tweets) are KEPT
   by default -- their TF-IDF slice is just a zero vector, and the
   behavioral features can still carry signal for them. Pass
   --require-nonempty-text to drop them instead (what run_text_tfidf2.py
   does, since it has nothing else to go on for those users).

4. TF-IDF --max-features defaults to 3000, not the content-only script's
   50000. Random Forest on a very high-dimensional sparse matrix is far
   slower and more memory-hungry than Logistic Regression is, so I kept
   this conservative. Raise it (e.g. --max-features 50000) for an
   apples-to-apples vocabulary with the content-only run if you have the
   time/memory budget.

5. Model/evaluation: Random Forest, nested CV (outer=5 by default, same
   as --outer-splits in run_text_tfidf2.py), with a small inner grid search
   over n_estimators / max_depth / min_samples_leaf / class_weight,
   selected by inner out-of-fold average precision -- the same selection
   rule run_text_tfidf2.py uses for its C/class_weight grid. Outputs
   (oof_predictions.csv, fold_metrics.csv, metrics.json, metrics_summary.csv,
   roc/pr curves, figures) are written in the same format/filenames as the
   content-only script so the three paths (behavioral / content / fusion)
   stay directly comparable.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, precision_recall_curve, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from run_text_tfidf2 import (  # reused, not reimplemented -- see module docstring
    SEED,
    canonical_label,
    find_column,
    make_user_documents,
    metrics,
    normalize_id_for_join,
    read_sheet,
    save_report_figures,
    threshold_from_inner_oof,
)

# ---------------------------------------------------------------------------
# Behavioral feature selection -- see ASSUMPTIONS in the module docstring.
# ---------------------------------------------------------------------------
BOOLEAN_BEHAVIORAL_COLS = [
    "default_profile", "default_profile_image", "has_custom_timelines",
    "is_translator", "possibly_sensitive", "verified", "want_retweets",
]

BEHAVIORAL_FEATURE_WHITELIST = [
    # raw counts / profile
    "fast_followers_count", "favourites_count", "followers_count",
    "friends_count", "listed_count", "media_count",
    "normal_followers_count", "statuses_count",
    *BOOLEAN_BEHAVIORAL_COLS,
    # engineered account-level features
    "user_age", "follower_growth_rate", "friends_growth_rate",
    "listed_growth_rate", "followers_friend_ratio",
    "hashtag_in_description", "numbers_in_description",
    "description_length", "num_digits_in_name", "num_digits_in_username",
    "url_in_description",
    # tweeting-behavior features
    "status_count", "no_type_tweet", "no_type_retweet_with_comment",
    "no_type_reply", "mean_no_media_per_tweet", "mean_no_words",
    "no_languages", "mean_no_hashtags", "mean_favourites_per_tweet",
    "time_between_tweets", "tweet_frequency", "min_tweets_per_hour",
    "min_tweets_per_day", "max_tweets_per_hour", "max_tweets_per_day",
    "max_occurence_of_same_gap", "unique_mention_rate_per_tweet",
    "mean_user_mentions_per_tweet", "retweet_as_tweet_rate",
    "retweet_ratio", "no_retweet_tweets", "mean_retweets_per_tweet",
    "no_tweets_on_creation_day", "no_tweets_on_creation_week",
]

OPTIONAL_EXTERNAL_DETECTOR_COLS = [
    "Botometer Score", "LLM(ChatGPT)", "LLM(Gemini)", "Grok 4",
]

MANUAL_REVIEW_FLAG_COLS = [
    "فرکانس استفاده از یک هشتگ یکسان بالا است.",
    "شباهت زیادی بین متن پست های کاربر هست.",
    "فواصل زمانی منظم و یکسان بین ارسال پست ها",
    "انفجار فعالیت ( ارسال تعداد پست زیاد در بازه زمانی یک ساعت)",
    "انفجار فعالیت ( ارسال تعداد پست زیاد در بازه زمانی یک روز)",
    "اغلب پست ها ریتوییت هستند",
    "تعداد منشن بقیه کاربران زیاد است",
    "تعداد حساب هایی که بیو های مشابه این کاربر دارند زیاد است",
    "بیو حساب شامل اطلاعات شخصی است ( برای مثال اسم دانشگاه، مدرک تحصیلی ، "
    "نام محل کار و...) ( نیاز به بررسی دستی حساب ها یا استفاده از llm دارد)",
    "ارسال تعداد پست زیاد در روز ایجاد حساب",
    "ارسال تعداد پست زیاد در هفته ایجاد حساب",
    "نداشتن گفت و گو با بقیه کاربران ( گذاشتن ریپلای های متوالی به پست های یکدیگر)",
    "تصویر پروفایل واقعی باشد ( نیاز به بررسی دستی حساب ها دارد)",
]


def robust_normalize_id_for_join(value: object) -> str | None:
    """Like normalize_id_for_join(), but first expands IDs that pandas has
    read as float64 into a plain digit string before handing off to it.

    If an id column has even one missing cell, pandas upcasts the whole
    column to float64. Python's str()/repr() of a float switches to
    scientific notation at 17+ significant digits (e.g. 1.23e+18) -- and
    modern Twitter snowflake IDs are 18-19 digits, so nearly every id in a
    float64 column stringifies to something normalize_id_for_join() can't
    parse, and it returns None for almost everything. This wrapper
    reformats such floats as plain integers first (losing at most a couple
    of the least-significant digits to float64's ~15-17 digit precision --
    which normalize_id_for_join() discards anyway by dropping the last 5
    digits), then defers entirely to the original function.
    """
    if isinstance(value, float) and not pd.isna(value):
        try:
            value = format(int(round(value)), "d")
        except (ValueError, OverflowError):
            return None
    return normalize_id_for_join(value)


def to_bool_numeric(series: pd.Series) -> pd.Series:
    def conv(v: object) -> float:
        s = str(v).strip().lower()
        if s in {"true", "1", "yes"}:
            return 1.0
        if s in {"false", "0", "no"}:
            return 0.0
        return np.nan
    return series.map(conv)


def build_pipeline(
    behavioral_cols: list[str], level: str, min_df: int, max_features: int,
    n_estimators: int, max_depth: int | None, min_samples_leaf: int,
    class_weight: str | None, random_state: int,
) -> Pipeline:
    if level == "word":
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df,
                                     max_features=max_features, norm="l2", sublinear_tf=True,
                                     token_pattern=r"(?u)\b\w+\b")
    else:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df,
                                     max_features=max_features, norm="l2", sublinear_tf=True)
    behavioral_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median"))])
    # A single string column name makes ColumnTransformer pass a 1D array to
    # the transformer (required by TfidfVectorizer); a list of names passes
    # a 2D array (required by SimpleImputer).
    pre = ColumnTransformer([
        ("tfidf", vectorizer, "text_document"),
        ("behavioral", behavioral_pipeline, behavioral_cols),
    ])
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight=class_weight, random_state=random_state, n_jobs=-1,
    )
    return Pipeline([("features", pre), ("clf", clf)])


def nested_oof_fusion(
    X_df: pd.DataFrame, y: np.ndarray, behavioral_cols: list[str], args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
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
    configs = [
        (n_est, max_depth, min_leaf, cw)
        for n_est in args.n_estimators_grid
        for max_depth in args.max_depth_grid
        for min_leaf in args.min_samples_leaf_grid
        for cw in [None, "balanced"]
    ]
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), y), 1):
        y_train = y[train_idx]
        X_train_outer = X_df.iloc[train_idx].reset_index(drop=True)
        inner_counts = np.bincount(y_train, minlength=2)
        inner_n = min(args.inner_splits, int(inner_counts.min()))
        if inner_n < 2:
            raise ValueError(
                f"Outer fold {fold} has too few minority-class training records for inner CV.")
        inner = StratifiedKFold(
            n_splits=inner_n, shuffle=True, random_state=SEED + fold)
        best = None
        for n_est, max_depth, min_leaf, cw in configs:
            pipe = build_pipeline(behavioral_cols, args.level, args.min_df, args.max_features,
                                  n_estimators=n_est, max_depth=max_depth, min_samples_leaf=min_leaf,
                                  class_weight=cw, random_state=SEED)
            inner_prob = cross_val_predict(pipe, X_train_outer, y_train, cv=inner,
                                           method="predict_proba", n_jobs=args.n_jobs)[:, 1]
            ap = average_precision_score(y_train, inner_prob)
            threshold = threshold_from_inner_oof(y_train, inner_prob)
            candidate = (ap, n_est, max_depth, min_leaf, cw, threshold)
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        _, best_n_est, best_depth, best_leaf, best_cw, threshold = best
        final = build_pipeline(behavioral_cols, args.level, args.min_df, args.max_features,
                               n_estimators=best_n_est, max_depth=best_depth, min_samples_leaf=best_leaf,
                               class_weight=best_cw, random_state=SEED)
        final.fit(X_train_outer, y_train)
        X_test_outer = X_df.iloc[test_idx].reset_index(drop=True)
        prob = final.predict_proba(X_test_outer)[:, 1]
        oof[test_idx] = prob
        vocab_size = len(final.named_steps["features"].named_transformers_[
                         "tfidf"].vocabulary_)
        row = {
            "fold": fold, "train_n": len(train_idx), "test_n": len(test_idx),
            "tfidf_vocabulary_size": int(vocab_size), "n_behavioral_features": len(behavioral_cols),
            "selected_n_estimators": best_n_est, "selected_max_depth": str(best_depth),
            "selected_min_samples_leaf": best_leaf, "selected_class_weight": best_cw or "none",
            "selected_threshold": threshold, "inner_oof_average_precision": float(best[0]),
            **metrics(y[test_idx], prob, threshold),
        }
        fold_rows.append(row)
        print(f"[outer fold {fold}/{outer_n}] n={len(test_idx)} AP={row['average_precision']:.4f} "
              f"F1={row['f1_bot']:.4f} (n_estimators={best_n_est}, max_depth={best_depth}, "
              f"min_samples_leaf={best_leaf}, class_weight={best_cw})")
    if np.isnan(oof).any():
        raise RuntimeError("Some users did not receive OOF predictions.")
    return oof, np.array([r["selected_threshold"] for r in fold_rows]), fold_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["d1", "d2"], required=True)
    p.add_argument("--behavioral-file", required=True,
                   help="Excel workbook with the behavioral feature columns.")
    p.add_argument("--behavioral-sheet", default=None)
    p.add_argument("--behavioral-id-column", default=None,
                   help="Defaults to auto-detect (id/user_id).")
    p.add_argument("--user-file", required=True,
                   help="Workbook containing biography text (may be the same path as --behavioral-file).")
    p.add_argument("--user-sheet", default=None)
    p.add_argument("--tweet-file", default=None,
                   help="Workbook containing tweet text; may be same as --user-file.")
    p.add_argument("--tweet-sheet", nargs="+", default=None,
                   help="One or more tweet sheet names (e.g. --tweet-sheet tweetMetaData_1 tweetMetaData_2).")
    p.add_argument("--label-column", default="final-tag")
    p.add_argument("--positive-label", default="bot")
    p.add_argument("--negative-label", default="human")
    p.add_argument("--tweet-user-id-column", default=None)
    p.add_argument("--bio-column", default=None)
    p.add_argument("--tweet-text-column", default=None)
    p.add_argument("--tweet-timestamp-column", default=None)
    p.add_argument("--max-tweets", type=int, default=20)
    p.add_argument("--require-nonempty-text", action="store_true",
                   help="Drop users with no bio/tweet text (default: keep them -- see assumption 3).")
    p.add_argument("--level", choices=["word", "char"], default="word")
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-features", type=int, default=3000,
                   help="TF-IDF vocabulary cap -- see assumption 4.")
    p.add_argument("--n-estimators-grid", type=int,
                   nargs="+", default=[300, 600])
    p.add_argument("--max-depth-grid", type=lambda v: None if v.lower() == "none" else int(v),
                   nargs="+", default=[None, 20])
    p.add_argument("--min-samples-leaf-grid",
                   type=int, nargs="+", default=[1, 3])
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=4)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--include-botometer", action="store_true")
    p.add_argument("--include-llm-scores", action="store_true")
    p.add_argument("--include-manual-review-flags", action="store_true")
    p.add_argument("--extra-behavioral-features", nargs="+", default=[])
    p.add_argument("--exclude-behavioral-features", nargs="+", default=[])
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- behavioral side ----
    behavioral_df = read_sheet(args.behavioral_file, args.behavioral_sheet)
    if args.label_column not in behavioral_df.columns:
        raise SystemExit(
            f"Label column {args.label_column!r} not found in --behavioral-file. "
            f"Available: {list(behavioral_df.columns)}")
    behav_id_col = find_column(behavioral_df, args.behavioral_id_column,
                               ["id", "user_id", "original.user_id"], "behavioral user ID")
    if behav_id_col is None:
        raise SystemExit("Could not find an id column in --behavioral-file.")
    behavioral_df["__fusion_key"] = behavioral_df[behav_id_col].map(
        robust_normalize_id_for_join)
    dup_mask = behavioral_df["__fusion_key"].notna(
    ) & behavioral_df["__fusion_key"].duplicated()
    if dup_mask.any():
        warnings.warn(f"{int(dup_mask.sum())} duplicate normalized IDs in --behavioral-file "
                      "(after dropping the last 5 digits); keeping the first occurrence of each.")
        behavioral_df = behavioral_df.drop_duplicates(
            subset="__fusion_key", keep="first")

    # ---- content/text side (reuses run_text_tfidf2.py's own join logic) ----
    text_users = read_sheet(args.user_file, args.user_sheet)
    tweets = None
    if args.tweet_file:
        tweet_sheets = args.tweet_sheet or [None]
        tweet_frames = []
        for sheet_name in tweet_sheets:
            sheet_frame = read_sheet(args.tweet_file, sheet_name)
            sheet_frame["__source_tweet_sheet"] = str(
                sheet_name if sheet_name is not None else "default")
            tweet_frames.append(sheet_frame)
        tweets = pd.concat(tweet_frames, ignore_index=True, sort=False)
    text_frame, coverage = make_user_documents(
        text_users, tweets, user_id_col=None, tweet_user_id_col=args.tweet_user_id_column,
        bio_col=args.bio_column, tweet_text_col=args.tweet_text_column,
        timestamp_col=args.tweet_timestamp_column, max_tweets=args.max_tweets,
    )
    text_id_col = find_column(
        text_frame, None, ["id", "user_id", "original.user_id"], "text-side user ID")
    if text_id_col is None:
        raise SystemExit(
            "Could not find an id column in --user-file to join on.")
    text_frame["__fusion_key"] = text_frame[text_id_col].map(
        robust_normalize_id_for_join)
    text_frame = text_frame.dropna(subset=["__fusion_key"]).drop_duplicates(
        subset="__fusion_key", keep="first")

    # ---- merge on last-5-digit-stripped id ----
    merged = behavioral_df.merge(
        text_frame[["__fusion_key", "text_document"]], on="__fusion_key", how="inner")
    print(f"Merged users (behavioral \u2229 text, by last-5-digit-stripped id): {len(merged)} "
          f"(behavioral file had {len(behavioral_df)}, text side had {len(text_frame)})")
    if merged.empty:
        raise SystemExit(
            "No users matched between --behavioral-file and --user-file/--tweet-file after the id join.")

    merged["__y"] = merged[args.label_column].map(
        lambda v: canonical_label(v, args.positive_label, args.negative_label))
    eligible = merged[merged["__y"].notna()].copy()
    if args.require_nonempty_text:
        eligible = eligible[eligible["text_document"].ne("")]
    eligible["__y"] = eligible["__y"].astype(int)
    if eligible.empty or eligible["__y"].nunique() != 2:
        raise SystemExit(
            f"After filtering, need both bot and human labels. Counts: {eligible['__y'].value_counts().to_dict()}")

    # ---- behavioral feature selection ----
    behavioral_cols = list(BEHAVIORAL_FEATURE_WHITELIST)
    if args.include_botometer:
        behavioral_cols.append("Botometer Score")
    if args.include_llm_scores:
        behavioral_cols += ["LLM(ChatGPT)", "LLM(Gemini)", "Grok 4"]
    if args.include_manual_review_flags:
        behavioral_cols += MANUAL_REVIEW_FLAG_COLS
    behavioral_cols += args.extra_behavioral_features
    exclude = set(args.exclude_behavioral_features)
    behavioral_cols = [c for c in dict.fromkeys(
        behavioral_cols) if c not in exclude]
    missing = [c for c in behavioral_cols if c not in eligible.columns]
    if missing:
        warnings.warn(
            f"These behavioral columns were not found in --behavioral-file and will be skipped: {missing}")
        behavioral_cols = [c for c in behavioral_cols if c in eligible.columns]
    if not behavioral_cols:
        raise SystemExit(
            "No behavioral feature columns resolved -- nothing to fuse with the text features.")

    for col in behavioral_cols:
        if col in BOOLEAN_BEHAVIORAL_COLS:
            eligible[col] = to_bool_numeric(eligible[col])
        else:
            eligible[col] = pd.to_numeric(eligible[col], errors="coerce")

    # Drop columns that are entirely missing in THIS dataset before CV starts.
    # Leaving them in makes SimpleImputer silently skip them per-fold (it has
    # nothing to impute from), which can vary fold to fold -- cleaner to drop
    # them once, consistently, up front.
    all_nan_cols = [
        c for c in behavioral_cols if eligible[c].notna().sum() == 0]
    if all_nan_cols:
        warnings.warn(f"These behavioral columns have no observed (non-missing) values in "
                      f"--dataset {args.dataset} and will be dropped: {all_nan_cols}")
        behavioral_cols = [c for c in behavioral_cols if c not in all_nan_cols]
    if not behavioral_cols:
        raise SystemExit(
            "No behavioral feature columns have any observed values in this dataset.")
    print(
        f"Behavioral features actually used ({len(behavioral_cols)}): {behavioral_cols}")

    y = eligible["__y"].to_numpy()
    X_df = eligible[["text_document"] + behavioral_cols].reset_index(drop=True)

    outdir = Path(
        args.output_dir or f"results_fusion_{args.level}_{args.dataset}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset={args.dataset}; label={args.label_column}; eligible={len(eligible)}; "
          f"bots={int(y.sum())}; humans={int((y == 0).sum())}")
    print(f"Text coverage: {json.dumps(coverage, ensure_ascii=False)}")

    oof, fold_thresholds, folds = nested_oof_fusion(
        X_df, y, behavioral_cols, args)
    thresholds_by_row = np.empty(len(y), dtype=float)
    outer = StratifiedKFold(n_splits=len(
        folds), shuffle=True, random_state=SEED)
    for threshold, (_, test_idx) in zip(fold_thresholds, outer.split(np.zeros(len(y)), y)):
        thresholds_by_row[test_idx] = threshold
    pred = (oof >= thresholds_by_row).astype(int)

    aggregate = metrics(y, oof, 0.5)
    aggregate["threshold_note"] = ("Ranking metrics use OOF probabilities; precision/recall/F1/accuracy "
                                   "use each outer fold's threshold selected from inner OOF predictions.")
    aggregate["threshold_median"] = float(np.median(fold_thresholds))
    aggregate["n_bot"] = int(y.sum())
    aggregate["n_human"] = int((y == 0).sum())
    aggregate["dataset"] = args.dataset
    aggregate["experiment"] = f"fusion_rf_{args.level}_{args.dataset}"
    aggregate["label_column"] = args.label_column
    aggregate["behavioral_features"] = behavioral_cols
    aggregate.update({
        "precision_bot_nested_threshold": float(precision_score(y, pred, zero_division=0)),
        "recall_bot_nested_threshold": float(recall_score(y, pred, zero_division=0)),
        "f1_bot_nested_threshold": float(f1_score(y, pred, zero_division=0)),
        "accuracy_nested_threshold": float(accuracy_score(y, pred)),
        "confusion_matrix_nested_threshold_tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
    })

    id_col = find_column(eligible, args.behavioral_id_column, [
                         "id", "user_id", "original.user_id"], "user ID")
    predictions = pd.DataFrame({
        "user_id": eligible[id_col].astype(str).to_numpy() if id_col else eligible.index.astype(str),
        "true_label_bot_1_human_0": y, "oof_bot_probability": oof,
        "selected_threshold": thresholds_by_row, "predicted_label_bot_1_human_0": pred,
    })
    predictions.to_csv(outdir / "oof_predictions.csv", index=False)
    pd.DataFrame(folds).to_csv(outdir / "fold_metrics.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(aggregate,
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "dataset_summary.json").write_text(json.dumps({
        **coverage, "label_column": args.label_column, "eligible_users": int(len(eligible)),
        "excluded_ambiguous_or_unlabeled": int(merged["__y"].isna().sum()),
        "bot_count": int(y.sum()), "human_count": int((y == 0).sum()), "positive_class": "bot",
        "seed": SEED, "outer_splits_actual": len(folds), "inner_splits_requested": args.inner_splits,
        "level": args.level, "max_features": args.max_features, "min_df": args.min_df,
        "n_estimators_grid": args.n_estimators_grid, "max_depth_grid": [str(d) for d in args.max_depth_grid],
        "min_samples_leaf_grid": args.min_samples_leaf_grid, "behavioral_features": behavioral_cols,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    fpr, tpr, _ = roc_curve(y, oof)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        outdir / "roc_curve.csv", index=False)
    precision, recall, _ = precision_recall_curve(y, oof)
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
        outdir / "pr_curve.csv", index=False)

    metrics_row = {
        "dataset": args.dataset, "experiment": aggregate["experiment"], "label_column": args.label_column,
        "n_users": int(len(y)), "n_bot": int(y.sum()), "n_human": int((y == 0).sum()),
        "accuracy": aggregate["accuracy_nested_threshold"], "precision": aggregate["precision_bot_nested_threshold"],
        "recall": aggregate["recall_bot_nested_threshold"], "roc_auc": aggregate["roc_auc"],
        "ap": aggregate["average_precision"], "f1": aggregate["f1_bot_nested_threshold"],
        "median_selected_threshold": aggregate["threshold_median"],
    }
    pd.DataFrame([metrics_row]).to_csv(
        outdir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    save_report_figures(outdir, y, oof, pred, folds, aggregate)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
