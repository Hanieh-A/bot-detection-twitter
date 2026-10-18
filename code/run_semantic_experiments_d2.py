"""Additive semantic-embedding experiments for dataset 2 (Persian Twitter bots).

Run from the repository root:
    python code/run_semantic_experiments_d2.py
    python code/run_semantic_experiments_d2.py --max-tweets 10 --text-mode mean

Differences from run_semantic_experiments.py (D1):
  * D2 has one row per USER in sheet "main" and MANY tweets per user in the
    sheets tweetsMetaData_1/2, so tweets are joined to users by numeric USER ID
    (never by screen_name: users can change it), with a tolerance on the last
    N characters of the id (Excel stores only ~15 significant digits, so long
    ids are corrupted in their last digits).
  * Id columns are read as text (dtype=str) and the user-id column of each tweet
    sheet is auto-detected (the 'id'-like column matching the most users).
  * Labels come from load_dataset2 (tagger probability >= 0.5 -> bot,
    reinforced by suspended accounts).
  * Semantic features: by default every sampled tweet is embedded separately and
    the embeddings are mean-pooled per user.
  * Results go to results_semantic_d2/ (D1 outputs are never touched).
Botometer scores are NOT used as features (external tool, kept out on purpose).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

from data_audit import audit_dataset2
from experiment_config import (
    DATASET2_PATH, INNER_SPLITS, OUTER_SPLITS, RF_PARAMS, SEED, THRESHOLD_GRID
)
from extra_features_d2 import TYPE_COLS, engineer_extra_features_d2
from features import engineer_behavioral_features, engineer_content_features
from load_dataset2 import load_dataset2
from run_semantic_experiments import evaluate_one, save_plots, sha256
from semantic_features import DEFAULT_MODEL_NAME, engineer_semantic_features

ROOT_RESULTS = Path(__file__).resolve().parent.parent / "results_semantic_d2"
TWEET_SHEETS = ("tweetsMetaData_1", "tweetsMetaData_2")


# ----------------------------------------------------------------- ids ----
def norm_id_scalar(x):
    """Clean digit string, or None. Handles '123.0' and '1.89e+18'."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    if "." in s or "e" in s.lower():
        try:
            d = Decimal(s)
            if d != d.to_integral_value():
                return None
            s = str(int(d))
        except (InvalidOperation, ValueError):
            return None
    return s if s.isdigit() else None


def norm_id(series):
    return series.map(norm_id_scalar)


def safe_name(model_name):
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
    )


def map_tweet_ids_to_users(tw, user_ids, tol, min_len):
    """Map tweet-sheet user ids onto main-sheet user ids.

    1) exact match first;
    2) for tweet ids without an exact match, tolerate differences in the last
       `tol` characters (prefix match) -- only for ids with >= `min_len`
       characters, and only when the prefix identifies exactly ONE main user
       (ambiguous prefixes are never merged).
    Returns (mapping tweet_id -> main_id, number of fuzzy-matched tweet ids).
    """
    users = {u for u in user_ids if isinstance(u, str) and u}
    tweet_ids = pd.Series(tw["user_id"].dropna().unique())
    mapping = {i: i for i in tweet_ids if i in users}
    n_fuzzy = 0
    if tol > 0:
        prefix_to_users = {}
        for u in users:
            if len(u) >= min_len:
                prefix_to_users.setdefault(u[:-tol], set()).add(u)
        for i in tweet_ids:
            if i in mapping or len(i) < min_len:
                continue
            cands = prefix_to_users.get(i[:-tol], set())
            if len(cands) == 1:
                mapping[i] = next(iter(cands))
                n_fuzzy += 1
    return mapping, n_fuzzy


def read_main_user_ids(xlsx_path, fallback):
    """user_id column of 'main' read as TEXT (no float parsing), row-aligned
    with load_dataset2's output."""
    s = pd.read_excel(xlsx_path, sheet_name="main", usecols=["user_id"], dtype=str)["user_id"]
    if len(s) != len(fallback):
        print("WARNING: main sheet row count differs from loader output; "
              "using loader ids.")
        return norm_id(fallback.astype(str))
    ids = pd.Series(norm_id(s).to_numpy(), index=fallback.index)
    return ids.fillna(norm_id(fallback.astype(str)))


# --------------------------------------------------------------- tweets ----
def load_tweets(xlsx_path, cache_dir, user_ids, tol, min_len):
    """All tweets from both sheets: user_id (raw, normalized), screen_name,
    tweet_id, text, lang. For each sheet the user-id column is auto-detected:
    among columns whose name contains 'id' (not 'project'), the one that maps
    onto the most distinct main users. Cached keyed by workbook size/mtime and
    the tolerance settings."""
    xlsx_path = Path(xlsx_path)
    stat = xlsx_path.stat()
    cache_path = cache_dir / (
        f"tweets_{stat.st_size}_{int(stat.st_mtime)}_tol{tol}_min{min_len}.pkl")
    if cache_path.exists():
        print(f"Loading cached tweets: {cache_path}")
        return pd.read_pickle(cache_path)

    frames = []
    for sheet in TWEET_SHEETS:
        print(f"Reading sheet {sheet} (slow for large xlsx files)...")
        raw = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)
        cands = [c for c in raw.columns
                 if "id" in str(c).lower() and "project" not in str(c).lower()]
        best_col, best_score, best_ids = None, 0, None
        for col in cands:
            ids = norm_id(raw[col])
            mapping, _ = map_tweet_ids_to_users(
                pd.DataFrame({"user_id": ids}), user_ids, tol, min_len)
            score = len(set(mapping.values()))
            print(f"  candidate id column {col!r}: {score} matched users")
            if score > best_score:
                best_col, best_score, best_ids = col, score, ids
        if best_col is None:
            print(f"  WARNING: no id column of {sheet!r} matches any user; sheet skipped.")
            continue
        print(f"  -> using {best_col!r} for {sheet!r}")

        tweet_col = "id.1" if "id.1" in raw.columns else None
        fallback_tid = pd.Series([f"{sheet}_{i}" for i in range(len(raw))], index=raw.index)
        tweet_id = (norm_id(raw[tweet_col]).fillna(fallback_tid)
                    if tweet_col else fallback_tid)
        frames.append(pd.DataFrame({
            "user_id": best_ids,
            "screen_name": raw["screen_name"].astype(str).str.strip().str.lower(),
            "tweet_id": tweet_id,
            "text": raw["text"],
            "lang": raw["lang"] if "lang" in raw.columns else np.nan,
        }))
    if not frames:
        raise ValueError("No tweet sheet could be matched to users by id.")
    tw = pd.concat(frames, ignore_index=True)
    tw = tw.dropna(subset=["user_id"])
    tw = tw.drop_duplicates(subset=["user_id", "tweet_id"])
    tw = tw[tw["text"].fillna("").astype(str).str.strip().ne("")].copy()
    tw["text"] = tw["text"].astype(str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tw.to_pickle(cache_path)
    return tw


def sample_tweets(tw, max_tweets, lang):
    """Deterministic per-user sample of at most max_tweets tweets."""
    if lang:
        tw = tw[tw["lang"] == lang]
    tw = tw.sample(frac=1, random_state=SEED).groupby("user_id").head(max_tweets)
    return tw.sort_values(["user_id", "tweet_id"]).reset_index(drop=True)


def semantic_user_features(matched, tw_sampled, mode, cache_dir):
    if mode == "concat":
        return engineer_semantic_features(matched, cache_dir=cache_dir)

    t = tw_sampled[tw_sampled["user_id"].isin(set(matched["id_str"]))].copy()
    t = t.rename(columns={"text": "status"}).reset_index(drop=True)
    print(f"Embedding {len(t)} tweets from {t['user_id'].nunique()} users...")
    emb = pd.DataFrame(engineer_semantic_features(t, cache_dir=cache_dir)).reset_index(drop=True)
    if len(emb) != len(t):
        raise ValueError(
            f"engineer_semantic_features returned {len(emb)} rows for {len(t)} tweets; "
            "it must return one row per input row (same order)."
        )
    pooled = emb.groupby(t["user_id"].to_numpy()).mean()
    out = pooled.reindex(matched["id_str"].to_numpy())
    out.index = matched.index
    return out


# ------------------------------------------------------------------ run ----
def run_d2(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    ref_date = datetime.fromisoformat(args.ref_date).replace(tzinfo=timezone.utc)

    # ---- users + labels -------------------------------------------------
    raw, _extra_cols = load_dataset2(str(DATASET2_PATH))
    raw["id_str"] = read_main_user_ids(DATASET2_PATH, raw["id_str"])
    audit = audit_dataset2(raw)
    n_bad_ids = int(raw["id_str"].isna().sum())
    if n_bad_ids:
        print(f"WARNING: {n_bad_ids} users in 'main' have a missing/invalid user_id "
              "and cannot be matched to tweets.")
    all_user_ids = set(raw["id_str"].dropna())
    df = raw.dropna(subset=["bot"]).copy()
    if args.tagger_only:
        df = df[df["label_source"] != "suspended_only"]
    df["bot"] = df["bot"].astype(int)
    print(f"Labeled users: {len(df)}  (bot rate {df['bot'].mean():.3f})")

    missing = [c for c in (*TYPE_COLS, "statuses_count") if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from load_dataset2 output: {missing}")

    # ---- tweets -> per-user text (joined by USER ID, with tolerance) ------
    tw = load_tweets(DATASET2_PATH, cache_dir, all_user_ids,
                     args.id_tolerance, args.id_tolerance_min_len)
    id_map, n_fuzzy = map_tweet_ids_to_users(
        tw, all_user_ids, args.id_tolerance, args.id_tolerance_min_len)
    n_before = len(tw)
    tw = tw.assign(user_id=tw["user_id"].map(id_map)).dropna(subset=["user_id"])
    users_with_tweets = set(tw["user_id"])
    print(f"Tweet ids -> users: {len(id_map)} distinct tweet ids mapped "
          f"({n_fuzzy} only via last-{args.id_tolerance}-char tolerance); "
          f"kept {len(tw)}/{n_before} tweets from {len(users_with_tweets)} users")
    lens = raw["id_str"].str.len()
    long_mask = lens >= args.id_tolerance_min_len
    for name, mask in (("long ids (tolerance applies)", long_mask),
                       ("short ids (exact only)", ~long_mask)):
        sub = raw.loc[mask, "id_str"]
        print(f"  {name}: {len(sub)} users, {sub.isin(users_with_tweets).sum()} with tweets")

    tw_s = sample_tweets(tw, args.max_tweets, args.lang)
    counts = tw_s.groupby("user_id").size()
    user_text = tw_s.groupby("user_id")["text"].apply(" ".join)

    df["n_sampled_tweets"] = df["id_str"].map(counts).fillna(0).astype(int)
    df["status"] = df["id_str"].map(user_text).fillna("")
    print(f"Labeled users with >=1 tweet: {(df['n_sampled_tweets'] > 0).mean():.1%}")

    matched = df[df["n_sampled_tweets"] >= args.min_tweets].reset_index(drop=True)
    if matched.empty:
        raise ValueError(
            "No labeled users with tweets. Check the id diagnostics printed above."
        )
    y = matched["bot"].to_numpy(dtype=int)
    min_class = int(np.bincount(y, minlength=2).min())
    if min_class < OUTER_SPLITS:
        raise ValueError(f"Smallest class has {min_class} users, fewer than OUTER_SPLITS={OUTER_SPLITS}.")
    print(f"Matched subset: n={len(matched)}, bots={int(y.sum())}, humans={int((1 - y).sum())}")

    # ---- features ----------------------------------------------------------
    matched_behavioral = engineer_behavioral_features(matched, ref_date=ref_date)
    matched_extended = matched_behavioral.join(engineer_extra_features_d2(matched))
    matched_content = engineer_content_features(matched)
    matched_semantic = semantic_user_features(matched, tw_s, args.text_mode, cache_dir)

    feature_sets = [
        ("RF behavioral extended (matched text subset)", matched_extended),
        ("RF simple content (matched text subset)", matched_content),
        ("RF semantic MiniLM (matched text subset)", matched_semantic),
        ("RF behavioral plus semantic MiniLM (matched text subset)",
         matched_extended.join(matched_semantic)),
    ]

    records, predictions, fold_frames = [], {}, []
    for model_name, X in feature_sets:
        print(f"\nEvaluating {model_name}: n={len(X)}, features={X.shape[1]}")
        summary, pred, folds = evaluate_one(
            "D2-matched-text-semantic", model_name, X.reset_index(drop=True), y
        )
        records.append(summary)
        predictions[model_name] = pred
        pred = pred.copy()
        pred["matched_row"] = np.arange(len(pred))
        pred["id_str"] = matched["id_str"].to_numpy()
        pred["screen_name"] = matched["screen_name"].to_numpy()
        pred.to_csv(output_dir / f"predictions_{safe_name(model_name)}.csv", index=False)
        fold_frames.append(folds)

    # Reference only: behavioral model on ALL labeled users (different n, not in plots).
    all_df = df.reset_index(drop=True)
    X_all = engineer_behavioral_features(all_df, ref_date=ref_date).join(
        engineer_extra_features_d2(all_df))
    name_all = "RF behavioral extended (all labeled users, reference)"
    print(f"\nEvaluating {name_all}: n={len(X_all)}, features={X_all.shape[1]}")
    s_all, p_all, f_all = evaluate_one(
        "D2-all-labeled-reference", name_all, X_all,
        all_df["bot"].to_numpy(dtype=int))
    records.append(s_all)
    p_all.to_csv(output_dir / f"predictions_{safe_name(name_all)}.csv", index=False)
    fold_frames.append(f_all)

    metrics = pd.DataFrame(records)
    metrics.to_csv(output_dir / "semantic_metrics.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        output_dir / "semantic_fold_metrics.csv", index=False)
    save_plots(predictions, y, output_dir)

    lang_counts = (tw_s[tw_s["user_id"].isin(set(matched["id_str"]))]["lang"]
                   .value_counts(dropna=False).head(8))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET2_PATH),
        "dataset_sha256": sha256(DATASET2_PATH),
        "semantic_model": DEFAULT_MODEL_NAME,
        "text_mode": args.text_mode,
        "max_tweets_per_user": args.max_tweets,
        "min_tweets_per_user": args.min_tweets,
        "lang_filter": args.lang,
        "tagger_only": args.tagger_only,
        "id_tolerance": args.id_tolerance,
        "id_tolerance_min_len": args.id_tolerance_min_len,
        "n_tweet_ids_fuzzy_matched": int(n_fuzzy),
        "n_users_with_tweets_all": int(len(users_with_tweets)),
        "ref_date": args.ref_date,
        "n_labeled_rows": int(len(df)),
        "n_matched_text_rows": int(len(matched)),
        "matched_class_counts": {"human": int((1 - y).sum()), "bot": int(y.sum())},
        "label_source_counts_matched": {
            str(k): int(v) for k, v in matched["label_source"].value_counts().items()},
        "sampled_tweet_lang_counts": {str(k): int(v) for k, v in lang_counts.items()},
        "seed": SEED,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "threshold_grid": list(THRESHOLD_GRID),
        "rf_params": RF_PARAMS,
        "audit": audit,
        "notes": [
            "Additive runner: existing scripts, nested_cv.py and D1 results are untouched.",
            "Tweets joined to users by numeric user id (text-read, tolerance on last digits), not screen_name.",
            "Botometer score is not used as a feature.",
            "Labels are Bot_Probability >= 0.5 (+ suspended accounts); labels near 0.5 are noisy.",
            "One row per user, so ordinary stratified nested CV has no user-level leakage.",
            "The semantic encoder is frozen; only the classifier is evaluated with nested CV.",
        ],
    }
    (output_dir / "semantic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nSemantic experiment metrics:")
    print(metrics.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--output-dir", default=str(ROOT_RESULTS))
    p.add_argument("--max-tweets", type=int, default=10,
                   help="Max sampled tweets per user (default 10).")
    p.add_argument("--min-tweets", type=int, default=1,
                   help="Keep only users with at least this many sampled tweets.")
    p.add_argument("--text-mode", choices=("mean", "concat"), default="mean",
                   help="mean: embed each tweet and average per user; "
                        "concat: embed one concatenated text per user.")
    p.add_argument("--lang", default=None,
                   help="Only use tweets with this 'lang' value (e.g. fa).")
    p.add_argument("--id-tolerance", type=int, default=5,
                   help="Ids whose last N characters differ still count as the same user "
                        "(only if no exact match exists). 0 = exact matching only.")
    p.add_argument("--id-tolerance-min-len", type=int, default=15,
                   help="Tolerance applies only to ids at least this long; shorter ids "
                        "(e.g. 9-10 digits) would collide between different users.")
    p.add_argument("--tagger-only", action="store_true",
                   help="Drop users labeled only through the suspended list.")
    p.add_argument("--ref-date", default="2025-04-01",
                   help="Reference date for account age (collection date).")
    run_d2(p.parse_args())


if __name__ == "__main__":
    main()