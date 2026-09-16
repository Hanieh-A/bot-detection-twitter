"""
dataset2_text_utils.py
------------------------
Pulls per-user aggregated tweet text from dataset 2's two tweet sheets
(tweetsMetaData_1, tweetsMetaData_2 -- both share the same schema, just
different collection batches/projects) and provides a coordination
(cross-account duplicate content) feature extractor analogous to
coordination_features.py from dataset 1.
"""
import re
import numpy as np
import pandas as pd
from collections import Counter

_URL_RE = re.compile(r'https?://\S+')
_WS_RE = re.compile(r'\s+')


def _normalize(text):
    if not isinstance(text, str):
        return ""
    t = _URL_RE.sub('', text)
    t = t.strip().lower()
    t = _WS_RE.sub(' ', t)
    return t


def load_all_tweets(xlsx_path, cache_pickle=None):
    """Combines tweetsMetaData_1 + tweetsMetaData_2 into one DataFrame
    with columns: screen_name, tweet_id (str), text. Deduplicates exact
    repeated (screen_name, tweet_id) rows that might appear in both sheets.
    NOTE: joins on screen_name rather than the numeric 'id' column --
    the 'main' sheet stores 'id' as float64, which silently loses
    precision for Twitter's 19-digit snowflake IDs (float64 can only
    exactly represent integers up to ~9e15), while the tweet sheets store
    it as int64. This makes numeric-id joins between 'main' and the tweet
    sheets unreliable; screen_name (a string) has no such precision issue.
    If cache_pickle is given and exists, loads from there instead of
    re-reading the (slow) xlsx sheets; otherwise computes and, if a path
    is given, saves the cache for next time."""
    import os
    if cache_pickle and os.path.exists(cache_pickle):
        return pd.read_pickle(cache_pickle)

    frames = []
    for sheet in ["tweetsMetaData_1", "tweetsMetaData_2"]:
        raw = pd.read_excel(xlsx_path, sheet_name=sheet)
        frames.append(pd.DataFrame({
            "screen_name": raw["screen_name"].astype(str).str.strip().str.lower(),
            "tweet_id": raw["id.1"].astype(str),
            "text": raw["text"],
        }))
    tw = pd.concat(frames, ignore_index=True)
    tw = tw.drop_duplicates(subset=["screen_name", "tweet_id"])
    if cache_pickle:
        tw.to_pickle(cache_pickle)
    return tw


def aggregate_text_per_user(tw, max_tweets_per_user=20):
    """tw: output of load_all_tweets. Returns a Series indexed by
    screen_name (lowercased) with concatenated tweet text (first
    max_tweets_per_user per user)."""
    return (
        tw.groupby("screen_name")["text"]
        .apply(lambda s: " ".join(str(x) for x in s.dropna().head(max_tweets_per_user).tolist()))
    )


def compute_coordination_features_d2(tw, min_text_len=15):
    """Same cross-account duplicate-content idea as coordination_features.py
    (dataset 1), applied to dataset 2's combined tweet sheets.
    Returns a DataFrame indexed by id (str) with duplicate_tweet_ratio and
    n_coordination_partners."""
    tw = tw.copy()
    tw["norm"] = tw["text"].apply(_normalize)
    tw.loc[tw["norm"].str.len() < min_text_len, "norm"] = np.nan

    valid = tw.dropna(subset=["norm"])
    text_to_users = valid.groupby("norm")["screen_name"].agg(lambda s: set(s))
    dup_texts = text_to_users[text_to_users.apply(len) > 1]
    dup_text_set = set(dup_texts.index)

    rows = []
    for uid, g in tw.groupby("screen_name"):
        n_tweets = len(g)
        g_valid = g.dropna(subset=["norm"])
        dup_mask = g_valid["norm"].isin(dup_text_set)
        n_dup = int(dup_mask.sum())
        partners = set()
        for norm_text in g_valid.loc[dup_mask, "norm"].unique():
            partners |= (text_to_users[norm_text] - {uid})
        rows.append({
            "screen_name": uid, "n_sampled_tweets": n_tweets,
            "n_duplicated_tweets": n_dup,
            "duplicate_tweet_ratio": n_dup / n_tweets if n_tweets else 0.0,
            "n_coordination_partners": len(partners),
        })
    return pd.DataFrame(rows).set_index("screen_name")
