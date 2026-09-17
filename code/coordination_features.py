"""
coordination_features.py
-------------------------
Detects cross-account duplicate/near-duplicate tweet content -- a classic
"coordinated inauthentic behavior" signal that neither the behavioral (45)
nor the per-user linguistic (17) features capture, since both only look
at a single account in isolation.

Exploratory analysis on this dataset found >1500 tweet texts shared
verbatim across multiple accounts (political-campaign-style copy-paste),
suggesting this could be a much stronger signal than generic linguistic
statistics for this particular dataset.

Usage:
    from coordination_features import compute_coordination_features
    coord_df = compute_coordination_features(xlsx_path)  # indexed by id_str
"""
import re
import pandas as pd
import numpy as np

_URL_RE = re.compile(r'https?://\S+')
_WS_RE = re.compile(r'\s+')


def _normalize(text):
    if not isinstance(text, str):
        return ""
    t = _URL_RE.sub('', text)          # ignore different t.co links
    t = t.strip().lower()
    t = _WS_RE.sub(' ', t)
    return t


def compute_coordination_features(xlsx_path, min_text_len=15, sheet_name=None, group_col=None):
    """Returns a DataFrame indexed by the grouping key (screen_name, lowercased,
    for the dataset-1-v2 schema; id_str for the original schema) with:
      - n_sampled_tweets
      - n_duplicated_tweets       : tweets whose (normalized) text also
                                     appears under >=1 OTHER account
      - duplicate_tweet_ratio     : n_duplicated_tweets / n_sampled_tweets
      - n_coordination_partners   : # of distinct other accounts sharing
                                     at least one duplicated text
    Tweets shorter than min_text_len chars (after normalization) are
    excluded from duplicate-matching to avoid false positives on generic
    short replies ("سلام", "ok", etc.).

    sheet_name / group_col let this work across both dataset-1 schemas:
      - original file (users_with_retweets.xlsx): sheet "tweetsMetaData",
        where the "id" column IS the user id -> group_col="id"
      - enriched file (1000user_sheet.xlsx): sheet "tweets_meta_data",
        where "id" is the TWEET id (not the user id) and tweets must be
        grouped by "screen_name" instead.
    If not given, both are auto-detected from the sheet names present.
    """
    if sheet_name is None:
        xl = pd.ExcelFile(xlsx_path)
        if "tweetsMetaData" in xl.sheet_names:
            sheet_name, group_col = "tweetsMetaData", (group_col or "id")
        elif "tweets_meta_data" in xl.sheet_names:
            sheet_name, group_col = "tweets_meta_data", (group_col or "screen_name")
        else:
            raise ValueError("No known tweet-metadata sheet found in this file.")
    group_col = group_col or "id"

    tw = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if group_col == "screen_name":
        tw["group_key"] = tw["screen_name"].astype(str).str.strip().str.lower()
    else:
        tw["group_key"] = tw[group_col].astype(str)
    tw["norm"] = tw["text"].apply(_normalize)
    tw.loc[tw["norm"].str.len() < min_text_len, "norm"] = np.nan  # too short to trust

    valid = tw.dropna(subset=["norm"]).copy()
    # groups of normalized text -> set of distinct user keys using it
    text_to_users = valid.groupby("norm")["group_key"].agg(lambda s: set(s))
    dup_texts = text_to_users[text_to_users.apply(len) > 1]
    dup_text_set = set(dup_texts.index)

    rows = []
    for uid, g in tw.groupby("group_key"):
        n_tweets = len(g)
        g_valid = g.dropna(subset=["norm"])
        dup_mask = g_valid["norm"].isin(dup_text_set)
        n_dup = int(dup_mask.sum())

        partners = set()
        for norm_text in g_valid.loc[dup_mask, "norm"].unique():
            partners |= (text_to_users[norm_text] - {uid})

        rows.append({
            "group_key": uid,
            "n_sampled_tweets": n_tweets,
            "n_duplicated_tweets": n_dup,
            "duplicate_tweet_ratio": n_dup / n_tweets if n_tweets else 0.0,
            "n_coordination_partners": len(partners),
        })

    out = pd.DataFrame(rows).set_index("group_key")
    return out


COORD_FEATURE_NAMES = ["duplicate_tweet_ratio", "n_coordination_partners"]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "users_with_retweets.xlsx"
    df = compute_coordination_features(path)
    print(df.shape)
    print(df.sort_values("n_coordination_partners", ascending=False).head(10))
    print("\nUsers with ANY coordination signal:", (df["n_coordination_partners"] > 0).sum())
