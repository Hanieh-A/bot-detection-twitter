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


def compute_coordination_features(xlsx_path, min_text_len=15):
    """Returns a DataFrame indexed by id_str (string) with:
      - n_sampled_tweets
      - n_duplicated_tweets       : tweets whose (normalized) text also
                                     appears under >=1 OTHER account
      - duplicate_tweet_ratio     : n_duplicated_tweets / n_sampled_tweets
      - n_coordination_partners   : # of distinct other accounts sharing
                                     at least one duplicated text
    Tweets shorter than min_text_len chars (after normalization) are
    excluded from duplicate-matching to avoid false positives on generic
    short replies ("سلام", "ok", etc.).
    """
    tw = pd.read_excel(xlsx_path, sheet_name="tweetsMetaData")
    tw["id"] = tw["id"].astype(str)
    tw["norm"] = tw["text"].apply(_normalize)
    tw.loc[tw["norm"].str.len() < min_text_len, "norm"] = np.nan  # too short to trust

    valid = tw.dropna(subset=["norm"]).copy()
    # groups of normalized text -> set of distinct user ids using it
    text_to_users = valid.groupby("norm")["id"].agg(lambda s: set(s))
    dup_texts = text_to_users[text_to_users.apply(len) > 1]
    dup_text_set = set(dup_texts.index)

    valid["is_dup"] = valid["norm"].isin(dup_text_set)

    rows = []
    for uid, g in tw.groupby("id"):
        n_tweets = len(g)
        g_valid = g.dropna(subset=["norm"])
        dup_mask = g_valid["norm"].isin(dup_text_set)
        n_dup = int(dup_mask.sum())

        partners = set()
        for norm_text in g_valid.loc[dup_mask, "norm"].unique():
            partners |= (text_to_users[norm_text] - {uid})

        rows.append({
            "id_str": uid,
            "n_sampled_tweets": n_tweets,
            "n_duplicated_tweets": n_dup,
            "duplicate_tweet_ratio": n_dup / n_tweets if n_tweets else 0.0,
            "n_coordination_partners": len(partners),
        })

    out = pd.DataFrame(rows).set_index("id_str")
    return out


COORD_FEATURE_NAMES = ["duplicate_tweet_ratio", "n_coordination_partners"]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "users_with_retweets.xlsx"
    df = compute_coordination_features(path)
    print(df.shape)
    print(df.sort_values("n_coordination_partners", ascending=False).head(10))
    print("\nUsers with ANY coordination signal:", (df["n_coordination_partners"] > 0).sum())
