"""
load_dataset2.py
-----------------
Adapter for the SECOND, larger dataset (all_users.xlsx, ~19,510 users)
into the same schema features.py expects. This dataset is structurally
different from the first one (users_with_retweets.xlsx):

  * Labels are continuous tagger PROBABILITIES (0-1), not a discrete
    class column, and only ~2000/19510 users have one.
  * A Botometer Score (external, independent tool) is available for
    most users -- format "X.X/5" or "Error - Timeout".
  * A suspend_users sheet lists screen_names Twitter/X has actually
    suspended -- strong independent bot evidence, used to reinforce
    (never override) the human label.
  * followers_followings sheet has follower SCREEN-NAME lists (as a
    comma-separated string) for only ~3437/19510 users; the
    'followings' column is entirely empty in this export.

Usage:
    from load_dataset2 import load_dataset2
    df, extra_raw_cols = load_dataset2("/path/to/all_users.xlsx")
"""
import re
import numpy as np
import pandas as pd

# columns from the "main" sheet this dataset actually has, usable as
# extra (paper-1-style) behavioral signal beyond the original 24 features.
# (No retweet_ratio / no_tweets_on_creation_day|week / listed_growth_rate
# here, unlike dataset 1 -- so extra_features.py from dataset 1 can't be
# reused as-is; see extra_features_v2.py.)
EXTRA_RAW_COLS_V2 = [
    'tweet_frequency', 'mean_no_hashtags', 'mean_no_words', 'time_between_tweets',
    'mean_user_mentions_per_tweet', 'unique_mention_rate_per_tweet',
    'mean_favourites_per_tweet', 'mean_retweets_per_tweet', 'retweet_as_tweet_rate',
    'max_tweets_per_hour', 'max_tweets_per_day', 'max_occurence_of_same_gap',
    'no_type_tweet', 'no_type_retweet_with_comment', 'no_type_reply', 'no_retweet_tweets',
    'no_languages', 'media_count', 'mean_no_media_per_tweet',
]


def _clean_prob(v):
    """Robustly parse a tagger probability cell: handles floats, ints,
    typos like '0..6', and non-answers like '؟' / '?' (-> NaN)."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return f if 0 <= f <= 1 else np.nan
    s = str(v).strip().replace('..', '.')
    try:
        f = float(s)
        return f if 0 <= f <= 1 else np.nan
    except ValueError:
        return np.nan


def _clean_botometer(v):
    """'2.3/5' -> 0.46 ; 'Error - Timeout' / NaN -> NaN"""
    if pd.isna(v):
        return np.nan
    m = re.match(r'\s*([\d.]+)\s*/\s*5', str(v))
    if not m:
        return np.nan
    try:
        return float(m.group(1)) / 5.0
    except ValueError:
        return np.nan


def load_dataset2(xlsx_path):
    """Returns (df, extra_raw_cols):
      df : DataFrame in the schema features.py expects, plus:
           - 'bot'            : 1/0/NaN, human-tagger-derived label
                                 (>=0.5 -> bot), reinforced by suspend_users
           - 'label_source'   : 'tagger' or 'tagger+suspended' or 'suspended_only'
           - 'botometer_score': normalized 0-1 (independent tool, NOT used to
                                 derive 'bot' -- kept separate to avoid circularity)
      extra_raw_cols : list of raw column names (EXTRA_RAW_COLS_V2) present,
                       to be fed into extra_features_v2.engineer_extra_features_v2
    """
    main = pd.read_excel(xlsx_path, sheet_name="main")
    suspended = pd.read_excel(xlsx_path, sheet_name="suspend_users")

    main["prob1_clean"] = main["prob1"].apply(_clean_prob)
    main["bot"] = np.where(main["prob1_clean"].notna(), (main["prob1_clean"] >= 0.5).astype(float), np.nan)
    main["label_source"] = np.where(main["prob1_clean"].notna(), "tagger", None)

    # reinforce with suspended accounts (strong independent bot evidence);
    # never downgrades an existing human label, only fills in gaps or
    # flags a same-direction confirmation.
    sus_screen_names = set(suspended["screen_name"].dropna())
    is_sus = main["screen_name"].isin(sus_screen_names)
    newly_added = is_sus & main["bot"].isna()
    confirmed = is_sus & (main["bot"] == 1)
    conflicted = is_sus & (main["bot"] == 0)

    main.loc[newly_added, "bot"] = 1.0
    main.loc[newly_added, "label_source"] = "suspended_only"
    main.loc[confirmed, "label_source"] = "tagger+suspended"
    # conflicted rows (tagger said human, but account was later suspended):
    # kept as the tagger's original label, just flagged -- suspension can
    # happen for reasons unrelated to being a bot (policy strikes, etc.)
    main.loc[conflicted, "label_source"] = "tagger(human)+suspended_conflict"

    main["botometer_score"] = main["Botometer Score"].apply(_clean_botometer)

    out = pd.DataFrame({
        "id_str": main["id"].astype(str),
        "screen_name": main["screen_name"],
        "name": main["name"],
        "description": main["clean_description"],
        "location": main["location"],
        "url": main["url"],
        "followers_count": main["followers_count"],
        "friends_count": main["friends_count"],
        "listed_count": main["listed_count"],
        "statuses_count": main["status_count"],
        "favourites_count": main["favourites_count"],
        "created_at": main["created_at"],
        "verified": main["verified"],
        "default_profile": main["default_profile"],
        "default_profile_image": main["default_profile_image"],
        "status": "",  # no per-user aggregated tweet text column in this export
        "bot": main["bot"],
        "label_source": main["label_source"],
        "botometer_score": main["botometer_score"],
    })
    extra_present = [c for c in EXTRA_RAW_COLS_V2 if c in main.columns]
    for c in extra_present:
        out[c] = main[c]

    return out, extra_present


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "all_users.xlsx"
    df, extra_cols = load_dataset2(path)
    print(df.shape)
    print("label_source counts:\n", df["label_source"].value_counts(dropna=False))
    print("bot value counts:\n", df["bot"].value_counts(dropna=False))
    print("extra cols available:", extra_cols)
    print("botometer coverage:", df["botometer_score"].notna().sum())
