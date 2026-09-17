"""
load_custom_dataset_v2.py
---------------------------
Adapter for the ENRICHED version of dataset 1 (1000user_sheet.xlsx),
which supersedes the original users_with_retweets.xlsx. Same underlying
population (~1103 vs 1104 users, same label distribution) but with:
  * Real 'verified' and 'default_profile_image' values (previously
    unavailable and hardcoded to 0 in load_custom_dataset.py).
  * A Botometer Score column (previously only available in dataset 2).
  * Tweet text for 1099/1103 users via the 'tweets_meta_data' sheet
    (vs only 174/1104 in the original file) -- joined on screen_name,
    consistent with the precision-safe join approach used for dataset 2.
  * A few extra precomputed columns (listed_growth_rate,
    followers_friend_ratio, description_length, num_digits_in_name,
    num_digits_in_username, url_in_description) not used here but
    available for future feature engineering.

Usage:
    from load_custom_dataset_v2 import load_custom_v2
    df = load_custom_v2("/path/to/1000user_sheet.xlsx")
"""
import re
import numpy as np
import pandas as pd

FINAL_LABEL_COL = "برچسب نهایی"

EXTRA_RAW_COLS = [
    'tweet_frequency', 'retweet_ratio', 'mean_no_hashtags', 'mean_no_words',
    'time_between_tweets', 'mean_user_mentions_per_tweet', 'unique_mention_rate_per_tweet',
    'mean_favourites_per_tweet', 'mean_retweets_per_tweet', 'retweet_as_tweet_rate',
    'max_tweets_per_hour', 'max_tweets_per_day', 'max_occurence_of_same_gap',
    'no_type_tweet', 'no_type_retweet_with_comment', 'no_type_reply', 'no_retweet_tweets',
    'no_tweets_on_creation_day', 'no_tweets_on_creation_week', 'no_languages',
    'media_count', 'mean_no_media_per_tweet',
]


def _parse_final_label(v):
    if pd.isna(v):
        return np.nan
    m = re.search(r'\((\d+)\)', str(v))
    return float(m.group(1)) if m else np.nan


def _clean_botometer(v):
    if pd.isna(v):
        return np.nan
    m = re.match(r'\s*([\d.]+)\s*/\s*5', str(v))
    return float(m.group(1)) / 5.0 if m else np.nan


def load_custom_v2(xlsx_path, max_tweets_per_user=20):
    users = pd.read_excel(xlsx_path, sheet_name="1k_users")
    tweets = pd.read_excel(xlsx_path, sheet_name="tweets_meta_data")

    users["label_raw"] = users[FINAL_LABEL_COL].apply(_parse_final_label)
    users["bot"] = (users["label_raw"] == 1).astype(float)
    users.loc[users["label_raw"].isna(), "bot"] = np.nan

    # join on screen_name (lowercased) -- safe against numeric-id precision
    # loss, and this file's tweet sheet has near-universal screen_name coverage.
    users["screen_name_norm"] = users["screen_name"].astype(str).str.strip().str.lower()
    tweets["screen_name_norm"] = tweets["screen_name"].astype(str).str.strip().str.lower()
    tw_text = (
        tweets.groupby("screen_name_norm")["text"]
        .apply(lambda s: " ".join(str(x) for x in s.dropna().head(max_tweets_per_user).tolist()))
    )
    users = users.set_index("screen_name_norm")
    users["status"] = tw_text.reindex(users.index).fillna("")
    users = users.reset_index(drop=True)

    users["botometer_score"] = users["Botometer Score"].apply(_clean_botometer)

    out = pd.DataFrame({
        "id_str": users["id"].astype(str),
        "screen_name": users["screen_name"],
        "name": users["name"],
        "description": users["clean_description"],
        "location": users["location"],
        "url": users["url"],
        "followers_count": users["followers_count"],
        "friends_count": users["friends_count"],
        "listed_count": users["listed_count"],
        "statuses_count": users["status_count"],
        "favourites_count": users["favourites_count"],
        "created_at": users["created_at"],
        "verified": users["verified"],                     # real values now
        "default_profile": users["default_profile"],
        "default_profile_image": users["default_profile_image"],  # real values now
        "status": users["status"],
        "bot": users["bot"],
        "label_raw": users["label_raw"],
        "botometer_score": users["botometer_score"],
    })
    for c in EXTRA_RAW_COLS:
        out[c] = users[c]
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "1000user_sheet.xlsx"
    df = load_custom_v2(path)
    print(df.shape)
    print(df["label_raw"].value_counts(dropna=False))
    print(df.dropna(subset=["bot"]).shape, "rows with a resolved bot/non-bot label")
    print((df["status"].str.len() > 0).sum(), "rows with at least one tweet of text")
    print(df["botometer_score"].notna().sum(), "rows with a Botometer score")
