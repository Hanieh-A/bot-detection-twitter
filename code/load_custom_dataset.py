"""
Adapter: load the student's own labeled dataset (users_with_retweets.xlsx)
into the same schema the original features.py / run_analysis.py expect.

Usage:
    from load_custom_dataset import load_custom
    df = load_custom("/path/to/users_with_retweets.xlsx")
    # df has columns: id_str, screen_name, name, description, location, url,
    # followers_count, friends_count, listed_count, statuses_count,
    # favourites_count, created_at, verified, default_profile,
    # default_profile_image, status, bot
"""
import re
import numpy as np
import pandas as pd

FINAL_LABEL_COL = "برچسب نهایی"  # authoritative resolved label, e.g. "bot(1)"


def _parse_final_label(v):
    """Extract the numeric class from strings like 'bot(1)', 'human(2)',
    'News Agent(3)', 'unverified(4)'. Returns np.nan for "can't find" or
    any other unparsable value."""
    if pd.isna(v):
        return np.nan
    m = re.search(r'\((\d+)\)', str(v))
    return float(m.group(1)) if m else np.nan


def load_custom(xlsx_path, tweet_agg="concat", max_tweets_per_user=20):
    """
    Parameters
    ----------
    xlsx_path : path to users_with_retweets.xlsx
    tweet_agg : "concat" (join several tweets into one string) or "first"
    max_tweets_per_user : cap on how many tweets to pull per user for "concat"

    Returns
    -------
    df : DataFrame in the schema features.py expects, with an extra
         'label_raw' column (1=bot,2=human,3=news agent,4=unverified) kept
         for reference, and 'bot' = binary target (1=bot, 0=all others),
         only for rows where 'برچسب نهایی' resolved to a real class
         (i.e. not "can't find").
    """
    users = pd.read_excel(xlsx_path, sheet_name="users_with_retweets")
    tweets = pd.read_excel(xlsx_path, sheet_name="tweetsMetaData")

    # ---- 1. label straight from the resolved "برچسب نهایی" column ----
    users["label_raw"] = users[FINAL_LABEL_COL].apply(_parse_final_label)
    users["bot"] = (users["label_raw"] == 1).astype(float)
    users.loc[users["label_raw"].isna(), "bot"] = np.nan

    # ---- 2. per-user tweet text (now ~174/1104 users have rows here) ----
    tweets = tweets.copy()
    tweets["id"] = tweets["id"].astype(str)
    if tweet_agg == "first":
        tw_text = tweets.groupby("id")["text"].first()
    else:
        tw_text = (
            tweets.groupby("id")["text"]
            .apply(lambda s: " ".join(str(x) for x in s.head(max_tweets_per_user).tolist()))
        )
    tw_text.name = "status"

    users["id"] = users["id"].astype(str)
    users = users.merge(tw_text, left_on="id", right_index=True, how="left")
    users["status"] = users["status"].fillna("")

    # ---- 3. rename / build columns to match features.py's expectations ----
    extra_raw_cols = [
        'tweet_frequency', 'retweet_ratio', 'mean_no_hashtags', 'mean_no_words',
        'time_between_tweets', 'mean_user_mentions_per_tweet', 'unique_mention_rate_per_tweet',
        'mean_favourites_per_tweet', 'mean_retweets_per_tweet', 'retweet_as_tweet_rate',
        'max_tweets_per_hour', 'max_tweets_per_day', 'max_occurence_of_same_gap',
        'no_type_tweet', 'no_type_retweet_with_comment', 'no_type_reply', 'no_retweet_tweets',
        'no_tweets_on_creation_day', 'no_tweets_on_creation_week', 'no_languages',
        'media_count', 'mean_no_media_per_tweet',
    ]
    out = pd.DataFrame({
        "id_str": users["id"],
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
        "created_at": users["created_at"],        # ISO format, see features.py patch
        "verified": 0,                              # not collected -> assume 0
        "default_profile": users["default_profile"],
        "default_profile_image": 0,                 # not collected -> assume 0
        "status": users["status"],
        "bot": users["bot"],
        "label_raw": users["label_raw"],
    })
    # pass through the extra pre-computed behavioral columns (raw, unengineered)
    # -- available for ~all 1104 users, unlike tweet text which only ~162 have.
    for c in extra_raw_cols:
        out[c] = users[c]
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "users_with_retweets.xlsx"
    df = load_custom(path)
    print(df.shape)
    print(df["label_raw"].value_counts(dropna=False))
    print(df.dropna(subset=["bot"]).shape, "rows with a resolved bot/non-bot label")
    print((df["status"].str.len() > 0).sum(), "rows with at least one tweet of text")
