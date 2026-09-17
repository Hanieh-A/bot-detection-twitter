"""
extra_features.py
------------------
Engineers the behavioral signals that exist in the student's own dataset
but were NOT part of the original Katyal (2026) feature set. This is the
"innovation" layer on top of the reproduced paper.

Design choices (documented so they can go straight into the report):
  * Heavily right-skewed counts/rates (favourites, retweets, gaps, media,
    time-between-tweets) are log1p-transformed, same convention as
    features.py uses for followers/friends/statuses.
  * The four raw tweet-type counts (no_type_tweet, no_type_retweet_with_comment,
    no_type_reply, no_retweet_tweets) are converted to PROPORTIONS of a
    user's own sampled tweets rather than kept as raw counts -- raw counts
    would just re-encode overall activity level, which is already captured
    by statuses_per_day/log_statuses in features.py. Proportions capture
    behavioral *composition* instead (e.g. "mostly retweets" vs "mostly
    original tweets"), which is the actually-new signal.
  * "posted within the account's first day/week" is kept as a binary flag:
    an account that starts tweeting the instant it's created is a classic
    bot-farm signal.
"""
import numpy as np
import pandas as pd

EXTRA_RAW_COLS = [
    'tweet_frequency', 'retweet_ratio', 'mean_no_hashtags', 'mean_no_words',
    'time_between_tweets', 'mean_user_mentions_per_tweet', 'unique_mention_rate_per_tweet',
    'mean_favourites_per_tweet', 'mean_retweets_per_tweet', 'retweet_as_tweet_rate',
    'max_tweets_per_hour', 'max_tweets_per_day', 'max_occurence_of_same_gap',
    'no_type_tweet', 'no_type_retweet_with_comment', 'no_type_reply', 'no_retweet_tweets',
    'no_tweets_on_creation_day', 'no_tweets_on_creation_week', 'no_languages',
    'media_count', 'mean_no_media_per_tweet',
]


def engineer_extra_features(df):
    """df must contain the columns in EXTRA_RAW_COLS (as produced by
    load_custom_dataset.load_custom). Returns a new DataFrame, same index,
    of cleaned/engineered extra behavioral features only."""
    out = pd.DataFrame(index=df.index)

    # --- composition of tweet types (proportions, not raw counts) ---
    type_cols = ['no_type_tweet', 'no_type_retweet_with_comment', 'no_type_reply', 'no_retweet_tweets']
    total_typed = df[type_cols].sum(axis=1).replace(0, np.nan)
    out['prop_original_tweet'] = df['no_type_tweet'] / total_typed
    out['prop_retweet_with_comment'] = df['no_type_retweet_with_comment'] / total_typed
    out['prop_reply'] = df['no_type_reply'] / total_typed
    out['prop_plain_retweet'] = df['no_retweet_tweets'] / total_typed

    # --- rates / ratios already normalized, used as-is ---
    out['retweet_ratio'] = df['retweet_ratio']
    out['mean_no_hashtags'] = df['mean_no_hashtags']
    out['mean_no_words'] = df['mean_no_words']
    out['mean_user_mentions_per_tweet'] = df['mean_user_mentions_per_tweet']
    out['unique_mention_rate_per_tweet'] = df['unique_mention_rate_per_tweet']
    out['mean_no_media_per_tweet'] = df['mean_no_media_per_tweet']
    out['no_languages'] = df['no_languages']

    # --- skewed counts/rates -> log1p ---
    for src, dst in [
        ('tweet_frequency', 'log_tweet_frequency'),
        ('time_between_tweets', 'log_time_between_tweets'),
        ('mean_favourites_per_tweet', 'log_mean_favourites_per_tweet'),
        ('mean_retweets_per_tweet', 'log_mean_retweets_per_tweet'),
        ('retweet_as_tweet_rate', 'log_retweet_as_tweet_rate'),
        ('max_tweets_per_hour', 'log_max_tweets_per_hour'),
        ('max_tweets_per_day', 'log_max_tweets_per_day'),
        ('max_occurence_of_same_gap', 'log_max_occurence_of_same_gap'),
        ('media_count', 'log_media_count'),
    ]:
        out[dst] = np.log1p(df[src].clip(lower=0))

    # --- bot-farm-style signal: tweeted immediately after account creation ---
    out['tweeted_on_creation_day_or_week'] = (
        (df['no_tweets_on_creation_day'] > 0) | (df['no_tweets_on_creation_week'] > 0)
    ).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out


EXTRA_FEATURE_NAMES = [
    'prop_original_tweet', 'prop_retweet_with_comment', 'prop_reply', 'prop_plain_retweet',
    'retweet_ratio', 'mean_no_hashtags', 'mean_no_words', 'mean_user_mentions_per_tweet',
    'unique_mention_rate_per_tweet', 'mean_no_media_per_tweet', 'no_languages',
    'log_tweet_frequency', 'log_time_between_tweets', 'log_mean_favourites_per_tweet',
    'log_mean_retweets_per_tweet', 'log_retweet_as_tweet_rate', 'log_max_tweets_per_hour',
    'log_max_tweets_per_day', 'log_max_occurence_of_same_gap', 'log_media_count',
    'tweeted_on_creation_day_or_week',
]
