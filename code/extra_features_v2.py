"""
extra_features_v2.py
---------------------
Same idea as extra_features.py (dataset 1), adapted to the columns that
actually exist in dataset 2 (all_users.xlsx). Differences from dataset 1:
  * no raw 'retweet_ratio' column -> approximate as no_retweet_tweets / statuses_count
  * no 'no_tweets_on_creation_day' / 'no_tweets_on_creation_week' -> that
    bot-farm flag is NOT available here and is simply omitted
"""
import numpy as np
import pandas as pd

TYPE_COLS = ['no_type_tweet', 'no_type_retweet_with_comment', 'no_type_reply', 'no_retweet_tweets']


def engineer_extra_features_v2(df, statuses_col='statuses_count'):
    """df must contain load_dataset2's output columns (extra raw cols +
    statuses_count). Returns a DataFrame of engineered extra features."""
    out = pd.DataFrame(index=df.index)

    total_typed = df[TYPE_COLS].sum(axis=1).replace(0, np.nan)
    out['prop_original_tweet'] = df['no_type_tweet'] / total_typed
    out['prop_retweet_with_comment'] = df['no_type_retweet_with_comment'] / total_typed
    out['prop_reply'] = df['no_type_reply'] / total_typed
    out['prop_plain_retweet'] = df['no_retweet_tweets'] / total_typed

    out['retweet_ratio_approx'] = df['no_retweet_tweets'] / df[statuses_col].replace(0, np.nan)

    out['mean_no_hashtags'] = df['mean_no_hashtags']
    out['mean_no_words'] = df['mean_no_words']
    out['mean_user_mentions_per_tweet'] = df['mean_user_mentions_per_tweet']
    out['unique_mention_rate_per_tweet'] = df['unique_mention_rate_per_tweet']
    out['mean_no_media_per_tweet'] = df['mean_no_media_per_tweet']
    out['no_languages'] = df['no_languages']

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

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out


EXTRA_FEATURE_NAMES_V2 = [
    'prop_original_tweet', 'prop_retweet_with_comment', 'prop_reply', 'prop_plain_retweet',
    'retweet_ratio_approx', 'mean_no_hashtags', 'mean_no_words', 'mean_user_mentions_per_tweet',
    'unique_mention_rate_per_tweet', 'mean_no_media_per_tweet', 'no_languages',
    'log_tweet_frequency', 'log_time_between_tweets', 'log_mean_favourites_per_tweet',
    'log_mean_retweets_per_tweet', 'log_retweet_as_tweet_rate', 'log_max_tweets_per_hour',
    'log_max_tweets_per_day', 'log_max_occurence_of_same_gap', 'log_media_count',
]
