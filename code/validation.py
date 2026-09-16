"""Dataset audits used before the official experiments."""
from __future__ import annotations
import pandas as pd


def audit_dataset1(df):
    required = {"id_str", "screen_name", "bot", "status", "created_at"}
    missing = sorted(required - set(df.columns))
    return {
        "required_columns_missing": missing,
        "rows": int(len(df)),
        "resolved_labels": int(df.bot.notna().sum()),
        "duplicate_id_str": int(df.id_str.duplicated().sum()),
        "duplicate_screen_name": int(df.screen_name.astype(str).str.lower().duplicated().sum()),
        "users_with_text": int(df.status.fillna("").astype(str).str.len().gt(0).sum()),
    }


def audit_dataset2(df, tweets=None):
    required = {"id_str", "screen_name", "bot", "label_source", "botometer_score"}
    report = {
        "required_columns_missing": sorted(required - set(df.columns)),
        "rows": int(len(df)),
        "resolved_labels": int(df.bot.notna().sum()),
        "label_source_counts": {str(k): int(v) for k, v in df.label_source.value_counts(dropna=False).items()},
        "duplicate_id_str": int(df.id_str.astype(str).duplicated().sum()),
        "duplicate_normalized_screen_name": int(df.screen_name.astype(str).str.strip().str.lower().duplicated().sum()),
    }
    if tweets is not None:
        names = set(df.screen_name.astype(str).str.strip().str.lower())
        tweet_names = tweets.screen_name.astype(str).str.strip().str.lower()
        report.update({
            "tweet_rows": int(len(tweets)),
            "tweet_duplicate_screen_name_tweet_id": int(tweets.duplicated(["screen_name", "tweet_id"]).sum()),
            "tweet_screen_names_matched_to_main": int(tweet_names.isin(names).sum()),
            "tweet_screen_names_unmatched_to_main": int((~tweet_names.isin(names)).sum()),
        })
    return report
