# ============================================================
# Rule-Based Bot Classifier — نسخه نهایی
# اجرا روی دو دیتاست + ارزیابی کل + ارزیابی هر قاعده جداگانه + SHAP
# ============================================================

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1) نرمال‌سازی نام ستون‌ها
# ============================================================
def normalize_columns(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    if 'retweet_as_tweet_rate' in df.columns and 'retweet_ratio' in df.columns:
        df = df.drop(columns=['retweet_as_tweet_rate'])
    if 'statuses_count' in df.columns and 'status_count' in df.columns:
        df = df.drop(columns=['statuses_count'])

    rename_map = {
        'created_at': 'created_at',
        'retweet_as_tweet_rate': 'retweet_ratio',
        'followers_friend_ratio': 'followers_friend_ratio',
        'user_age': 'user_age',
        'tweet_frequency': 'tweet_frequency',
        'statuses_count': 'status_count',
        'num_digits_in_username': 'num_digits_in_username',
        'screen_name': 'screen_name',
        'name': 'name',
        'final-tag': 'true_label',
        'Predicted_Label': 'true_label',
    }
    df = df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


# ============================================================
# 2) مهندسی ویژگی‌ها
# ============================================================
def engineer_features(df):
    df = df.copy()
    if 'num_digits_in_username' not in df.columns:
        src_col = 'screen_name' if 'screen_name' in df.columns else 'name'
        if src_col in df.columns:
            df['num_digits_in_username'] = df[src_col].astype(str).apply(
                lambda x: sum(c.isdigit() for c in x)
            )
    if 'retweet_ratio' not in df.columns:
        if 'no_retweet_tweets' in df.columns and 'status_count' in df.columns:
            df['retweet_ratio'] = df['no_retweet_tweets'] / \
                df['status_count'].replace(0, np.nan)
        else:
            df['retweet_ratio'] = np.nan
    if 'followers_friend_ratio' not in df.columns:
        if 'followers_count' in df.columns and 'friends_count' in df.columns:
            df['followers_friend_ratio'] = df['followers_count'] / \
                df['friends_count'].replace(0, np.nan)
    if 'tweet_frequency' not in df.columns and 'max_tweets_per_day' in df.columns:
        df['tweet_frequency'] = df['max_tweets_per_day']
    return df


# ============================================================
# 3) قواعد دسته‌بندی ترکیبی (همان نسخه قبلی)
# ============================================================
def _get_val(row, key, default=np.nan):
    if key not in row.index:
        return default
    val = row[key]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def classify_user(row):
    tp = _get_val(row, 'tweets_per_day')
    rtr = _get_val(row, 'retweet_ratio')
    age = _get_val(row, 'user_age_days')
    ffr = _get_val(row, 'ff_ratio')
    dig = _get_val(row, 'digits_in_username', 0)
    freq = _get_val(row, 'tweet_freq')

    score_bot, score_news, score_human = 0, 0, 0
    reasons = []

    if pd.notna(age):
        if age < 30:
            score_bot += 2
            reasons.append("سن حساب < 30 روز")
        elif age < 90:
            score_bot += 1
            reasons.append("سن حساب < 90 روز")
        elif age > 365:
            score_human += 1

    if pd.notna(tp):
        if tp > 200:
            score_bot += 3
            reasons.append(f"توییت در روز خیلی بالا ({tp:.0f})")
        elif tp > 50:
            score_bot += 2
            reasons.append(f"توییت در روز بالا ({tp:.0f})")
        elif tp > 20:
            score_bot += 1
        elif 1 <= tp <= 15:
            score_human += 1

    if pd.notna(rtr):
        if rtr > 0.8:
            score_bot += 3
            reasons.append(f"نسبت بازنشر بسیار بالا ({rtr:.2f})")
        elif rtr > 0.6:
            score_bot += 2
            reasons.append(f"نسبت بازنشر بالا ({rtr:.2f})")
        elif rtr > 0.4:
            score_news += 2
            reasons.append(f"نسبت بازنشر متوسط-بالا (خبری) ({rtr:.2f})")
        elif 0.1 < rtr < 0.5:
            score_news += 1

    if pd.notna(ffr):
        if ffr < 0.05:
            score_bot += 2
            reasons.append(f"نسبت follower/friend بسیار پایین ({ffr:.3f})")
        elif ffr < 0.2:
            score_bot += 1
        elif ffr > 50:
            score_news += 1
            reasons.append(f"نسبت follower/friend بسیار بالا ({ffr:.1f})")
        elif 0.5 <= ffr <= 5:
            score_human += 1

    if pd.notna(dig):
        if dig >= 5:
            score_bot += 3
            reasons.append(f"تعداد رقم زیاد در نام کاربری ({int(dig)})")
        elif dig >= 3:
            score_bot += 1
            reasons.append(f"تعداد رقم متوسط در نام کاربری ({int(dig)})")

    if pd.notna(freq):
        if freq > 300:
            score_bot += 2
            reasons.append(f"فراوانی توییت بالا ({freq:.0f})")
        elif freq > 100:
            score_bot += 1

    if score_news >= 2 and score_bot <= 2:
        label = "News Agent"
    elif score_bot >= 5:
        label = "Bot"
    elif score_bot >= 3 and score_human <= 1:
        label = "Likely Bot"
    elif score_human >= 2 and score_bot <= 1:
        label = "Human"
    else:
        label = "Unverified"

    return pd.Series({
        'predicted_label': label,
        'bot_score': score_bot,
        'news_score': score_news,
        'human_score': score_human,
        'reasons': " | ".join(reasons) if reasons else "-"
    })


# ============================================================
# 4) پردازش دیتاست
# ============================================================
def classify_dataset(df, dataset_name="dataset"):
    print(f"\n{'='*60}\nپردازش {dataset_name} — شکل اولیه: {df.shape}\n{'='*60}")
    df = normalize_columns(df)
    df = engineer_features(df)

    if 'max_tweets_per_day' in df.columns:
        df['tweets_per_day'] = df['max_tweets_per_day']
    elif 'tweet_frequency' in df.columns:
        df['tweets_per_day'] = df['tweet_frequency']
    else:
        df['tweets_per_day'] = np.nan

    if 'user_age' in df.columns:
        df['user_age_days'] = df['user_age']
    else:
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(
                df['created_at'], errors='coerce', utc=True)
            df['user_age_days'] = (
                pd.Timestamp.utcnow() - df['created_at']).dt.days
        else:
            df['user_age_days'] = np.nan

    df['ff_ratio'] = df.get('followers_friend_ratio', np.nan)
    df['tweet_freq'] = df.get(
        'tweet_frequency', df.get('tweets_per_day', np.nan))
    df['digits_in_username'] = df.get('num_digits_in_username', 0)

    results = df.apply(classify_user, axis=1)
    df = pd.concat([df, results], axis=1)

    print("\nتوزیع برچسب‌های پیش‌بینی‌شده:")
    print(df['predicted_label'].value_counts())
    return df


# ============================================================
# 5) ذخیره نتایج
# ============================================================
def save_results(df, dataset_name, results_dir="results_rule_based"):
    os.makedirs(results_dir, exist_ok=True)
    excel_path = os.path.join(results_dir, f"{dataset_name}_classified.xlsx")
    csv_path = os.path.join(results_dir, f"{dataset_name}_classified.csv")
    summary_path = os.path.join(results_dir, f"{dataset_name}_summary.csv")
    report_path = os.path.join(results_dir, f"{dataset_name}_report.txt")

    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    df['predicted_label'].value_counts().to_csv(
        summary_path, encoding='utf-8-sig')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"گزارش دسته‌بندی قاعده‌محور — {dataset_name}\n")
        f.write(
            f"تاریخ اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"تعداد کل رکوردها: {len(df)}\n")
        f.write("="*50 + "\n")
        f.write(df['predicted_label'].value_counts().to_string() + "\n")

    print(f"\n✅ نتایج {dataset_name} ذخیره شد.")


# ============================================================
# 6) نگاشت برچسب‌ها
# ============================================================
def normalize_labels(series):
    def _map(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s in ('1', '1.0'):
            return 'Bot'
        if s in ('2', '2.0'):
            return 'Human'
        if s in ('3', '3.0'):
            return 'News Agent'
        if s in ('4', '4.0'):
            return 'Unverified'
        if 'bot' in s and 'likely' not in s:
            return 'Bot'
        if 'human' in s:
            return 'Human'
        if 'news' in s or 'agent' in s:
            return 'News Agent'
        if 'unverified' in s or 'unknown' in s:
            return 'Unverified'
        return np.nan
    return series.apply(_map)


def normalize_predicted(series):
    def _map(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if 'likely' in s and 'bot' in s:
            return 'Bot'
        if s == 'bot':
            return 'Bot'
        if s == 'human':
            return 'Human'
        if 'news' in s:
            return 'News Agent'
        if 'unverified' in s:
            return 'Unverified'
        return val
    return series.apply(_map)


# ============================================================
# 7) ارزیابی کل (ترکیبی) + گزارش تصویری
# ============================================================
def evaluate_and_plot(df, dataset_name, results_dir="results_rule_based"):
    if 'true_label' not in df.columns:
        print(f"⚠️ {dataset_name}: ستون true_label موجود نیست.")
        return None

    df_eval = df.copy()
    df_eval['true_std'] = normalize_labels(df_eval['true_label'])
    df_eval['pred_std'] = normalize_predicted(df_eval['predicted_label'])
    mask = df_eval['true_std'].notna() & df_eval['pred_std'].notna()
    n_total = len(df_eval)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return None

    print(f"\n{'='*60}\nارزیابی کل {dataset_name}\n{'='*60}")
    print(f"کل: {n_total} | قابل ارزیابی: {n_valid}")

    y_true = df_eval.loc[mask, 'true_std']
    y_pred = df_eval.loc[mask, 'pred_std']
    labels_order = ['Bot', 'Human', 'News Agent', 'Unverified']
    present = [l for l in labels_order if l in set(y_true) | set(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0
    )
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.average(f1, weights=support)
                        ) if support.sum() > 0 else 0.0

    print(f"Accuracy: {acc*100:.2f}%  |  Macro F1: {macro_f1:.3f}")

    out_dir = os.path.join(results_dir, f"eval_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present, yticklabels=present, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(
        f'Confusion Matrix — {dataset_name}\nAccuracy = {acc*100:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(
        out_dir, f'{dataset_name}_confusion_matrix.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(present))
    w = 0.25
    ax.bar(x - w, precision, w, label='Precision', color='#4C72B0')
    ax.bar(x,     recall,    w, label='Recall',    color='#55A868')
    ax.bar(x + w, f1,        w, label='F1',        color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'Per-class Metrics — {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(
        out_dir, f'{dataset_name}_per_class_metrics.png'), dpi=150)
    plt.close()

    return {'dataset': dataset_name, 'n_valid': n_valid,
            'accuracy': float(acc), 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}


# ============================================================
# 8) مقایسه دو دیتاست (ترکیبی)
# ============================================================
def plot_comparison(summaries, results_dir="results_rule_based"):
    summaries = [s for s in summaries if s is not None]
    if not summaries:
        return
    df = pd.DataFrame(summaries)
    out_dir = os.path.join(results_dir, "eval_comparison")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df['accuracy'],    w, label='Accuracy',    color='#4C72B0')
    ax.bar(x,     df['macro_f1'],    w, label='Macro F1',    color='#55A868')
    ax.bar(x + w, df['weighted_f1'], w, label='Weighted F1', color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(df['dataset'])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Rule-Based Classifier — Dataset Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'datasets_comparison.png'), dpi=150)
    plt.close()


# ============================================================
# 9) [جدید] قواعد مستقل + ارزیابی هر قاعده جداگانه
# ============================================================
# هر قاعده یک امتیاز پیوسته در [0,1] برمی‌گرداند
# (بالاتر = احتمال بیشتر بات بودن بر اساس آن قاعده)

def _rule_age(row):
    v = _get_val(row, 'user_age_days')
    if pd.isna(v):
        return np.nan
    if v < 30:
        return 1.0
    if v < 90:
        return 0.6
    if v < 365:
        return 0.3
    return 0.0


def _rule_tweets_per_day(row):
    v = _get_val(row, 'tweets_per_day')
    if pd.isna(v):
        return np.nan
    if v > 200:
        return 1.0
    if v > 50:
        return 0.7
    if v > 20:
        return 0.4
    return 0.0


def _rule_retweet_ratio(row):
    v = _get_val(row, 'retweet_ratio')
    if pd.isna(v):
        return np.nan
    if v > 0.8:
        return 1.0
    if v > 0.6:
        return 0.7
    if v > 0.4:
        return 0.4
    return 0.0


def _rule_ff_ratio(row):
    v = _get_val(row, 'ff_ratio')
    if pd.isna(v):
        return np.nan
    if v < 0.05:
        return 1.0
    if v < 0.2:
        return 0.5
    return 0.0


def _rule_digits_in_username(row):
    v = _get_val(row, 'digits_in_username', 0)
    if pd.isna(v):
        return np.nan
    if v >= 5:
        return 1.0
    if v >= 3:
        return 0.4
    return 0.0


def _rule_tweet_freq(row):
    v = _get_val(row, 'tweet_freq')
    if pd.isna(v):
        return np.nan
    if v > 300:
        return 1.0
    if v > 100:
        return 0.5
    return 0.0


RULES = {
    'Rule_Age':            {'fn': _rule_age,              'desc': 'سن حساب کم',                 'threshold': 0.5},
    'Rule_TweetsPerDay':   {'fn': _rule_tweets_per_day,   'desc': 'توییت در روز بالا',          'threshold': 0.5},
    'Rule_RetweetRatio':   {'fn': _rule_retweet_ratio,    'desc': 'نسبت بازنشر بالا',           'threshold': 0.5},
    'Rule_FFRatio':        {'fn': _rule_ff_ratio,         'desc': 'نسبت follower/friend پایین', 'threshold': 0.4},
    'Rule_DigitsUsername': {'fn': _rule_digits_in_username, 'desc': 'تعداد رقم زیاد در username', 'threshold': 0.4},
    'Rule_TweetFreq':      {'fn': _rule_tweet_freq,       'desc': 'فراوانی توییت بالا',         'threshold': 0.5},
}


def evaluate_each_rule(df, dataset_name, results_dir="results_rule_based"):
    """
    هر قاعده به‌طور مستقل اجرا و ارزیابی می‌شود.
    مقایسه: پیش‌بینی دودویی قاعده (Bot vs non-Bot) در برابر برچسب حقیقی.
    AUC با استفاده از امتیاز پیوسته‌ی قاعده محاسبه می‌شود.
    """
    if 'true_label' not in df.columns:
        print(
            f"⚠️ {dataset_name}: ستون true_label موجود نیست — ارزیابی قواعد رد شد.")
        return None

    df_eval = df.copy()
    df_eval['true_std'] = normalize_labels(df_eval['true_label'])
    df_eval['is_bot_true'] = (df_eval['true_std'] == 'Bot').astype(int)
    mask = df_eval['true_std'].notna()
    df_eval = df_eval[mask].reset_index(drop=True)

    if len(df_eval) == 0:
        print(f"⚠️ {dataset_name}: رکورد قابل ارزیابی نیست.")
        return None

    n_bot_true = int(df_eval['is_bot_true'].sum())
    n_nonbot_true = int((1 - df_eval['is_bot_true']).sum())

    print(f"\n{'='*60}\nارزیابی قواعد جداگانه — {dataset_name}\n{'='*60}")
    print(
        f"کل رکورد قابل ارزیابی: {len(df_eval)}  (Bot={n_bot_true}، non-Bot={n_nonbot_true})")

    out_dir = os.path.join(results_dir, f"rules_eval_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for rule_name, info in RULES.items():
        scores = df_eval.apply(lambda r: info['fn'](r), axis=1)
        valid_mask = scores.notna()
        n_valid = int(valid_mask.sum())
        if n_valid < 10:
            print(f"⚠️ {rule_name}: رکورد کافی نیست ({n_valid})")
            continue

        y_true = df_eval.loc[valid_mask, 'is_bot_true'].values
        y_score = scores[valid_mask].values.astype(float)
        y_pred = (y_score >= info['threshold']).astype(int)

        # محاسبه متریک‌ها
        acc = float(accuracy_score(y_true, y_pred))
        f1_val = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))

        if len(np.unique(y_true)) < 2:
            auc_val = np.nan
        else:
            try:
                auc_val = float(roc_auc_score(y_true, y_score))
            except Exception:
                auc_val = np.nan

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        results.append({
            'rule': rule_name,
            'description': info['desc'],
            'n_valid': n_valid,
            'threshold': info['threshold'],
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1_val, 4),
            'auc': round(auc_val, 4) if not np.isnan(auc_val) else np.nan,
            'TP': int(cm[1, 1]),
            'FP': int(cm[0, 1]),
            'TN': int(cm[0, 0]),
            'FN': int(cm[1, 0]),
        })

        # confusion matrix جداگانه برای هر قاعده
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['non-Bot', 'Bot'], yticklabels=['non-Bot', 'Bot'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        auc_str = f"{auc_val:.3f}" if not np.isnan(auc_val) else "N/A"
        ax.set_title(
            f'{rule_name}\n{info["desc"]}\nAcc={acc*100:.1f}%  F1={f1_val:.3f}  AUC={auc_str}')
        plt.tight_layout()
        plt.savefig(os.path.join(
            out_dir, f'{rule_name}_confusion.png'), dpi=150)
        plt.close()

    if not results:
        print("⚠️ هیچ قاعده‌ای قابل ارزیابی نبود.")
        return None

    results_df = pd.DataFrame(results)

    # ذخیره CSV
    csv_path = os.path.join(out_dir, f'{dataset_name}_rules_eval.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ذخیره گزارش متنی
    txt_path = os.path.join(out_dir, f'{dataset_name}_rules_eval.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"ارزیابی مستقل قواعد — {dataset_name}\n")
        f.write(f"تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"تعداد رکورد: {len(df_eval)}\n")
        f.write(
            f"Bot واقعی: {n_bot_true}  |  non-Bot واقعی: {n_nonbot_true}\n")
        f.write("="*70 + "\n")
        f.write(results_df.to_string(index=False) + "\n")

    print("\n📊 جدول ارزیابی قواعد:")
    print(results_df.to_string(index=False))
    print(f"\n✅ ذخیره شد: {csv_path}")
    print(f"✅ ذخیره شد: {txt_path}")

    # نمودار مقایسه‌ای هر سه متریک
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(results_df))
    w = 0.25
    ax.bar(x - w, results_df['accuracy'], w, label='Accuracy', color='#4C72B0')
    ax.bar(x,     results_df['f1'],       w, label='F1',       color='#55A868')
    auc_plot = results_df['auc'].fillna(0).clip(lower=0).values
    ax.bar(x + w, auc_plot, w, label='AUC', color='#C44E52')

    for i, row in results_df.iterrows():
        ax.text(i - w, row['accuracy'] + 0.01,
                f"{row['accuracy']*100:.0f}%", ha='center', fontsize=8)
        ax.text(i,     row['f1'] + 0.01,
                f"{row['f1']*100:.0f}%",       ha='center', fontsize=8)
        if not pd.isna(row['auc']):
            ax.text(i + w, row['auc'] + 0.01,
                    f"{row['auc']*100:.0f}%", ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df['rule'], rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title(
        f'Rule-by-Rule Evaluation — {dataset_name}  (vs. Bot-vs-Rest)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    cmp_path = os.path.join(out_dir, f'{dataset_name}_rules_comparison.png')
    plt.savefig(cmp_path, dpi=150)
    plt.close()
    print(f"✅ ذخیره شد: {cmp_path}")

    return results_df


# ============================================================
# 10) اجرای اصلی
# ============================================================
if __name__ == "__main__":
    RESULTS_DIR = "../results_rule_based"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    SEARCH_DIRS = [Path("."), Path("./data"), Path("./datasets"),
                   Path("../data"), Path("../datasets"), Path("..")]

    def find_file(filename):
        for d in SEARCH_DIRS:
            candidate = d / filename
            if candidate.exists():
                return candidate
        for p in Path("..").rglob(filename):
            return p
        return None

    dataset_files = {
        "dataset1": "../datasets/dataset1.xlsx",
        "dataset2": "../datasets/dataset2.xlsx",
    }

    all_outputs = {}
    eval_summaries = []
    rules_eval_all = {}   # ← جدید

    for name, filename in dataset_files.items():
        path = find_file(filename)
        if path is None:
            print(f"⚠️ فایل {filename} پیدا نشد.")
            continue
        print(f"\n📂 {name} پیدا شد: {path.resolve()}")
        df = pd.read_excel(path)
        out = classify_dataset(df, name)
        save_results(out, name, RESULTS_DIR)

        # ارزیابی کلی
        summary = evaluate_and_plot(out, name, RESULTS_DIR)
        if summary:
            eval_summaries.append(summary)

        # ← جدید: ارزیابی مستقل هر قاعده
        rule_df = evaluate_each_rule(out, name, RESULTS_DIR)
        if rule_df is not None:
            rules_eval_all[name] = rule_df

        all_outputs[name] = out

    # گزارش ترکیبی
    if all_outputs:
        combined_path = os.path.join(RESULTS_DIR, "combined_summary.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("خلاصه ترکیبی دیتاست‌ها\n")
            f.write(
                f"تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for name, out in all_outputs.items():
                f.write(f"--- {name} ({len(out)} رکورد) ---\n")
                f.write(
                    out['predicted_label'].value_counts().to_string() + "\n\n")
            if eval_summaries:
                f.write("="*50 + "\nخلاصه ارزیابی کلی:\n")
                for s in eval_summaries:
                    f.write(f"{s['dataset']}: Acc={s['accuracy']*100:.2f}%  "
                            f"MacroF1={s['macro_f1']:.3f}\n")
            if rules_eval_all:
                f.write("\n" + "="*50 + "\nخلاصه ارزیابی قواعد جداگانه:\n")
                for dname, rdf in rules_eval_all.items():
                    f.write(f"\n--- {dname} ---\n")
                    f.write(rdf.to_string(index=False) + "\n")
        print(f"\n📊 گزارش ترکیبی: {combined_path}")

    # مقایسه تصویری کلی
    plot_comparison(eval_summaries, RESULTS_DIR)

    print("\n🎉 اجرا کامل شد.")
