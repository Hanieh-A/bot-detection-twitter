# ============================================================
# Rule-Based Bot Classifier — نسخه نهایی
# اجرا روی دو دیتاست + ارزیابی + گزارش تصویری
# ============================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1) نرمال‌سازی نام ستون‌ها
# ============================================================
def normalize_columns(df):
    df = df.copy()

    # حذف ستون‌های بی‌استفاده
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # حل تکراری‌های احتمالی قبل از rename
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
        'class (1:bot, 2:human, 3:News Agent, 4:unverified)': 'true_label',
        'final-tag': 'true_label',
        'Predicted_Label': 'true_label',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # safeguard نهایی: هر ستون تکراری فقط بار اول نگه داشته شود
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


# ============================================================
# 2) مهندسی ویژگی‌های جایگزین
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
            df['retweet_ratio'] = df['no_retweet_tweets'] / df['status_count'].replace(0, np.nan)
        else:
            df['retweet_ratio'] = np.nan

    if 'followers_friend_ratio' not in df.columns:
        if 'followers_count' in df.columns and 'friends_count' in df.columns:
            df['followers_friend_ratio'] = df['followers_count'] / df['friends_count'].replace(0, np.nan)

    if 'tweet_frequency' not in df.columns and 'max_tweets_per_day' in df.columns:
        df['tweet_frequency'] = df['max_tweets_per_day']

    return df


# ============================================================
# 3) قواعد دسته‌بندی
# ============================================================
def _get_val(row, key, default=np.nan):
    """مقدار تک‌عددی برمی‌گردونه، حتی اگه ستون تکراری یا غیرعددی باشه."""
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
    tp   = _get_val(row, 'tweets_per_day')
    rtr  = _get_val(row, 'retweet_ratio')
    age  = _get_val(row, 'user_age_days')
    ffr  = _get_val(row, 'ff_ratio')
    dig  = _get_val(row, 'digits_in_username', 0)
    freq = _get_val(row, 'tweet_freq')

    score_bot, score_news, score_human = 0, 0, 0
    reasons = []

    if pd.notna(age):
        if age < 30:
            score_bot += 2; reasons.append(f"سن حساب < 30 روز")
        elif age < 90:
            score_bot += 1; reasons.append(f"سن حساب < 90 روز")
        elif age > 365:
            score_human += 1

    if pd.notna(tp):
        if tp > 200:
            score_bot += 3; reasons.append(f"توییت در روز خیلی بالا ({tp:.0f})")
        elif tp > 50:
            score_bot += 2; reasons.append(f"توییت در روز بالا ({tp:.0f})")
        elif tp > 20:
            score_bot += 1
        elif 1 <= tp <= 15:
            score_human += 1

    if pd.notna(rtr):
        if rtr > 0.8:
            score_bot += 3; reasons.append(f"نسبت بازنشر بسیار بالا ({rtr:.2f})")
        elif rtr > 0.6:
            score_bot += 2; reasons.append(f"نسبت بازنشر بالا ({rtr:.2f})")
        elif rtr > 0.4:
            score_news += 2; reasons.append(f"نسبت بازنشر متوسط-بالا (خبری) ({rtr:.2f})")
        elif 0.1 < rtr < 0.5:
            score_news += 1

    if pd.notna(ffr):
        if ffr < 0.05:
            score_bot += 2; reasons.append(f"نسبت follower/friend بسیار پایین ({ffr:.3f})")
        elif ffr < 0.2:
            score_bot += 1
        elif ffr > 50:
            score_news += 1; reasons.append(f"نسبت follower/friend بسیار بالا ({ffr:.1f})")
        elif 0.5 <= ffr <= 5:
            score_human += 1

    if pd.notna(dig):
        if dig >= 5:
            score_bot += 3; reasons.append(f"تعداد رقم زیاد در نام کاربری ({int(dig)})")
        elif dig >= 3:
            score_bot += 1; reasons.append(f"تعداد رقم متوسط در نام کاربری ({int(dig)})")

    if pd.notna(freq):
        if freq > 300:
            score_bot += 2; reasons.append(f"فراوانی توییت بالا ({freq:.0f})")
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
# 4) پردازش هر دیتاست
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
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
            df['user_age_days'] = (
                pd.Timestamp.utcnow() - df['created_at']
            ).dt.days
        else:
            df['user_age_days'] = np.nan

    df['ff_ratio'] = df.get('followers_friend_ratio', np.nan)
    df['tweet_freq'] = df.get('tweet_frequency', df.get('tweets_per_day', np.nan))
    df['digits_in_username'] = df.get('num_digits_in_username', 0)

    results = df.apply(classify_user, axis=1)
    df = pd.concat([df, results], axis=1)

    print("\nتوزیع برچسب‌های پیش‌بینی‌شده:")
    print(df['predicted_label'].value_counts())
    cols_show = [c for c in ['screen_name', 'predicted_label', 'bot_score', 'reasons'] if c in df.columns]
    print("\nچند نمونه:")
    print(df[cols_show].head(5).to_string())

    return df


# ============================================================
# 5) ذخیره نتایج
# ============================================================
def save_results(df, dataset_name, results_dir="results_rule_based"):
    os.makedirs(results_dir, exist_ok=True)

    excel_path = os.path.join(results_dir, f"{dataset_name}_classified.xlsx")
    csv_path   = os.path.join(results_dir, f"{dataset_name}_classified.csv")
    summary_path = os.path.join(results_dir, f"{dataset_name}_summary.csv")
    report_path  = os.path.join(results_dir, f"{dataset_name}_report.txt")

    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    df['predicted_label'].value_counts().to_csv(summary_path, encoding='utf-8-sig')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"گزارش دسته‌بندی قاعده‌محور — {dataset_name}\n")
        f.write(f"تاریخ اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"تعداد کل رکوردها: {len(df)}\n")
        f.write("="*50 + "\n")
        f.write("توزیع برچسب‌ها:\n")
        f.write(df['predicted_label'].value_counts().to_string() + "\n")

    print(f"\n✅ نتایج {dataset_name} ذخیره شد در: {results_dir}/")
    print(f"   - {excel_path}")
    print(f"   - {csv_path}")
    print(f"   - {summary_path}")
    print(f"   - {report_path}")


# ============================================================
# 6) نگاشت برچسب‌ها به فضای مشترک
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
# 7) ارزیابی + گزارش تصویری
# ============================================================
def evaluate_and_plot(df, dataset_name, results_dir="results_rule_based"):
    if 'true_label' not in df.columns:
        print(f"⚠️ {dataset_name}: ستون true_label موجود نیست — ارزیابی رد شد.")
        return None

    df_eval = df.copy()
    df_eval['true_std'] = normalize_labels(df_eval['true_label'])
    df_eval['pred_std'] = normalize_predicted(df_eval['predicted_label'])

    mask = df_eval['true_std'].notna() & df_eval['pred_std'].notna()
    n_total = len(df_eval)
    n_valid = int(mask.sum())

    if n_valid == 0:
        print(f"⚠️ {dataset_name}: هیچ رکورد معتبری برای ارزیابی پیدا نشد.")
        return None

    print(f"\n{'='*60}\nارزیابی {dataset_name}\n{'='*60}")
    print(f"کل رکوردها: {n_total}  |  قابل ارزیابی: {n_valid}  ({n_valid/n_total*100:.1f}%)")

    y_true = df_eval.loc[mask, 'true_std']
    y_pred = df_eval.loc[mask, 'pred_std']

    labels_order = ['Bot', 'Human', 'News Agent', 'Unverified']
    present_labels = [l for l in labels_order if l in set(y_true) | set(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, zero_division=0
    )
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() > 0 else 0.0

    print(f"\nAccuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print("\nگزارش تفصیلی:")
    print(classification_report(y_true, y_pred, labels=present_labels, zero_division=0))

    out_dir = os.path.join(results_dir, f"eval_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_labels, yticklabels=present_labels, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix — {dataset_name}\nAccuracy = {acc*100:.2f}%')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f'{dataset_name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ ذخیره شد: {cm_path}")

    # --- Precision/Recall/F1 per class ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(present_labels))
    width = 0.25
    ax.bar(x - width, precision, width, label='Precision', color='#4C72B0')
    ax.bar(x,         recall,    width, label='Recall',    color='#55A868')
    ax.bar(x + width, f1,        width, label='F1',        color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(present_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(f'Per-class Metrics — {dataset_name}  (Macro F1 = {macro_f1:.2f})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    metrics_path = os.path.join(out_dir, f'{dataset_name}_per_class_metrics.png')
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    print(f"✅ ذخیره شد: {metrics_path}")

    # --- Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    true_counts = y_true.value_counts().reindex(present_labels, fill_value=0)
    pred_counts = y_pred.value_counts().reindex(present_labels, fill_value=0)

    sns.barplot(x=true_counts.index, y=true_counts.values, ax=axes[0], palette='Blues_d')
    axes[0].set_title('True Label Distribution')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(true_counts.values):
        axes[0].text(i, v, str(v), ha='center', va='bottom')

    sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=axes[1], palette='Greens_d')
    axes[1].set_title('Predicted Label Distribution')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(pred_counts.values):
        axes[1].text(i, v, str(v), ha='center', va='bottom')

    fig.suptitle(f'Label Distribution — {dataset_name}', fontsize=13)
    plt.tight_layout()
    dist_path = os.path.join(out_dir, f'{dataset_name}_distribution.png')
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"✅ ذخیره شد: {dist_path}")

    # --- Accuracy bar ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(['Accuracy'], [acc], color='#4C72B0')
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Score')
    ax.set_title(f'Overall Accuracy — {dataset_name}')
    ax.text(acc + 0.01, 0, f'{acc*100:.2f}%', va='center')
    plt.tight_layout()
    acc_path = os.path.join(out_dir, f'{dataset_name}_accuracy.png')
    plt.savefig(acc_path, dpi=150)
    plt.close()

    # --- Eval report (txt) ---
    report_txt = os.path.join(out_dir, f'{dataset_name}_eval_report.txt')
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write(f"گزارش ارزیابی — {dataset_name}\n")
        f.write(f"تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"کل رکوردها: {n_total}  |  قابل ارزیابی: {n_valid}\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy    : {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Macro F1    : {macro_f1:.4f}\n")
        f.write(f"Weighted F1 : {weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, labels=present_labels, zero_division=0))
    print(f"✅ ذخیره شد: {report_txt}")

    return {
        'dataset': dataset_name,
        'n_valid': n_valid,
        'accuracy': float(acc),
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
    }


# ============================================================
# 8) مقایسه دو دیتاست
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
    width = 0.25
    ax.bar(x - width, df['accuracy'],    width, label='Accuracy',    color='#4C72B0')
    ax.bar(x,         df['macro_f1'],    width, label='Macro F1',    color='#55A868')
    ax.bar(x + width, df['weighted_f1'], width, label='Weighted F1', color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(df['dataset'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Rule-Based Classifier — Dataset Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, row in df.iterrows():
        ax.text(i - width, row['accuracy']    + 0.01, f"{row['accuracy']*100:.1f}%",    ha='center', fontsize=9)
        ax.text(i,         row['macro_f1']    + 0.01, f"{row['macro_f1']*100:.1f}%",    ha='center', fontsize=9)
        ax.text(i + width, row['weighted_f1'] + 0.01, f"{row['weighted_f1']*100:.1f}%", ha='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'datasets_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ ذخیره شد: {path}")


# ============================================================
# 9) اجرای اصلی
# ============================================================
if __name__ == "__main__":
    RESULTS_DIR = "../results_rule_based"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    SEARCH_DIRS = [
        Path("."),
        Path("./data"),
        Path("./datasets"),
        Path("../data"),
        Path("../datasets"),
        Path(".."),
    ]

    def find_file(filename):
        for d in SEARCH_DIRS:
            candidate = d / filename
            if candidate.exists():
                return candidate
        for p in Path("..").rglob(filename):
            return p
        return None

    # ← اسم دقیق فایل‌ها را اینجا تنظیم کن
    dataset_files = {
        "dataset1": "../datasets/dataset1.xlsx",
        "dataset2": "../datasets/dataset2.xlsx",
    }

    all_outputs = {}
    eval_summaries = []

    for name, filename in dataset_files.items():
        path = find_file(filename)
        if path is None:
            print(f"⚠️ فایل {filename} پیدا نشد. مسیرهای جست‌وجوشده: {[str(d) for d in SEARCH_DIRS]}")
            continue
        print(f"\n📂 {name} پیدا شد در: {path.resolve()}")
        df = pd.read_excel(path)
        out = classify_dataset(df, name)
        save_results(out, name, RESULTS_DIR)

        summary = evaluate_and_plot(out, name, RESULTS_DIR)
        if summary:
            eval_summaries.append(summary)

        all_outputs[name] = out

    # --- گزارش ترکیبی ---
    if all_outputs:
        combined_path = os.path.join(RESULTS_DIR, "combined_summary.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("خلاصه ترکیبی دیتاست‌ها\n")
            f.write(f"تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for name, out in all_outputs.items():
                f.write(f"--- {name} ({len(out)} رکورد) ---\n")
                f.write(out['predicted_label'].value_counts().to_string() + "\n\n")
            if eval_summaries:
                f.write("="*50 + "\n")
                f.write("خلاصه ارزیابی:\n")
                for s in eval_summaries:
                    f.write(f"{s['dataset']}: Accuracy={s['accuracy']*100:.2f}%  "
                            f"MacroF1={s['macro_f1']:.3f}  WeightedF1={s['weighted_f1']:.3f}\n")
        print(f"\n📊 گزارش ترکیبی: {combined_path}")

    # --- مقایسه تصویری ---
    plot_comparison(eval_summaries, RESULTS_DIR)

    print("\n🎉 اجرا کامل شد.")