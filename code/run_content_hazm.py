"""
پایپ‌لاین ارزیابی مدل زبانی برای تشخیص ربات/انسان بر مبنای متن توییت‌ها
----------------------------------------------------------------------
مراحل:
1) خواندن شیت اصلی (main) و شیت توییت‌ها (tweets)
2) match کردن user_id ها با نادیده گرفتن ۵ رقم آخر
3) تجمیع و پاکسازی توییت‌های هر کاربر (با hazm)
4) ارسال به مدل زبانی برای طبقه‌بندی Bot/Human
5) محاسبه‌ی دقت در مقابل برچسب واقعی

پیش‌نیازها:
    pip install pandas openpyxl hazm anthropic scikit-learn tqdm
    export ANTHROPIC_API_KEY=...   # یا تنظیم کلید مربوط به مدل زبانی دیگر
"""

import os
import re
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from hazm import Normalizer
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import anthropic

# ============================ تنظیمات ============================
MAIN_PATH = "main_users.xlsx"      # مسیر فایل شیت اصلی (برچسب بات/انسان ~۱۹۵۰۰ کاربر)
MAIN_SHEET = 0                      # اسم یا ایندکس شیت در صورت نیاز
TWEETS_PATH = "tweets.xlsx"         # مسیر فایل شیت توییت‌ها
TWEETS_SHEET = 0

MAIN_ID_COL = "user_id"
TWEETS_ID_COL = "user_id"
TEXT_COL = "text"

# ستونی که به عنوان "برچسب واقعی" مبنا قرار می‌دهی:
# اگر برچسب دستی/واقعی دیگری داری (مثلا ستون human-labeled)، اسمش را اینجا بگذار.
LABEL_COL = "Predicted_Label"

TRUNCATE_DIGITS = 5          # تعداد رقم آخر که در match نادیده گرفته می‌شود
MAX_TWEETS_PER_USER = 30     # سقف تعداد توییت در پرامپت هر کاربر (برای سبک ماندن)
MAX_CHARS_IN_PROMPT = 6000   # سقف طول کلی متن ارسالی به مدل

MODEL_NAME = "claude-sonnet-4-6"
MAX_WORKERS = 8              # تعداد ریکوئست هم‌زمان؛ اگر rate-limit خوردی کم‌ترش کن

OUTPUT_DIR = Path("llm_bot_detection_output")
CHECKPOINT_PATH = OUTPUT_DIR / "predictions_checkpoint.csv"
FINAL_OUTPUT_PATH = OUTPUT_DIR / "llm_predictions_final.csv"
CHECKPOINT_EVERY = 50
# ===================================================================

OUTPUT_DIR.mkdir(exist_ok=True)
normalizer = Normalizer()
client = anthropic.Anthropic()  # از متغیر محیطی ANTHROPIC_API_KEY استفاده می‌کند

PROMPT_TEMPLATE = """\
در ادامه مجموعه‌ای از توییت‌های یک کاربر توییتر آمده است (ممکن است فارسی یا انگلیسی باشد).
بر اساس محتوا، سبک نوشتار، تکراری یا تبلیغاتی بودن، و لحن کلی، تشخیص بده که این اکانت به احتمال بیشتر یک ربات (Bot) است یا یک انسان واقعی (Human).

فقط دقیقاً یکی از دو کلمه‌ی زیر را به عنوان پاسخ بنویس و هیچ توضیح اضافه‌ای نده:
Bot
Human

توییت‌های کاربر:
{tweets}
"""


def truncate_id(uid, n=TRUNCATE_DIGITS) -> str:
    s = str(uid).strip()
    return s[:-n] if len(s) > n else s


def load_data():
    main_df = pd.read_excel(MAIN_PATH, sheet_name=MAIN_SHEET)
    tweets_df = pd.read_excel(TWEETS_PATH, sheet_name=TWEETS_SHEET)
    return main_df, tweets_df


def match_users(main_df: pd.DataFrame, tweets_df: pd.DataFrame):
    main_df = main_df.copy()
    tweets_df = tweets_df.copy()
    main_df["id_trunc"] = main_df[MAIN_ID_COL].apply(truncate_id)
    tweets_df["id_trunc"] = tweets_df[TWEETS_ID_COL].apply(truncate_id)

    matched_ids = set(main_df["id_trunc"]) & set(tweets_df["id_trunc"])
    main_matched = main_df[main_df["id_trunc"].isin(matched_ids)].copy()
    tweets_matched = tweets_df[tweets_df["id_trunc"].isin(matched_ids)].copy()

    print(f"[match] تعداد کاربران match‌شده: {main_matched['id_trunc'].nunique()}")
    return main_matched, tweets_matched


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = normalizer.normalize(text)
    text = re.sub(r"http\S+", "", text)       # حذف لینک
    text = re.sub(r"@\w+", "", text)          # حذف منشن
    text = re.sub(r"#", "", text)             # نگه‌داشتن متن هشتگ، حذف علامت
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_user_corpus(tweets_matched: pd.DataFrame) -> pd.DataFrame:
    df = tweets_matched.copy()
    df["clean_text"] = df[TEXT_COL].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    grouped = (
        df.groupby("id_trunc")["clean_text"]
        .apply(lambda texts: " \n".join(list(texts)[:MAX_TWEETS_PER_USER]))
        .reset_index()
        .rename(columns={"clean_text": "corpus"})
    )
    return grouped


def classify_user(corpus: str, retries: int = 3) -> str | None:
    prompt = PROMPT_TEMPLATE.format(tweets=corpus[:MAX_CHARS_IN_PROMPT])
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL_NAME,
                max_tokens=5,
                messages=[{"role": "user", "content": prompt}],
            )
            label = resp.content[0].text.strip().lower()
            return "Bot" if "bot" in label else "Human"
        except Exception as e:
            wait = 2 * (attempt + 1)
            print(f"[warn] خطا در فراخوانی مدل: {e} — تلاش دوباره بعد از {wait} ثانیه")
            time.sleep(wait)
    return None


def normalize_label(x) -> str:
    s = str(x).strip().lower()
    return "Bot" if "bot" in s else "Human"


def run_pipeline():
    main_df, tweets_df = load_data()
    main_matched, tweets_matched = match_users(main_df, tweets_df)
    user_corpus = build_user_corpus(tweets_matched)

    labels = main_matched[["id_trunc", LABEL_COL]].drop_duplicates("id_trunc")
    merged = user_corpus.merge(labels, on="id_trunc", how="inner")
    print(f"[info] تعداد کاربران نهایی برای ارزیابی: {len(merged)}")

    # ادامه از checkpoint قبلی در صورت وجود
    done_ids = set()
    if CHECKPOINT_PATH.exists():
        prev = pd.read_csv(CHECKPOINT_PATH)
        done_ids = set(prev["id_trunc"].astype(str))
        print(f"[resume] {len(done_ids)} کاربر قبلاً پردازش شده‌اند، از آن‌ها رد می‌شویم")
    else:
        prev = pd.DataFrame(columns=["id_trunc", "llm_prediction"])

    todo = merged[~merged["id_trunc"].astype(str).isin(done_ids)]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(classify_user, row["corpus"]): row["id_trunc"]
            for _, row in todo.iterrows()
        }
        for i, future in enumerate(tqdm(as_completed(future_to_id), total=len(future_to_id))):
            uid = future_to_id[future]
            pred = future.result()
            results.append({"id_trunc": uid, "llm_prediction": pred})

            if (i + 1) % CHECKPOINT_EVERY == 0:
                checkpoint_df = pd.concat([prev, pd.DataFrame(results)], ignore_index=True)
                checkpoint_df.to_csv(CHECKPOINT_PATH, index=False, encoding="utf-8-sig")

    all_preds = pd.concat([prev, pd.DataFrame(results)], ignore_index=True)
    all_preds.to_csv(CHECKPOINT_PATH, index=False, encoding="utf-8-sig")

    final = merged.merge(all_preds, on="id_trunc", how="left")
    final.to_csv(FINAL_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    valid = final.dropna(subset=["llm_prediction"])
    y_true = valid[LABEL_COL].apply(normalize_label)
    y_pred = valid["llm_prediction"]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== نتیجه نهایی ===")
    print(f"تعداد کاربران ارزیابی‌شده: {len(valid)}")
    print(f"دقت (accuracy) مدل زبانی: {acc:.4f}\n")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    run_pipeline()
