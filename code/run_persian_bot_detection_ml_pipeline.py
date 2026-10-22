"""
پایپ‌لاین تشخیص ربات/انسان در توییتر فارسی — مبتنی بر ویژگی‌های متنی مهندسی‌شده
=================================================================================
این نسخه کاملاً جایگزین نسخه‌ی قبلی (که از Claude برای طبقه‌بندی مستقیم استفاده
می‌کرد) شده است. در این‌جا هیچ مدل زبانی/تولیدی و هیچ embedding معنایی از پیش
آموزش‌دیده استفاده نمی‌شود؛ تمام پیش‌بینی‌ها صرفاً حاصل یک مدل یادگیری ماشین
سنتی (RandomForest، با LogisticRegression به‌عنوان baseline) است که روی
ویژگی‌های آماری/لغوی/زبانی فارسیِ استخراج‌شده از متن توییت‌ها آموزش می‌بیند.

خانواده‌های ویژگی پیاده‌سازی‌شده:
    1) آمار پایه‌ی متن (تعداد کلمه، طول توییت، ...)
    2) تنوع واژگانی و تکرار (lexical diversity، hapax، bigram تکراری، ...)
    3) ویژگی‌های نوشتاری فارسی (نسبت حروف فارسی/عربی/لاتین، نیم‌فاصله،
       نشانه‌گذاری، کشیدگی حروف، ایموجی)
    4) ویژگی‌های صرفی فارسی (جمع، می‌/نمی‌، تر/ترین/مند/ناک/دار)
    5) ویژگی‌های نحوی/POS (در صورت در دسترس بودن POS Tagger در hazm)
    6) ویژگی‌های محاوره‌ای/عامیانه (بر اساس یک لغت‌نامه‌ی کوچک و قابل‌توسعه)

پروتکل ارزیابی:
    - یک تقسیم train/test لایه‌بندی‌شده (stratified) برای گزارش نهایی
    - اعتبارسنجی متقاطع لایه‌بندی‌شده (Stratified K-Fold) روی داده‌ی train
      برای گزارش میانگین/انحراف‌معیار معیارها و پیش‌بینی‌های out-of-fold
    - تمام پیش‌پردازش (imputation/scaling) داخل یک sklearn Pipeline انجام
      می‌شود تا نشتِ داده (data leakage) از فولدهای تست رخ ندهد.

پیش‌نیازها:
    pip install pandas openpyxl hazm scikit-learn matplotlib numpy

نکته درباره‌ی POS Tagger:
    hazm برای POS tagging به یک فایل مدل جداگانه نیاز دارد (معمولاً
    pos_tagger.model) که باید جدا دانلود/دانلود و مسیرش در HAZM_POS_MODEL_PATH
    تنظیم شود. اگر مدل در دسترس نباشد، این اسکریپت ویژگی‌های POS را به‌جای
    حدس‌زدن، صراحتاً NaN می‌گذارد و یک پیام هشدار چاپ می‌کند.
"""

from __future__ import annotations
import matplotlib.pyplot as plt

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hazm import Normalizer, word_tokenize

try:
    from hazm import POSTagger  # noqa: F401
    _HAZM_HAS_POSTAGGER_CLASS = True
except ImportError:
    POSTagger = None  # type: ignore
    _HAZM_HAS_POSTAGGER_CLASS = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

import matplotlib
matplotlib.use("Agg")

# ============================== تنظیمات کلی ==============================

# مسیر فایل شیت اصلی (برچسب بات/انسان ~۱۹۵۰۰ کاربر)
MAIN_PATH = "../datasets/dataset1.xlsx"
# MAIN_PATH = "../datasets/dataset2.xlsx"

# اطلاعات کاربران
MAIN_SHEET = "1k_users"
# MAIN_SHEET = "main"

# اطلاعات توییت‌ها
TWEETS_SHEETS = ["tweets_meta_data"]
# TWEETS_SHEETS = [
#     "tweetsMetaData_1",
#     "tweetsMetaData_2",
# ]

MAIN_ID_COL = "id"
# MAIN_ID_COL = "user_id"
TWEETS_ID_COL = "id"
TEXT_COL = "text"

# ستونی که به‌عنوان «برچسب مرجع» در نظر می‌گیریم.
# توجه: اگر این ستون خروجی یک مدل دیگر (مثل Botometer) باشد، نه برچسب واقعی
# دست‌ساز/annotate‌شده، نتیجه‌ی ارزیابی فقط «میزان توافق» با آن مدل است، نه
# لزوماً دقت واقعی تشخیص ربات. این موضوع در خروجی هم صراحتاً چاپ می‌شود.
LABEL_COL = "Predicted_Label"

TRUNCATE_DIGITS = 5          # تعداد رقم آخر که در match نادیده گرفته می‌شود
MAX_TWEETS_PER_USER = 30     # سقف تعداد توییت در نظر گرفته‌شده برای هر کاربر
# اگر truncate باعث تناقض برچسب شد، آن گروه کنار گذاشته شود
DROP_AMBIGUOUS_LABEL_COLLISIONS = True

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
# حداقل نمونه در هر کلاس برای split لایه‌بندی‌شده
MIN_SAMPLES_PER_CLASS_FOR_STRATIFY = 2

# مسیر فایل مدل POS Tagger هازم (در صورت وجود). اگر نداری، None بگذار یا مسیر
# اشتباه بگذار؛ اسکریپت خودش fallback می‌کند.
HAZM_POS_MODEL_PATH: Optional[str] = "pos_tagger.model"

OUTPUT_DIR = Path("../results_ml_bot_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = OUTPUT_DIR / "user_text_features.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "ml_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "evaluation_metrics.json"
CLASSIFICATION_REPORT_TXT = OUTPUT_DIR / "classification_report.txt"
CONFUSION_MATRIX_PNG = OUTPUT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_CSV = OUTPUT_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PNG = OUTPUT_DIR / "feature_importance.png"
METRICS_CSV = OUTPUT_DIR / "metrics_summary.csv"
ROC_CURVE_PNG = OUTPUT_DIR / "roc_curve_random_forest.png"
PR_CURVE_PNG = OUTPUT_DIR / "precision_recall_curve_random_forest.png"
METRICS_COMPARISON_PNG = OUTPUT_DIR / "model_metrics_comparison.png"
CLASS_DISTRIBUTION_PNG = OUTPUT_DIR / "class_distribution.png"
CV_METRICS_PNG = OUTPUT_DIR / "cross_validation_metrics.png"

POSITIVE_LABEL = "Bot"   # کلاس مثبت برای ROC-AUC / PR-AUC و گزارش‌ها


# ============================== ابزارهای متنی ==============================

normalizer = Normalizer()

URL_RE = re.compile(r"http\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SIGN_RE = re.compile(r"#")

PERSIAN_ARABIC_LETTERS_RE = re.compile(
    r"[\u0621-\u063A\u0641-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06CC\u06CB\u0670\u0629]"
)
LATIN_LETTERS_RE = re.compile(r"[A-Za-z]")
PERSIAN_DIGITS_RE = re.compile(r"[\u06F0-\u06F9]")
ARABIC_INDIC_DIGITS_RE = re.compile(r"[\u0660-\u0669]")
LATIN_DIGITS_RE = re.compile(r"[0-9]")
# حروف عربی‌ای که نرمال‌سازی hazm آن‌ها را عوض می‌کند (ي -> ی ، ك -> ک)
RAW_ARABIC_SPECIFIC_RE = re.compile(r"[\u064A\u0643]")
ZWNJ = "\u200c"
PUNCTUATION_RE = re.compile(r"[،؛؟.,!?;:\"'\(\)\[\]{}«»…\-]")
EXCLAMATION_RE = re.compile(r"!")
QUESTION_MARK_RE = re.compile(r"[؟?]")
# ۳ بار یا بیشتر تکرار یک کاراکتر پشت‌سرهم
ELONGATION_RE = re.compile(r"(.)\1{2,}")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF\U00002B00-\U00002BFF]"
)

# لغت‌نامه‌ی نمونه (قابل‌توسعه) از صورت‌های محاوره‌ای رایج فارسی و معادل رسمی آن‌ها.
# توجه: این لغت‌نامه کامل نیست و تشخیص بر مبنای آن می‌تواند به لهجه/گویش وابسته
# باشد؛ صرفاً یک سیگنال کمکی است، نه یک قاعده‌ی قطعی.
COLLOQUIAL_TO_FORMAL = {
    "واسه": "برای",
    "آخه": "آخر",
    "میخوام": "می‌خواهم",
    "می‌خوام": "می‌خواهم",
    "نمیدونم": "نمی‌دانم",
    "نمی‌دونم": "نمی‌دانم",
    "چیه": "چی است",
    "اونا": "آنها",
    "اینا": "اینها",
    "میگم": "می‌گویم",
    "میگه": "می‌گوید",
    "نمیشه": "نمی‌شود",
    "میشه": "می‌شود",
    "دیگه": "دیگر",
    "ولی": "اما",  # نمونه‌ی مرزی: در گفتار رایج هم به کار می‌رود، صرفاً به‌عنوان نمونه
}
FORMAL_FORMS = set(COLLOQUIAL_TO_FORMAL.values())
COLLOQUIAL_FORMS = set(COLLOQUIAL_TO_FORMAL.keys())

# الگوهای صرفی فارسی (محافظه‌کارانه، بر مبنای token کامل نه substring دلخواه)
PLURAL_SUFFIXES = ("ها", "های", "ان")
VERB_PREFIXES = ("می‌", "نمی‌", "می", "نمی")
DERIVATIONAL_SUFFIXES = ("ترین", "تر", "مند", "ناک", "دار")
EZAFE_MARK_RE = re.compile(r".+" + ZWNJ + r"ی$")  # مثل «خانه‌ی»


def clean_text_basic(raw_text) -> str:
    """حذف لینک، منشن و علامت هشتگ (قبل از هر پردازش دیگری). متن اصلی را تا حد
    امکان دست‌نخورده نگه می‌دارد تا ویژگی‌های کاراکتریِ وابسته به نگارش اصلی
    (حروف عربی خام، ایموجی، کشیدگی) از بین نرود."""
    if not isinstance(raw_text, str):
        return ""
    text = URL_RE.sub("", raw_text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_SIGN_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(raw_clean_text: str) -> str:
    """نرمال‌سازی hazm (یکدست‌سازی ي/ك، نیم‌فاصله‌ها، اعداد و ...). این نسخه
    برای توکنایز کردن و ویژگی‌های مبتنی بر توکن استفاده می‌شود."""
    if not raw_clean_text:
        return ""
    return normalizer.normalize(raw_clean_text)


def tokenize_tweets(normalized_tweets: list[str]) -> list[list[str]]:
    """توکنایز هر توییت به‌صورت جداگانه تا مرز توییت‌ها حفظ شود (برای آمار
    سطح‌توییت مثل median_words_per_tweet لازم است)."""
    token_lists = []
    for t in normalized_tweets:
        if not t:
            token_lists.append([])
            continue
        try:
            toks = word_tokenize(t)
        except Exception:
            toks = t.split()
        token_lists.append([tok for tok in toks if tok.strip()])
    return token_lists


# ============================== استخراج ویژگی‌ها ==============================

def extract_statistical_features(token_lists: list[list[str]], tweet_char_lens: list[int]) -> dict:
    """ویژگی‌های آماری پایه + تنوع/تکرار واژگانی. token_lists مرز هر توییت را
    حفظ می‌کند؛ بسیاری از آمارها (median_words_per_tweet، short/long_tweet_ratio)
    بدون این مرزبندی قابل‌محاسبه نیستند."""
    feats: dict = {}

    tweet_count = sum(1 for toks in token_lists if len(toks) > 0)
    words_per_tweet = [len(toks) for toks in token_lists if len(toks) > 0]
    all_tokens = [tok for toks in token_lists for tok in toks]
    total_word_count = len(all_tokens)
    unique_tokens = set(all_tokens)
    unique_word_count = len(unique_tokens)

    feats["tweet_count"] = tweet_count
    feats["total_word_count"] = total_word_count
    feats["unique_word_count"] = unique_word_count
    feats["avg_words_per_tweet"] = float(
        np.mean(words_per_tweet)) if words_per_tweet else 0.0
    feats["avg_word_length"] = (
        float(np.mean([len(tok) for tok in all_tokens])) if all_tokens else 0.0
    )
    feats["avg_chars_per_tweet"] = float(
        np.mean(tweet_char_lens)) if tweet_char_lens else 0.0
    feats["median_words_per_tweet"] = float(
        np.median(words_per_tweet)) if words_per_tweet else 0.0
    feats["short_tweet_ratio"] = (
        sum(1 for n in words_per_tweet if n < 5) /
        len(words_per_tweet) if words_per_tweet else 0.0
    )
    feats["long_tweet_ratio"] = (
        sum(1 for n in words_per_tweet if n > 30) /
        len(words_per_tweet) if words_per_tweet else 0.0
    )

    # --- تنوع واژگانی و تکرار ---
    token_freq = Counter(all_tokens)
    lexical_diversity = unique_word_count / \
        total_word_count if total_word_count else 0.0
    feats["lexical_diversity"] = lexical_diversity
    # type_token_ratio از لحاظ ریاضی معادل lexical_diversity است (هر دو
    # unique/total هستند)؛ این‌جا فقط برای سازگاری با نام‌گذاری رایج در ادبیات
    # نگه داشته شده و منبع اطلاعاتی مستقلی نیست.
    feats["type_token_ratio"] = lexical_diversity

    hapax_count = sum(1 for tok, c in token_freq.items() if c == 1)
    feats["hapax_ratio"] = hapax_count / \
        unique_word_count if unique_word_count else 0.0
    # repeated_token_ratio: سهم occurrence هایی که به توکن‌های تکرارشونده (freq>1)
    # تعلق دارند، از کل توکن‌ها: (total - hapax_occurrences) / total
    feats["repeated_token_ratio"] = (
        (total_word_count - hapax_count) /
        total_word_count if total_word_count else 0.0
    )
    max_freq = max(token_freq.values()) if token_freq else 0
    feats["max_token_frequency_ratio"] = max_freq / \
        total_word_count if total_word_count else 0.0
    top10_sum = sum(c for _, c in token_freq.most_common(10))
    feats["top_10_token_ratio"] = top10_sum / \
        total_word_count if total_word_count else 0.0

    # repeated_bigram_ratio: سهم occurrenceهای bigram که تکراری‌اند (bigram‌هایی
    # که بیش از یک بار دیده شده‌اند)، از کل occurrenceهای bigram.
    bigrams = [
        (all_tokens[i], all_tokens[i + 1]) for i in range(len(all_tokens) - 1)
    ]
    if bigrams:
        bigram_freq = Counter(bigrams)
        total_bigrams = len(bigrams)
        repeated_bigram_occurrences = sum(
            c for c in bigram_freq.values() if c > 1)
        feats["repeated_bigram_ratio"] = repeated_bigram_occurrences / total_bigrams
    else:
        feats["repeated_bigram_ratio"] = 0.0

    return feats


def shannon_char_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    probs = [c / total for c in freq.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))


def extract_persian_linguistic_features(
    raw_clean_concat: str,
    normalized_concat: str,
    all_tokens: list[str],
) -> dict:
    """ویژگی‌های نوشتاری فارسی (بر مبنای متن خام قبل از نرمال‌سازی، در مواردی
    که نرمال‌سازی شواهد را از بین می‌برد) + ویژگی‌های صرفی (بر مبنای توکن‌های
    نرمال‌شده)."""
    feats: dict = {}

    n_chars_raw = len(raw_clean_concat)
    n_chars_norm = len(normalized_concat) if normalized_concat else 0
    total_word_count = len(all_tokens)

    persian_letters = len(PERSIAN_ARABIC_LETTERS_RE.findall(raw_clean_concat))
    latin_letters = len(LATIN_LETTERS_RE.findall(raw_clean_concat))
    alpha_total = persian_letters + latin_letters
    feats["persian_char_ratio"] = persian_letters / \
        alpha_total if alpha_total else 0.0
    feats["latin_char_ratio"] = latin_letters / \
        alpha_total if alpha_total else 0.0

    # arabic_char_ratio: حروف عربی‌ای که نرمال‌سازی آن‌ها را تغییر می‌دهد (ي، ك)
    # باید روی متن خام (پیش از نرمال‌سازی) حساب شود.
    raw_arabic_specific = len(RAW_ARABIC_SPECIFIC_RE.findall(raw_clean_concat))
    feats["arabic_char_ratio"] = raw_arabic_specific / \
        alpha_total if alpha_total else 0.0

    persian_digits = len(PERSIAN_DIGITS_RE.findall(raw_clean_concat)) + len(
        ARABIC_INDIC_DIGITS_RE.findall(raw_clean_concat)
    )
    latin_digits = len(LATIN_DIGITS_RE.findall(raw_clean_concat))
    digit_total = persian_digits + latin_digits
    feats["persian_digit_ratio"] = persian_digits / \
        digit_total if digit_total else 0.0

    half_space_count = raw_clean_concat.count(ZWNJ)
    feats["half_space_count"] = half_space_count
    feats["half_space_ratio"] = half_space_count / max(total_word_count, 1)

    feats["punctuation_ratio"] = (
        len(PUNCTUATION_RE.findall(raw_clean_concat)) /
        n_chars_raw if n_chars_raw else 0.0
    )
    feats["exclamation_ratio"] = (
        len(EXCLAMATION_RE.findall(raw_clean_concat)) /
        n_chars_raw if n_chars_raw else 0.0
    )
    feats["question_mark_ratio"] = (
        len(QUESTION_MARK_RE.findall(raw_clean_concat)) /
        n_chars_raw if n_chars_raw else 0.0
    )

    elong_matches = ELONGATION_RE.findall(raw_clean_concat)
    # findall با گروه برمی‌گرداند؛ طول واقعی هر match را با finditer می‌گیریم
    elong_char_count = sum(len(m.group(0))
                           for m in ELONGATION_RE.finditer(raw_clean_concat))
    feats["elongated_char_ratio"] = elong_char_count / \
        n_chars_raw if n_chars_raw else 0.0

    emoji_matches = EMOJI_RE.findall(raw_clean_concat)
    feats["emoji_count"] = len(emoji_matches)
    feats["emoji_ratio"] = len(emoji_matches) / \
        n_chars_raw if n_chars_raw else 0.0

    feats["unique_char_ratio"] = (
        len(set(normalized_concat)) / n_chars_norm if n_chars_norm else 0.0
    )
    feats["char_entropy"] = shannon_char_entropy(normalized_concat)

    # --- ویژگی‌های صرفی (بر مبنای توکن‌های نرمال‌شده) ---
    n_tok = total_word_count if total_word_count else 1

    plural_count = sum(
        1 for tok in all_tokens if len(tok) > 2 and any(tok.endswith(sfx) for sfx in PLURAL_SUFFIXES)
    )
    feats["plural_suffix_freq_per_100tok"] = 100.0 * plural_count / n_tok

    ezafe_count = sum(1 for tok in all_tokens if EZAFE_MARK_RE.match(tok))
    feats["ezafe_marker_freq_per_100tok"] = 100.0 * ezafe_count / n_tok

    verb_prefix_count = sum(
        1 for tok in all_tokens if any(tok.startswith(pfx) for pfx in VERB_PREFIXES)
    )
    feats["verb_prefix_freq_per_100tok"] = 100.0 * verb_prefix_count / n_tok

    derivational_count = sum(
        1
        for tok in all_tokens
        if len(tok) > 3 and any(tok.endswith(sfx) for sfx in DERIVATIONAL_SUFFIXES)
    )
    feats["derivational_suffix_freq_per_100tok"] = 100.0 * \
        derivational_count / n_tok

    # --- ویژگی‌های محاوره‌ای ---
    colloquial_count = sum(1 for tok in all_tokens if tok in COLLOQUIAL_FORMS)
    formal_count = sum(1 for tok in all_tokens if tok in FORMAL_FORMS)
    feats["colloquial_token_count"] = colloquial_count
    feats["colloquial_token_ratio"] = colloquial_count / n_tok
    feats["formal_token_ratio"] = formal_count / n_tok
    distinct_colloquial_found = len(
        {tok for tok in all_tokens if tok in COLLOQUIAL_FORMS})
    feats["slang_lexicon_coverage"] = (
        distinct_colloquial_found /
        len(COLLOQUIAL_FORMS) if COLLOQUIAL_FORMS else 0.0
    )

    return feats


# ------------------------------ POS tagging (اختیاری) ------------------------------

_pos_tagger = None
_pos_tagger_checked = False
_POS_WARNING_SHOWN = False

# نگاشت برچسب‌های احتمالی hazm (هم نسخه‌ی قدیمی بر مبنای Bijankhan، هم نسخه‌ی
# جدیدتر بر مبنای Universal Dependencies) به دسته‌های موردنظر ما. اگر مدل نصب‌شده
# برچسب‌های دیگری تولید کند، باید این نگاشت به‌روزرسانی شود.
POS_TAG_MAP = {
    # Universal Dependencies style
    "NOUN": "noun", "PROPN": "noun",
    "VERB": "verb", "AUX": "verb",
    "ADJ": "adjective",
    "ADV": "adverb",
    "PRON": "pronoun",
    "ADP": "preposition",
    "CCONJ": "conjunction", "SCONJ": "conjunction",
    "NUM": "numeral",
    # Bijankhan / legacy hazm style
    "N": "noun", "Ne": "noun",
    "V": "verb",
    "AJ": "adjective", "AJe": "adjective",
    "ADV": "adverb",
    "PRO": "pronoun",
    "P": "preposition",
    "CONJ": "conjunction",
    "NUM": "numeral", "NUMe": "numeral",
}
FUNCTION_WORD_CATEGORIES = {"preposition", "conjunction", "pronoun"}

FIRST_PERSON_PRONOUNS = {"من", "ما", "مرا", "مان", "م"}
SECOND_PERSON_PRONOUNS = {"تو", "شما", "ترا", "تان", "ت"}
THIRD_PERSON_PRONOUNS = {"او", "وی", "آنها", "آن‌ها", "ایشان", "شان", "ش"}


def _try_load_pos_tagger():
    global _pos_tagger, _pos_tagger_checked, _POS_WARNING_SHOWN
    if _pos_tagger_checked:
        return _pos_tagger
    _pos_tagger_checked = True
    if not _HAZM_HAS_POSTAGGER_CLASS or not HAZM_POS_MODEL_PATH:
        if not _POS_WARNING_SHOWN:
            print(
                "[هشدار] کلاس POSTagger در hazm در دسترس نیست یا مسیر مدل تنظیم "
                "نشده. ویژگی‌های POS-based برای همه‌ی کاربران NaN خواهند بود. "
                "برای فعال کردن، مدل POS Tagger هازم را دانلود کن و مسیرش را در "
                "HAZM_POS_MODEL_PATH بگذار."
            )
            _POS_WARNING_SHOWN = True
        return None
    try:
        tagger = POSTagger(model=HAZM_POS_MODEL_PATH)
        _pos_tagger = tagger
        return tagger
    except Exception as e:
        if not _POS_WARNING_SHOWN:
            print(
                f"[هشدار] بارگذاری POS Tagger هازم ناموفق بود ({e}). ویژگی‌های "
                "POS-based برای همه‌ی کاربران NaN خواهند بود."
            )
            _POS_WARNING_SHOWN = True
        return None


def extract_pos_features(token_lists: list[list[str]]) -> dict:
    """ویژگی‌های نحوی/POS. اگر POS tagger در دسترس نباشد، همه‌ی مقادیر NaN
    برمی‌گردند (به‌جای حدس زدن یا جعل برچسب)."""
    pos_feature_names = [
        "noun_ratio", "verb_ratio", "adjective_ratio", "adverb_ratio",
        "pronoun_ratio", "preposition_ratio", "conjunction_ratio", "numeral_ratio",
        "function_word_ratio", "first_person_pronoun_ratio",
        "second_person_pronoun_ratio", "third_person_pronoun_ratio",
    ]

    tagger = _try_load_pos_tagger()
    if tagger is None:
        return {name: np.nan for name in pos_feature_names}

    all_tagged = []
    try:
        for toks in token_lists:
            if not toks:
                continue
            all_tagged.extend(tagger.tag(toks))
    except Exception as e:
        warnings.warn(f"خطا در اجرای POS tagger: {e}")
        return {name: np.nan for name in pos_feature_names}

    n_tagged = len(all_tagged)
    if n_tagged == 0:
        return {name: 0.0 for name in pos_feature_names}

    category_counts = Counter()
    pronoun_tokens = []
    for tok, tag in all_tagged:
        category = POS_TAG_MAP.get(tag)
        if category:
            category_counts[category] += 1
        if category == "pronoun":
            pronoun_tokens.append(tok)

    feats = {}
    for cat in ["noun", "verb", "adjective", "adverb", "pronoun", "preposition", "conjunction", "numeral"]:
        feats[f"{cat}_ratio"] = category_counts.get(cat, 0) / n_tagged

    function_word_count = sum(category_counts.get(c, 0)
                              for c in FUNCTION_WORD_CATEGORIES)
    feats["function_word_ratio"] = function_word_count / n_tagged

    feats["first_person_pronoun_ratio"] = (
        sum(1 for t in pronoun_tokens if t in FIRST_PERSON_PRONOUNS) / n_tagged
    )
    feats["second_person_pronoun_ratio"] = (
        sum(1 for t in pronoun_tokens if t in SECOND_PERSON_PRONOUNS) / n_tagged
    )
    feats["third_person_pronoun_ratio"] = (
        sum(1 for t in pronoun_tokens if t in THIRD_PERSON_PRONOUNS) / n_tagged
    )
    return feats


# ------------------------------ تجمیع در سطح کاربر ------------------------------

def extract_user_features(user_tweets_raw: list[str]) -> dict:
    """تمام ویژگی‌های یک کاربر را از لیست توییت‌های خامش می‌سازد. مرز توییت‌ها
    تا مرحله‌ی توکنایز حفظ می‌شود تا آمار سطح‌توییت درست محاسبه شود."""
    raw_clean_tweets = [clean_text_basic(t) for t in user_tweets_raw]
    # حذف توییت‌های خالی بعد از پاکسازی
    raw_clean_tweets = [t for t in raw_clean_tweets if t]

    normalized_tweets = [normalize_text(t) for t in raw_clean_tweets]
    token_lists = tokenize_tweets(normalized_tweets)
    all_tokens = [tok for toks in token_lists for tok in toks]

    tweet_char_lens = [len(t) for t in normalized_tweets]
    raw_clean_concat = " ".join(raw_clean_tweets)
    normalized_concat = " ".join(normalized_tweets)

    feats = {}
    feats.update(extract_statistical_features(token_lists, tweet_char_lens))
    feats.update(
        extract_persian_linguistic_features(
            raw_clean_concat, normalized_concat, all_tokens)
    )
    feats.update(extract_pos_features(token_lists))
    return feats


def build_feature_dataset(tweets_matched: pd.DataFrame) -> pd.DataFrame:
    """یک ردیف به‌ازای هر کاربر (id_trunc) با تمام ویژگی‌های استخراج‌شده."""
    rows = []
    grouped = tweets_matched.groupby("id_trunc")[TEXT_COL].apply(list)
    for user_id, tweet_texts in grouped.items():
        tweet_texts = tweet_texts[:MAX_TWEETS_PER_USER]
        feats = extract_user_features(list(tweet_texts))
        feats["id_trunc"] = user_id
        feats["n_raw_tweets_considered"] = len(tweet_texts)
        rows.append(feats)
    df = pd.DataFrame(rows)
    cols = ["id_trunc", "n_raw_tweets_considered"] + [
        c for c in df.columns if c not in ("id_trunc", "n_raw_tweets_considered")
    ]
    return df[cols]


# ============================== تطبیق و برچسب‌ها ==============================

def truncate_id(value):
    """Remove the last 5 digits of an ID for matching."""
    if pd.isna(value):
        return None

    s = str(value).strip()

    # Handle IDs stored as floats or scientific notation
    try:
        from decimal import Decimal
        number = Decimal(s)
        if number == number.to_integral_value():
            s = str(int(number))
    except Exception:
        pass

    if not s.isdigit() or len(s) <= 5:
        return None

    return s[:-5]


def load_data():
    """Load user data and combine tweets from both metadata sheets."""

    print("[INFO] Loading user data...")

    main_df = pd.read_excel(
        MAIN_PATH,
        sheet_name=MAIN_SHEET
    )

    print(f"[INFO] Users loaded: {len(main_df):,}")
    print(f"[INFO] Loading tweet metadata sheets: {TWEETS_SHEETS}")

    tweet_dfs = []

    for sheet_name in TWEETS_SHEETS:
        df = pd.read_excel(
            MAIN_PATH,
            sheet_name=sheet_name
        )

        print(
            f"[INFO] {sheet_name}: "
            f"{len(df):,} rows"
        )

        # بررسی ستون‌های ضروری
        required_columns = [TWEETS_ID_COL, TEXT_COL]
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in sheet '{sheet_name}': "
                f"{missing_columns}. "
                f"Available columns: {df.columns.tolist()}"
            )

        tweet_dfs.append(df)

    # ادغام اطلاعات توییت‌ها از هر دو شیت
    tweets_df = pd.concat(
        tweet_dfs,
        ignore_index=True
    )

    print(
        f"[INFO] Total tweets loaded: "
        f"{len(tweets_df):,}"
    )

    # بررسی ستون‌های ضروری شیت کاربران
    required_main_columns = [MAIN_ID_COL, LABEL_COL]
    missing_main_columns = [
        col for col in required_main_columns
        if col not in main_df.columns
    ]

    if missing_main_columns:
        raise ValueError(
            f"Missing columns in sheet '{MAIN_SHEET}': "
            f"{missing_main_columns}. "
            f"Available columns: {main_df.columns.tolist()}"
        )

    return main_df, tweets_df


def normalize_label(x) -> Optional[str]:
    """نرمال‌سازی مقدار برچسب به دقیقاً دو کلاس Bot/Human. مقادیر ناشناخته یا
    خالی را به Human map نمی‌کند؛ None برمی‌گرداند تا بعداً حذف شوند."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"bot", "b", "1", "fake", "bot account"}:
        return "Bot"
    if s in {"human", "h", "0", "genuine", "real", "real account"}:
        return "Human"
    return None  # مقدار ناشناخته -> نامعتبر، نه به‌طور خودکار Human


def resolve_label_collisions(main_df: pd.DataFrame) -> pd.DataFrame:
    """بررسی می‌کند که آیا truncate کردن آیدی باعث شده چند کاربر متفاوت (با
    برچسب‌های متفاوت) زیر یک id_trunc یکسان جمع شوند. گروه‌های متناقض
    (ambiguous) گزارش و (در صورت فعال بودن DROP_AMBIGUOUS_LABEL_COLLISIONS)
    از داده‌ی قابل استفاده برای آموزش/ارزیابی حذف می‌شوند."""
    main_df = main_df.copy()
    main_df["_norm_label"] = main_df[LABEL_COL].apply(normalize_label)

    group_sizes = main_df.groupby("id_trunc")[MAIN_ID_COL].nunique()
    collided_groups = group_sizes[group_sizes > 1].index
    n_collisions = len(collided_groups)
    print(
        f"[collision-check] تعداد id_trunc با بیش از یک user_id اصلی متفاوت: {n_collisions}")

    if n_collisions > 0:
        label_nunique = (
            main_df[main_df["id_trunc"].isin(collided_groups)]
            .groupby("id_trunc")["_norm_label"]
            .nunique()
        )
        ambiguous_groups = label_nunique[label_nunique > 1].index
        print(
            f"[collision-check] از این‌ها، {len(ambiguous_groups)} گروه برچسب "
            f"متناقض دارند (یعنی کاربران جمع‌شده زیر یک id_trunc، برچسب یکسان ندارند)."
        )
        if DROP_AMBIGUOUS_LABEL_COLLISIONS and len(ambiguous_groups) > 0:
            before = len(main_df)
            main_df = main_df[~main_df["id_trunc"].isin(ambiguous_groups)]
            print(
                f"[collision-check] {before - len(main_df)} ردیف به‌دلیل تناقض "
                "برچسب در گروه‌های collide‌شده حذف شدند (DROP_AMBIGUOUS_LABEL_COLLISIONS=True)."
            )

    # برای هر id_trunc باقی‌مانده، یک ردیف نماینده نگه می‌داریم (اولین مورد)
    main_df = main_df.sort_values(
        MAIN_ID_COL).drop_duplicates("id_trunc", keep="first")
    return main_df


def match_users(main_df: pd.DataFrame, tweets_df: pd.DataFrame):
    main_df = main_df.copy()
    tweets_df = tweets_df.copy()
    main_df["id_trunc"] = main_df[MAIN_ID_COL].apply(truncate_id)
    tweets_df["id_trunc"] = tweets_df[TWEETS_ID_COL].apply(truncate_id)

    n_unique_main_users = main_df[MAIN_ID_COL].nunique()
    print(f"[info] تعداد کاربران یکتا در شیت اصلی: {n_unique_main_users}")

    matched_ids = set(main_df["id_trunc"]) & set(tweets_df["id_trunc"])
    main_matched = main_df[main_df["id_trunc"].isin(matched_ids)].copy()
    tweets_matched = tweets_df[tweets_df["id_trunc"].isin(matched_ids)].copy()

    print(f"[match] تعداد id_trunc مشترک بین دو شیت: {len(matched_ids)}")
    return main_matched, tweets_matched


# ============================== آموزش و ارزیابی مدل ==============================

NON_FEATURE_COLS = {"id_trunc", LABEL_COL, "_norm_label", MAIN_ID_COL}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def compute_metrics(y_true, y_pred, y_score=None, positive_label=POSITIVE_LABEL) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=positive_label, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "n_samples": len(y_true),
    }
    if y_score is not None and len(set(y_true)) == 2:
        y_true_bin = [1 if v == positive_label else 0 for v in y_true]
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_bin, y_score)
            metrics["average_precision"] = average_precision_score(
                y_true_bin, y_score)
        except ValueError:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def run_pipeline():
    # ---------- 1) بارگذاری و match ----------
    main_df, tweets_df = load_data()
    main_matched, tweets_matched = match_users(main_df, tweets_df)
    main_matched = resolve_label_collisions(main_matched)

    # ---------- 2) استخراج ویژگی ----------
    feature_df = build_feature_dataset(tweets_matched)
    n_with_corpus = (feature_df["n_raw_tweets_considered"] > 0).sum()
    print(
        f"[info] تعداد کاربران match‌شده با حداقل یک توییت غیرخالی: {n_with_corpus}")

    labels_df = main_matched[[MAIN_ID_COL,
                              "id_trunc", LABEL_COL, "_norm_label"]]
    merged = feature_df.merge(labels_df, on="id_trunc", how="inner")

    # ---------- 3) توزیع برچسب‌ها ----------
    print("\n[labels] توزیع خام ستون برچسب (قبل از نرمال‌سازی):")
    print(main_matched[LABEL_COL].value_counts(dropna=False))

    print("\n[labels] توزیع برچسب نرمال‌شده (Bot/Human/نامعتبر):")
    print(merged["_norm_label"].value_counts(dropna=False))

    valid = merged.dropna(subset=["_norm_label"]).copy()
    n_valid_labels = len(valid)
    print(
        f"\n[info] تعداد کاربران با برچسب معتبر (Bot/Human) و قابل استفاده برای آموزش: {n_valid_labels}")

    if LABEL_COL == "Predicted_Label":
        print(
            "\n[توجه] ستون برچسب فعلی (Predicted_Label) ظاهراً خروجی یک مدل دیگر "
            "(مثلاً Botometer) است، نه لزوماً برچسب واقعی دست‌ساز. بنابراین معیارهای "
            "ارزیابی زیر صرفاً میزان «توافق» مدل یادگیری‌ماشینی ما با آن برچسب‌ها را "
            "نشان می‌دهند، نه لزوماً دقت واقعی تشخیص ربات."
        )

    # ذخیره‌ی جدول ویژگی کامل (شامل کاربرانی که شاید برچسب معتبر نداشته باشند)
    feature_cols_all = get_feature_columns(merged)
    export_cols = [MAIN_ID_COL, "id_trunc", LABEL_COL,
                   "n_raw_tweets_considered"] + feature_cols_all
    merged[export_cols].to_csv(FEATURES_CSV, index=False, encoding="utf-8-sig")
    print(f"[output] جدول ویژگی‌ها ذخیره شد: {FEATURES_CSV}")

    if n_valid_labels == 0:
        print("[خطا] هیچ کاربری با برچسب معتبر یافت نشد؛ آموزش مدل امکان‌پذیر نیست.")
        return

    feature_cols = get_feature_columns(valid)
    print(f"[info] تعداد ویژگی‌های عددی استخراج‌شده: {len(feature_cols)}")

    missing_counts = valid[feature_cols].isna().sum()
    print("\n[diagnostics] تعداد مقادیر گم‌شده به‌ازای هر ویژگی (۱۰ مورد اول):")
    print(missing_counts.sort_values(ascending=False).head(10))

    near_constant = [
        c for c in feature_cols if valid[c].nunique(dropna=True) <= 1
    ]
    if near_constant:
        print(
            f"[diagnostics] ویژگی‌های ثابت/تقریباً ثابت (نادیده گرفته نمی‌شوند ولی گزارش می‌شوند): {near_constant}")

    X = valid[feature_cols].replace([np.inf, -np.inf], np.nan).values
    y = valid["_norm_label"].values
    ids = valid[MAIN_ID_COL].values
    id_trunc_vals = valid["id_trunc"].values

    class_counts = pd.Series(y).value_counts()
    print(
        f"\n[labels] توزیع نهایی کلاس‌ها برای آموزش/ارزیابی:\n{class_counts}")

    # ---------- 4) تقسیم train/test ----------
    min_class_count = class_counts.min() if len(class_counts) > 0 else 0
    can_stratify_split = (
        len(class_counts) == 2 and min_class_count >= MIN_SAMPLES_PER_CLASS_FOR_STRATIFY
    )

    if can_stratify_split:
        X_train, X_test, y_train, y_test, ids_train, ids_test, idt_train, idt_test = train_test_split(
            X, y, ids, id_trunc_vals,
            test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
        )
    else:
        print(
            f"[هشدار] حداقل تعداد نمونه در یک کلاس ({min_class_count}) برای split "
            "لایه‌بندی‌شده کافی نیست؛ از تقسیم تصادفی ساده (بدون stratify) استفاده می‌شود."
        )
        X_train, X_test, y_train, y_test, ids_train, ids_test, idt_train, idt_test = train_test_split(
            X, y, ids, id_trunc_vals,
            test_size=TEST_SIZE, random_state=RANDOM_SEED,
        )

    print(f"\n[split] train: {len(y_train)} نمونه | test: {len(y_test)} نمونه")
    print(f"[split] توزیع کلاس در train:\n{pd.Series(y_train).value_counts()}")
    print(f"[split] توزیع کلاس در test:\n{pd.Series(y_test).value_counts()}")

    # ---------- 5) اعتبارسنجی متقاطع روی train ----------
    train_class_counts = pd.Series(y_train).value_counts()
    n_folds = min(N_CV_FOLDS, int(train_class_counts.min())
                  ) if len(train_class_counts) == 2 else 0

    cv_metrics_rf = None
    oof_pred_df = None
    if n_folds >= 2:
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        rf_pipeline_cv = build_pipeline(
            RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_SEED, class_weight="balanced")
        )

        fold_metrics = []
        oof_preds = np.empty(len(y_train), dtype=object)
        oof_scores = np.full(len(y_train), np.nan)

        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            fold_pipeline = build_pipeline(
                RandomForestClassifier(
                    n_estimators=300, random_state=RANDOM_SEED, class_weight="balanced")
            )
            fold_pipeline.fit(X_train[tr_idx], y_train[tr_idx])
            fold_pred = fold_pipeline.predict(X_train[val_idx])
            classes_ = list(fold_pipeline.named_steps["model"].classes_)
            pos_idx = classes_.index(
                POSITIVE_LABEL) if POSITIVE_LABEL in classes_ else None
            fold_proba = (
                fold_pipeline.predict_proba(X_train[val_idx])[:, pos_idx]
                if pos_idx is not None
                else None
            )
            oof_preds[val_idx] = fold_pred
            if fold_proba is not None:
                oof_scores[val_idx] = fold_proba

            m = compute_metrics(y_train[val_idx], fold_pred, fold_proba)
            fold_metrics.append(m)
            print(
                f"[cv] فولد {fold_i + 1}/{n_folds}: accuracy={m['accuracy']:.4f} f1={m['f1']:.4f}")

        cv_metrics_rf = {
            key: {
                "mean": float(np.nanmean([m[key] for m in fold_metrics if m[key] is not None])),
                "std": float(np.nanstd([m[key] for m in fold_metrics if m[key] is not None])),
            }
            for key in ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "average_precision"]
            if any(m[key] is not None for m in fold_metrics)
        }
        print("\n[cv] میانگین/انحراف‌معیار معیارها روی فولدهای train:")
        for k, v in cv_metrics_rf.items():
            print(f"  {k}: mean={v['mean']:.4f}  std={v['std']:.4f}")

        oof_pred_df = pd.DataFrame(
            {
                MAIN_ID_COL: ids_train,
                "id_trunc": idt_train,
                "true_label": y_train,
                "oof_prediction": oof_preds,
                "oof_score_bot": oof_scores,
            }
        )
    else:
        print(
            f"[هشدار] تعداد نمونه‌ی کمینه‌ی هر کلاس در train ({train_class_counts.min() if len(train_class_counts) else 0}) "
            "برای اعتبارسنجی متقاطع لایه‌بندی‌شده کافی نیست؛ این مرحله رد شد."
        )

    # ---------- 6) آموزش نهایی روی کل train و ارزیابی روی test (held-out) ----------
    results_by_model = {}
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced"
        ),
    }

    fitted_pipelines = {}
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        classes_ = list(pipe.named_steps["model"].classes_)
        pos_idx = classes_.index(
            POSITIVE_LABEL) if POSITIVE_LABEL in classes_ else None
        y_score = pipe.predict_proba(
            X_test)[:, pos_idx] if pos_idx is not None else None

        metrics = compute_metrics(y_test, y_pred, y_score)
        results_by_model[name] = {
            "test_metrics": metrics,
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["Human", "Bot"]).tolist(),
        }
        print(f"\n[test-set نتایج مدل: {name}]")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    primary_model_name = "random_forest"
    primary_pipeline = fitted_pipelines[primary_model_name]
    y_pred_primary = primary_pipeline.predict(X_test)
    classes_ = list(primary_pipeline.named_steps["model"].classes_)
    pos_idx = classes_.index(
        POSITIVE_LABEL) if POSITIVE_LABEL in classes_ else None
    y_score_primary = (
        primary_pipeline.predict_proba(
            X_test)[:, pos_idx] if pos_idx is not None else None
    )

    # ---------- 7) خروجی‌های متریک و نمودارهای تکمیلی ----------
    # یک ردیف برای هر مدل روی test؛ برای Random Forest، میانگین و انحراف معیار CV هم ثبت می‌شود.
    metric_rows = []
    for model_name, model_result in results_by_model.items():
        row = {"model": model_name, "split": "test"}
        row.update(model_result["test_metrics"])
        metric_rows.append(row)

    if cv_metrics_rf:
        for stat_name in ("mean", "std"):
            row = {"model": "random_forest", "split": f"cv_{stat_name}"}
            for metric_name, stats in cv_metrics_rf.items():
                row[metric_name] = stats.get(stat_name)
            row["n_samples"] = len(y_train)
            metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    preferred_cols = [
        "model", "split", "accuracy", "precision", "recall",
        "roc_auc", "average_precision", "f1", "macro_f1", "n_samples"
    ]
    metrics_df = metrics_df[
        [c for c in preferred_cols if c in metrics_df.columns]
        + [c for c in metrics_df.columns if c not in preferred_cols]
    ]
    metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"[output] جدول متریک‌ها ذخیره شد: {METRICS_CSV}")

    # نمودار ROC برای Random Forest (روی test مستقل)
    y_test_bin = np.asarray([1 if v == POSITIVE_LABEL else 0 for v in y_test])
    if y_score_primary is not None and len(np.unique(y_test_bin)) == 2:
        fpr, tpr, _ = roc_curve(y_test_bin, y_score_primary)
        test_roc_auc = roc_auc_score(y_test_bin, y_score_primary)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"Random Forest (AUC = {test_roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — held-out test set")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(ROC_CURVE_PNG, dpi=180)
        plt.close(fig)
        print(f"[output] نمودار ROC ذخیره شد: {ROC_CURVE_PNG}")

        precision_vals, recall_vals, _ = precision_recall_curve(
            y_test_bin, y_score_primary)
        test_ap = average_precision_score(y_test_bin, y_score_primary)
        prevalence = float(np.mean(y_test_bin))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall_vals, precision_vals,
                label=f"Random Forest (AP = {test_ap:.3f})")
        ax.axhline(prevalence, linestyle="--",
                   label=f"Bot prevalence = {prevalence:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve — held-out test set")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(PR_CURVE_PNG, dpi=180)
        plt.close(fig)
        print(f"[output] نمودار Precision-Recall ذخیره شد: {PR_CURVE_PNG}")
    else:
        print("[warning] نمودارهای ROC/PR ساخته نشدند؛ test باید هر دو کلاس را داشته باشد و امتیاز احتمال موجود باشد.")

    # مقایسه‌ی متریک‌های اصلی دو مدل روی test
    compare_metrics = ["accuracy", "precision",
                       "recall", "roc_auc", "average_precision"]
    plot_df = metrics_df[metrics_df["split"] == "test"].set_index("model")
    available_metrics = [
        m for m in compare_metrics if m in plot_df.columns and plot_df[m].notna().any()]
    if available_metrics:
        ax = plot_df[available_metrics].plot(
            kind="bar", figsize=(10, 6), ylim=(0, 1))
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — held-out test set")
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(METRICS_COMPARISON_PNG, dpi=180)
        plt.close()
        print(
            f"[output] نمودار مقایسه‌ی متریک‌ها ذخیره شد: {METRICS_COMPARISON_PNG}")

    # توزیع کلاس‌ها در مجموعه‌ی نهایی، train و test
    dist_df = pd.DataFrame({
        "all": pd.Series(y).value_counts(),
        "train": pd.Series(y_train).value_counts(),
        "test": pd.Series(y_test).value_counts(),
    }).fillna(0).astype(int)
    ax = dist_df.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Number of users")
    ax.set_xlabel("Class")
    ax.set_title("Class Distribution")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CLASS_DISTRIBUTION_PNG, dpi=180)
    plt.close()
    print(f"[output] نمودار توزیع کلاس‌ها ذخیره شد: {CLASS_DISTRIBUTION_PNG}")

    # میانگین متریک‌های اعتبارسنجی متقاطع
    if cv_metrics_rf:
        cv_plot_df = pd.DataFrame({
            metric: values["mean"]
            for metric, values in cv_metrics_rf.items()
            if values.get("mean") is not None
        }, index=["Random Forest CV mean"])
        if not cv_plot_df.empty:
            ax = cv_plot_df.plot(kind="bar", figsize=(10, 5), ylim=(0, 1))
            ax.set_ylabel("Score")
            ax.set_title("Random Forest — Cross-validation Mean Metrics")
            ax.set_xlabel("")
            ax.grid(axis="y", alpha=0.25)
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(CV_METRICS_PNG, dpi=180)
            plt.close()
            print(f"[output] نمودار متریک‌های CV ذخیره شد: {CV_METRICS_PNG}")

    # ---------- 8) گزارش طبقه‌بندی و خروجی‌های موجود ----------
    report_txt = classification_report(y_test, y_pred_primary, digits=4)
    with open(CLASSIFICATION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(
            f"=== Classification report ({primary_model_name}, held-out test set) ===\n")
        f.write(report_txt)
        if cv_metrics_rf:
            f.write(
                "\n\n=== Cross-validation summary (train folds, random_forest) ===\n")
            for k, v in cv_metrics_rf.items():
                f.write(f"{k}: mean={v['mean']:.4f} std={v['std']:.4f}\n")
    print(f"[output] گزارش طبقه‌بندی ذخیره شد: {CLASSIFICATION_REPORT_TXT}")

    cm = confusion_matrix(y_test, y_pred_primary, labels=["Human", "Bot"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Human", "Bot"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Human", "Bot"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({primary_model_name}, test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center",
                    va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=150)
    plt.close(fig)
    print(
        f"[output] نمودار ماتریس درهم‌ریختگی ذخیره شد: {CONFUSION_MATRIX_PNG}")

    importances = primary_pipeline.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV,
                         index=False, encoding="utf-8-sig")
    print(f"[output] اهمیت ویژگی‌ها ذخیره شد: {FEATURE_IMPORTANCE_CSV}")

    top_n = importance_df.head(20)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.barh(top_n["feature"][::-1], top_n["importance"][::-1])
    ax2.set_xlabel("Random Forest Feature Importance")
    ax2.set_title("Top 20 Features")
    fig2.tight_layout()
    fig2.savefig(FEATURE_IMPORTANCE_PNG, dpi=150)
    plt.close(fig2)
    print(f"[output] نمودار اهمیت ویژگی‌ها ذخیره شد: {FEATURE_IMPORTANCE_PNG}")

    test_pred_df = pd.DataFrame(
        {
            MAIN_ID_COL: ids_test,
            "id_trunc": idt_test,
            "true_label": y_test,
            "predicted_label": y_pred_primary,
            "predicted_score_bot": y_score_primary,
            "split": "test",
        }
    )
    if oof_pred_df is not None:
        oof_pred_df = oof_pred_df.rename(
            columns={"oof_prediction": "predicted_label",
                     "oof_score_bot": "predicted_score_bot"}
        )
        oof_pred_df["split"] = "train_oof"
        all_preds_df = pd.concat(
            [test_pred_df, oof_pred_df], ignore_index=True)
    else:
        all_preds_df = test_pred_df
    all_preds_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    print(f"[output] پیش‌بینی‌ها ذخیره شدند: {PREDICTIONS_CSV}")

    final_metrics = {
        "label_column_used": LABEL_COL,
        "label_caveat": (
            "ستون برچسب ممکن است خروجی یک مدل دیگر باشد نه ground truth مستقل؛ "
            "معیارها میزان توافق را نشان می‌دهند."
            if LABEL_COL == "Predicted_Label"
            else None
        ),
        "n_matched_users": int(len(merged)),
        "n_users_with_corpus": int(n_with_corpus),
        "n_users_valid_label": int(n_valid_labels),
        "n_features": len(feature_cols),
        "class_distribution_overall": class_counts.to_dict(),
        "class_distribution_train": pd.Series(y_train).value_counts().to_dict(),
        "class_distribution_test": pd.Series(y_test).value_counts().to_dict(),
        "cv_metrics_random_forest": cv_metrics_rf,
        "test_metrics_by_model": results_by_model,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"[output] معیارهای ارزیابی ذخیره شدند: {METRICS_JSON}")

    print("\n=== خلاصه‌ی نهایی ===")
    print(
        f"کاربران match‌شده: {len(merged)} | دارای توییت: {n_with_corpus} | برچسب معتبر: {n_valid_labels}")
    print(f"تعداد ویژگی: {len(feature_cols)}")
    print(f"فایل‌های خروجی در: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n[توقف] اجرای برنامه توسط کاربر متوقف شد.")
    except FileNotFoundError as exc:
        print(f"\n[خطا] فایل ورودی پیدا نشد: {exc}")
        print("مقادیر MAIN_PATH و TWEETS_PATH را بررسی کن.")
    except KeyError as exc:
        print(f"\n[خطا] ستون یا شیت موردنیاز پیدا نشد: {exc}")
        print(
            "نام ستون‌های user_id، text و Predicted_Label "
            "و تنظیمات شیت‌های ورودی را بررسی کن."
        )
    except Exception as exc:
        print(f"\n[خطای غیرمنتظره] {type(exc).__name__}: {exc}")
        raise
