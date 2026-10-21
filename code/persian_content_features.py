# -*- coding: utf-8 -*-
"""
استخراج ویژگی‌های محتوایی متن فارسی با Hazm
==============================================
دو دسته ویژگی:
  1) ویژگی‌های آماری و واژگانی (Lexical)
  2) ویژگی‌های زبان‌شناختی مخصوص فارسی (Linguistic)

ورودی: یک دیتافریم با ستون متنی (پیش‌فرض: 'clean_description')
خروجی: دیتافریم ویژگی‌ها (یک ردیف به ازای هر متن)
"""

import re
import string
from collections import Counter

import numpy as np
import pandas as pd

from hazm import (
    Normalizer,
    WordTokenizer,
    SentenceTokenizer,
    Lemmatizer,
    POSTagger,
)

# ==============================================================
# 1) تنظیمات
# ==============================================================

# کاراکترهای نیم‌فاصله و سایر نشانه‌های خاص فارسی
ZWNJ = "\u200c"           # نیم‌فاصله
TATWEEL = "\u0640"        # کشیده (کشش)
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652]")  # اعراب

# حروف فارسی (شامل آ، ا، ب... ی)
PERSIAN_LETTERS = re.compile(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# حروف لاتین
LATIN_LETTERS = re.compile(r"[A-Za-z]")

# ایموجی (بازه‌های رایج یونیکد)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+", re.IGNORECASE
)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")

# مجموعه‌های کوچک برای ویژگی‌های زبان‌شناختی
# (به‌عنوان مکمل POS Tagger؛ در صورت تمایل می‌توانید گسترش دهید)
PERSIAN_PRONOUNS = {
    "من", "تو", "او", "ما", "شما", "آنها", "ایشان",
    "مرا", "ترا", "اورا", "مارا", "شما را", "آنها را",
    "این", "آن", "اینها", "آنها", "خود", "خویش", "خودم", "خودت",
}
PERSIAN_PREPOSITIONS = {
    "از", "به", "با", "در", "بر", "برای", "تا", "جز", "رب",
    "روی", "زیر", "پیش", "پس", "میان", "بین", "بدون", "مانند",
    "مثل", "نزد", "سوی", "طرف", "درباره", "مورد", "وسیله",
}
PERSIAN_CONJUNCTIONS = {
    "و", "یا", "اما", "ولی", "که", "اگر", "چون", "زیرا",
    "پس", "بنابراین", "هم", "نیز", "نه", "بلکه", "تا", "هرگاه",
}

# پسوندها و پیشوندهای رایج فارسی
PERSIAN_SUFFIXES = ["ها", "های", "هایی", "هائی",
                    "ان", "ات", "اش", "مان", "تان", "شان"]
PERSIAN_PREFIXES = ["می", "نمی", "بی", "نا",
                    "هم", "غیر", "پیش", "پس", "بر", "فرو"]

# واژگان محاوره‌ای رایج (برای نسبت رسمی/محاوره‌ای)
COLLOQUIAL_WORDS = {
    "میره", "میرم", "میری", "میره", "میریم", "میرید", "میرن",
    "میگم", "میگی", "میگه", "میگیم", "میگید", "میگن",
    "نمیره", "نمیرم", "نمیگم", "نمیگه",
    "چیه", "کیه", "کجا", "اینه", "اون", "اینو", "اونو",
    "توو", "تو", "واسه", "واس", "برا", "برای",
    "بچه‌ها", "داداش", "آبجی", "مامان", "بابا",
}


# ==============================================================
# 2) راه‌اندازی ابزارهای Hazm
# ==============================================================

class PersianFeatureExtractor:
    def __init__(self, pos_tagger=None):
        self.normalizer = Normalizer()
        self.word_tokenizer = WordTokenizer(join_verb_parts=False)
        self.sentence_tokenizer = SentenceTokenizer()
        self.lemmatizer = Lemmatizer()
        # POSTagger اختیاری است؛ اگر None باشد، ویژگی‌های POS خالی می‌مانند
        self.pos_tagger = pos_tagger

    # ----------------------------------------------------------
    # پیش‌پردازش پایه
    # ----------------------------------------------------------
    def _preprocess(self, text: str) -> dict:
        """
        نرمال‌سازی و توکن‌سازی. علاوه بر آن، نسخه‌ای از متن که
        نیم‌فاصله‌ها حفظ شده‌اند را نیز نگه می‌داریم.
        """
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)

        # نسخه خام (برای شمارش نیم‌فاصله و کشیده)
        raw = text

        # نرمال‌سازی Hazm (یکسان‌سازی ی/ک عربی، حذف اعراب، اصلاح فاصله‌ها)
        normalized = self.normalizer.normalize(raw)

        # جمله‌بندی
        sentences = self.sentence_tokenizer.tokenize(normalized)

        # کلمه‌بندی (با join_verb_parts=False تا «می‌روم» به «می» و «روم» نشکند)
        words = self.word_tokenizer.tokenize(normalized)

        # لمّatize (برای ویژگی تنوع واژگانی در سطح لم)
        lemmas = []
        for w in words:
            try:
                lem = self.lemmatizer.lemmatize(w)
                # Hazm ممکن است چند لم با # برگرداند؛ اولین را می‌گیریم
                lem = lem.split("#")[0] if lem else w
            except Exception:
                lem = w
            lemmas.append(lem)

        return {
            "raw": raw,
            "normalized": normalized,
            "sentences": sentences,
            "words": words,
            "lemmas": lemmas,
        }

    # ==========================================================
    # روش اول: ویژگی‌های آماری و واژگانی
    # ==========================================================
    def lexical_features(self, pre: dict) -> dict:
        raw = pre["raw"]
        normalized = pre["normalized"]
        sentences = pre["sentences"]
        words = pre["words"]

        # --- آمار پایه ---
        n_chars = len(normalized)
        n_words = len(words)
        n_sentences = max(len(sentences), 1)

        unique_words = set(words)
        n_unique_words = len(unique_words)

        # --- میانگین‌ها ---
        word_lengths = [len(w) for w in words]
        mean_word_len = np.mean(word_lengths) if word_lengths else 0.0
        sent_word_counts = [len(self.word_tokenizer.tokenize(s))
                            for s in sentences]
        mean_sent_len_words = np.mean(
            sent_word_counts) if sent_word_counts else 0.0
        sent_char_counts = [len(s) for s in sentences]
        mean_sent_len_chars = np.mean(
            sent_char_counts) if sent_char_counts else 0.0

        # --- تنوع واژگانی ---
        # Type-Token Ratio ساده
        ttr = n_unique_words / n_words if n_words > 0 else 0.0
        # نسبت کلمات تکراری به کل
        repeated_ratio = 1.0 - ttr if n_words > 0 else 0.0
        # تنوع واژگانی بر اساس لم (نسبت لم‌های منحصربه‌فرد به کل کلمات)
        n_unique_lemmas = len(set(pre["lemmas"]))
        lemma_diversity = n_unique_lemmas / n_words if n_words > 0 else 0.0

        # --- شمارش نشانه‌ها ---
        n_hashtags = len(HASHTAG_PATTERN.findall(normalized))
        n_mentions = len(MENTION_PATTERN.findall(normalized))
        n_urls = len(URL_PATTERN.findall(normalized))
        n_emojis = len(EMOJI_PATTERN.findall(raw))
        n_numbers = len(re.findall(r"\d+", normalized))

        # --- نسبت حروف فارسی به لاتین ---
        n_persian_letters = len(PERSIAN_LETTERS.findall(raw))
        n_latin_letters = len(LATIN_LETTERS.findall(raw))
        total_letters = n_persian_letters + n_latin_letters
        persian_latin_ratio = (
            n_persian_letters / n_latin_letters if n_latin_letters > 0
            else float(n_persian_letters)
        )
        persian_letter_share = (
            n_persian_letters / total_letters if total_letters > 0 else 0.0
        )

        return {
            # حجم متن
            "n_chars": n_chars,
            "n_words": n_words,
            "n_sentences": len(sentences),
            "n_unique_chars": len(set(normalized)),
            "n_unique_words": n_unique_words,
            # میانگین‌ها
            "mean_word_length": mean_word_len,
            "mean_sent_len_words": mean_sent_len_words,
            "mean_sent_len_chars": mean_sent_len_chars,
            # تنوع
            "type_token_ratio": ttr,
            "repeated_word_ratio": repeated_ratio,
            "lemma_diversity": lemma_diversity,
            # نشانه‌ها
            "n_hashtags": n_hashtags,
            "n_mentions": n_mentions,
            "n_urls": n_urls,
            "n_emojis": n_emojis,
            "n_numbers": n_numbers,
            # فارسی/لاتین
            "n_persian_letters": n_persian_letters,
            "n_latin_letters": n_latin_letters,
            "persian_latin_ratio": persian_latin_ratio,
            "persian_letter_share": persian_letter_share,
        }

    # ==========================================================
    # روش دوم: ویژگی‌های زبان‌شناختی مخصوص فارسی
    # ==========================================================
    def linguistic_features(self, pre: dict) -> dict:
        raw = pre["raw"]
        normalized = pre["normalized"]
        words = pre["words"]
        n_words = max(len(words), 1)

        # --- نیم‌فاصله و کشیده ---
        n_zwnj = raw.count(ZWNJ)
        zwnj_per_word = n_zwnj / n_words
        n_tatweel = raw.count(TATWEEL)
        tatweel_per_word = n_tatweel / n_words

        # --- پسوندها و پیشوندها ---
        suffix_counts = {s: 0 for s in PERSIAN_SUFFIXES}
        prefix_counts = {p: 0 for p in PERSIAN_PREFIXES}
        for w in words:
            for s in PERSIAN_SUFFIXES:
                if w.endswith(s) and len(w) > len(s):
                    suffix_counts[s] += 1
            for p in PERSIAN_PREFIXES:
                if w.startswith(p) and len(w) > len(p):
                    prefix_counts[p] += 1

        total_suffix = sum(suffix_counts.values())
        total_prefix = sum(prefix_counts.values())

        # --- ضمایر، حروف اضافه، حروف ربط ---
        n_pronouns = sum(1 for w in words if w in PERSIAN_PRONOUNS)
        n_prepositions = sum(1 for w in words if w in PERSIAN_PREPOSITIONS)
        n_conjunctions = sum(1 for w in words if w in PERSIAN_CONJUNCTIONS)

        # اگر POS Tagger موجود باشد، شمارش دقیق‌تر بر اساس برچسب‌ها
        pos_pronoun = pos_preposition = pos_conjunction = 0
        if self.pos_tagger is not None:
            try:
                tagged = self.pos_tagger.tag(words)
                for _, tag in tagged:
                    tag_u = tag.upper()
                    # ضمایر
                    if "PRON" in tag_u:
                        pos_pronoun += 1
                    # حروف اضافه: ADP (Universal) یا PREP/POSTP (سنتي)
                    if tag_u in {"ADP", "PREP", "POSTP"} or "PREP" in tag_u or "POSTP" in tag_u:
                        pos_preposition += 1
                    # حروف ربط: CCONJ, SCONJ, CONJ
                    if "CONJ" in tag_u:
                        pos_conjunction += 1
            except Exception:
                pass

        # --- نسبت رسمی به محاوره‌ای ---
        n_colloquial = sum(1 for w in words if w in COLLOQUIAL_WORDS)
        colloquial_ratio = n_colloquial / n_words
        formal_ratio = 1.0 - colloquial_ratio

        # --- الگوهای املایی ---
        # کشیدن حروف (تکرار ۳ بار یا بیشتر یک حرف)
        elongation = len(re.findall(r"(.)\1{2,}", raw))
        # تکرار حروف (تکرار ۲ بار)
        char_double = len(re.findall(r"(.)\1", raw))
        # نسبت کلمات با نیم‌فاصله به کل کلمات
        words_with_zwnj = sum(1 for w in words if ZWNJ in w)
        zwnj_word_ratio = words_with_zwnj / n_words

        # --- علائم نگارشی و نشانه‌های عاطفی ---
        persian_punct = set("،؛؟!.:«»…ـ")
        latin_punct = set(string.punctuation)
        n_persian_punct = sum(1 for ch in raw if ch in persian_punct)
        n_latin_punct = sum(1 for ch in raw if ch in latin_punct)
        n_exclamation = raw.count("!") + raw.count("！")
        n_question = raw.count("?") + raw.count("؟")
        emotional_marks = n_exclamation + n_question
        punct_per_char = (n_persian_punct + n_latin_punct) / max(len(raw), 1)

        return {
            # نیم‌فاصله و کشیده
            "n_zwnj": n_zwnj,
            "zwnj_per_word": zwnj_per_word,
            "n_tatweel": n_tatweel,
            "tatweel_per_word": tatweel_per_word,
            "zwnj_word_ratio": zwnj_word_ratio,
            # پسوندها
            "total_suffix_count": total_suffix,
            "suffix_per_word": total_suffix / n_words,
            **{f"suffix_{s}": c for s, c in suffix_counts.items()},
            # پیشوندها
            "total_prefix_count": total_prefix,
            "prefix_per_word": total_prefix / n_words,
            **{f"prefix_{p}": c for p, c in prefix_counts.items()},
            # ضمایر / اضافه / ربط
            "n_pronouns_lexicon": n_pronouns,
            "n_prepositions_lexicon": n_prepositions,
            "n_conjunctions_lexicon": n_conjunctions,
            "pronoun_ratio": n_pronouns / n_words,
            "preposition_ratio": n_prepositions / n_words,
            "conjunction_ratio": n_conjunctions / n_words,
            "n_pronouns_pos": pos_pronoun,
            "n_prepositions_pos": pos_preposition,
            "n_conjunctions_pos": pos_conjunction,
            # رسمی / محاوره‌ای
            "n_colloquial": n_colloquial,
            "colloquial_ratio": colloquial_ratio,
            "formal_ratio": formal_ratio,
            # الگوهای املایی
            "n_elongation": elongation,
            "n_char_double": char_double,
            "elongation_per_char": elongation / max(len(raw), 1),
            # نگارشی و عاطفی
            "n_persian_punct": n_persian_punct,
            "n_latin_punct": n_latin_punct,
            "n_exclamation": n_exclamation,
            "n_question": n_question,
            "emotional_marks": emotional_marks,
            "punct_per_char": punct_per_char,
        }

    # ----------------------------------------------------------
    # استخراج ترکیبی
    # ----------------------------------------------------------
    def extract(self, text: str) -> dict:
        pre = self._preprocess(text)
        feats = {}
        feats.update(self.lexical_features(pre))
        feats.update(self.linguistic_features(pre))
        return feats

    def extract_dataframe(
        self, df: pd.DataFrame, text_col: str = "clean_description"
    ) -> pd.DataFrame:
        records = []
        total = len(df)
        for i, text in enumerate(df[text_col], start=1):
            if i % 500 == 0 or i == total:
                print(f"  پردازش {i}/{total}...")
            records.append(self.extract(text))
        feat_df = pd.DataFrame(records)
        # حفظ شناسه‌ها (اختیاری)
        for col in ["user_id", "screen_name"]:
            if col in df.columns:
                feat_df.insert(0, col, df[col].values)
        return feat_df


# ==============================================================
# 3) مثال استفاده
# ==============================================================

if __name__ == "__main__":
    # --- نمونه متن برای تست سریع ---
    sample_texts = [
        "سلام! من امروز خیلی خوشحالم 😊 #توییتر @user https://t.co/abc",
        "این یک متن رسمی است که با دقت نوشته شده است.",
        "بچه‌ها میگم بیاید بریم بیرون، چیه؟",
        "وای خدای من!!!! این خیلی عالیه 😍😍😍",
    ]

    # --- راه‌اندازی استخراج‌کننده ---
    # اگر مدل POS را دارید و می‌خواهید ویژگی‌های POS هم فعال شوند:
    tagger = POSTagger(repo_id="roshan-research/hazm-postagger",
                       model_filename="pos_tagger.model")
    print(tagger.tag(["با", "که", "من", "و", "در"]))
    extractor = PersianFeatureExtractor(pos_tagger=tagger)
    # extractor = PersianFeatureExtractor(pos_tagger=None)

    # --- استخراج ویژگی برای نمونه‌ها ---
    feats = [extractor.extract(t) for t in sample_texts]
    feat_df = pd.DataFrame(feats)
    print("\nنمونه ویژگی‌های استخراج‌شده:")
    print(feat_df.T.to_string())

    # --- مثال استفاده روی دیتافریم واقعی ---
    # df = pd.read_excel("../datasets/dataset2.xlsx", sheet_name="main")
    # feature_df = extractor.extract_dataframe(df, text_col="clean_description")
    # feature_df.to_csv("../results_graph/persian_content_features.csv",
    #                   index=False, encoding="utf-8-sig")
    # print(f"\nویژگی‌ها برای {len(feature_df)} ردیف ذخیره شد.")
