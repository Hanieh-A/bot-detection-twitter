"""
persian_lfc.py
--------------
Persian-language adaptation of the "LFC" (Linguistic Features Classifier)
idea from Trokhymovych et al. (2026) -- their original module uses
lftk + spaCy, which explicitly do NOT support Persian ("No for bg, ar, ja,
fa" per their own code comment). Since our tweet text is ~100% Persian,
we substitute a comparable set of *language-agnostic + Persian-aware*
linguistic features, built with `hazm` (a Persian NLP library), and use
the same "linguistic features -> gradient-boosted tree classifier" recipe
they used -- just with a different feature-extraction backend.

IMPORTANT ADAPTATION NOTE (state this plainly in the thesis):
  * The original paper trains LFC to distinguish REAL vs LLM-GENERATED
    text (paired samples of the same author/topic). We do not have such
    paired data -- we only have bot-authored vs human-authored tweets.
    So this module instead trains a bot-vs-human linguistic classifier,
    which is a reasonable adaptation of the same idea (linguistic
    "fingerprint" classification) but is NOT the same task the original
    paper solves. Its output ("linguistic bot score") is used as ONE
    additional feature for the fusion model, not as a direct reproduction
    of their reported numbers.
  * hazm's POS-tagger requires downloading a model from the HuggingFace
    Hub, which is unreachable in this environment -- so POS-ratio
    features (which the original lftk feature set relies on heavily)
    are NOT included here. Word-count, wentence-length, stopword-ratio,
    lexical-diversity, punctuation, script-mixing and ZWNJ features are
    used instead.

Usage:
    from persian_lfc import extract_persian_ling_features
    feat_df = extract_persian_ling_features(list_of_texts)
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
import hazm

_normalizer = hazm.Normalizer()
_tokenizer = hazm.WordTokenizer()  # reused instance -- hazm.word_tokenize() re-loads
                                     # its word/verb lexicon files from disk on every
                                     # call, which is ~1000x slower at scale.
_stopwords = set(hazm.stopwords_list())

_PERSIAN_CHAR_RE = re.compile(r'[\u0600-\u06FF]')
_LATIN_CHAR_RE = re.compile(r'[A-Za-z]')
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
_SENT_SPLIT_RE = re.compile(r'[.!?؟\n]+')


def _char_entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    probs = np.array(list(counts.values())) / n
    return float(-(probs * np.log2(probs)).sum())


def _features_for_text(text):
    if not isinstance(text, str) or not text.strip():
        return _empty_features()

    raw = text
    norm = _normalizer.normalize(raw)
    words = _tokenizer.tokenize(norm)
    words_alpha = [w for w in words if any(c.isalpha() for c in w)]
    n_words = len(words_alpha)
    n_chars = len(norm)

    sentences = [s for s in _SENT_SPLIT_RE.split(norm) if s.strip()]
    n_sent = max(len(sentences), 1)

    unique_ratio = len(set(words_alpha)) / n_words if n_words else 0.0
    avg_word_len = np.mean([len(w) for w in words_alpha]) if n_words else 0.0
    avg_sent_len_words = n_words / n_sent

    stop_ratio = (sum(1 for w in words_alpha if w in _stopwords) / n_words) if n_words else 0.0

    persian_chars = len(_PERSIAN_CHAR_RE.findall(raw))
    latin_chars = len(_LATIN_CHAR_RE.findall(raw))
    total_alpha_chars = persian_chars + latin_chars
    latin_ratio = latin_chars / total_alpha_chars if total_alpha_chars else 0.0

    zwnj_ratio = raw.count('\u200c') / n_chars if n_chars else 0.0

    n_excl = raw.count('!')
    n_quest = raw.count('؟') + raw.count('?')
    n_emoji = len(_EMOJI_RE.findall(raw))
    n_hashtag = len(re.findall(r'#\w+', raw))
    n_mention = len(re.findall(r'@\w+', raw))
    n_url = len(re.findall(r'https?://\S+', raw))

    # repetition: how dominant is the most frequent word (bots repeat slogans/hashtags)
    if n_words:
        top_freq = Counter(words_alpha).most_common(1)[0][1]
        repetition_ratio = top_freq / n_words
    else:
        repetition_ratio = 0.0

    return {
        'ling_n_chars': n_chars,
        'ling_n_words': n_words,
        'ling_n_sentences': n_sent,
        'ling_unique_word_ratio': unique_ratio,
        'ling_avg_word_len': avg_word_len,
        'ling_avg_sentence_len_words': avg_sent_len_words,
        'ling_stopword_ratio': stop_ratio,
        'ling_latin_char_ratio': latin_ratio,
        'ling_zwnj_ratio': zwnj_ratio,
        'ling_excl_per_100chars': 100 * n_excl / max(n_chars, 1),
        'ling_quest_per_100chars': 100 * n_quest / max(n_chars, 1),
        'ling_emoji_per_100chars': 100 * n_emoji / max(n_chars, 1),
        'ling_hashtag_per_100chars': 100 * n_hashtag / max(n_chars, 1),
        'ling_mention_per_100chars': 100 * n_mention / max(n_chars, 1),
        'ling_url_per_100chars': 100 * n_url / max(n_chars, 1),
        'ling_repetition_ratio': repetition_ratio,
        'ling_char_entropy': _char_entropy(norm),
    }


def _empty_features():
    return {k: 0.0 for k in [
        'ling_n_chars', 'ling_n_words', 'ling_n_sentences', 'ling_unique_word_ratio',
        'ling_avg_word_len', 'ling_avg_sentence_len_words', 'ling_stopword_ratio',
        'ling_latin_char_ratio', 'ling_zwnj_ratio', 'ling_excl_per_100chars',
        'ling_quest_per_100chars', 'ling_emoji_per_100chars', 'ling_hashtag_per_100chars',
        'ling_mention_per_100chars', 'ling_url_per_100chars', 'ling_repetition_ratio',
        'ling_char_entropy',
    ]}


LFC_FEATURE_NAMES = list(_empty_features().keys())


def extract_persian_ling_features(texts):
    """texts: iterable of raw tweet-sample strings (one per user).
    Returns a DataFrame with LFC_FEATURE_NAMES columns, same length/order."""
    rows = [_features_for_text(t) for t in texts]
    return pd.DataFrame(rows, columns=LFC_FEATURE_NAMES)
