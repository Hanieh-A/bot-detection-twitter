# Persian Twitter Bot Detection

Undergraduate final project: detecting bot accounts in Persian-language Twitter using account-behavior features, Persian linguistic features, and semantic text embeddings, evaluated with a leakage-safe nested cross-validation protocol.

<div dir="rtl" align="right">

**خلاصه:** این پروژه‌ی پایانی کارشناسی به تشخیص حساب‌های ربات در توییتر فارسی می‌پردازد. روش اصلی، جنگل تصادفی روی ویژگی‌های رفتاری حساب است (۲۴ ویژگی مقاله‌ی مرجع و ویژگی‌های تکمیلی افزوده‌شده)؛ ویژگی‌های محتوایی، زبانی (hazm) و معنایی (MiniLM) نیز آزموده شده‌اند. نتایج اصلی با ارزیابی تودرتو و پیش‌پردازش محلی در هر فولد (بدون نشت داده) در پوشه‌ی `results_main/` قرار دارند. نتایج قدیمی‌تر در `results_legacy_d1/` و `results_legacy_d2/` فقط به‌عنوان آزمایش‌های اکتشافی نگه‌داری شده‌اند.

</div>

## Contents

- [Overview](#overview)
- [Data](#data)
- [Features](#features)
- [Evaluation protocol](#evaluation-protocol)
- [Results](#results)
- [Repository layout](#repository-layout)
- [How to run](#how-to-run)
- [Limitations](#limitations)
- [References](#references)

## Overview

The project starts from the behavioral-fingerprinting idea of Katyal (2026): account-history features (account age, follower/friend counts, profile completeness, screen-name structure) are harder for a language-model-driven bot to disguise than tweet text. It reproduces that 24-feature baseline on two Persian datasets and then studies:

1. **Behavioral features.** How well a random forest on account features separates bots from non-bots, and what additional activity features (tweet-type proportions, posting rates, time-gap patterns) contribute.
2. **Content, linguistic and semantic features.** Whether text-derived signals (simple content statistics, Persian linguistic features adapted from Trokhymovych et al. (2026), cross-account duplicate-content features, and multilingual sentence embeddings) add anything beyond behavior, compared on exactly the same accounts.
3. **Robustness and generalization.** Behavior of the models under heuristic text rewriting and when trained on one dataset and tested on the other (legacy experiments).
4. **Evaluation hygiene.** All headline numbers come from nested cross-validation in which imputation, scaling and threshold selection are fitted only on training data.

## Data

The datasets are **not included** in this repository (`datasets/` is git-ignored). Place the two Excel files at:

```
datasets/dataset1.xlsx
datasets/dataset2.xlsx
```

Both come from the same source: Persian-language accounts active around a politically charged topic, collected by a social-media analytics company. Dataset 1 was labeled by the project team.

| | Dataset 1 | Dataset 2 |
|---|---|---|
| Accounts | 1,103 (1,089 with a resolved label) | 19,510 |
| Sheets used | `1k_users`, `tweets_meta_data` | `main`, `suspend_users` (+ tweet sheets for text experiments) |
| Tweet text | 75,913 tweets covering 1,099 accounts | Available only for a subset of accounts |
| Label source | `final-tag`: majority vote of three annotators | `Bot_Probability` column: a probability (0 to 1) that the account is a bot, present for every account |
| Binary target | bot = class 1 (bot); all other classes = 0; unresolved ("can't find") rows dropped | bot = probability ≥ 0.5; suspended accounts only fill missing labels; conflicts are flagged in `label_source` |
| Bots in `results_main` | 77 of 1,089 (7.1%) | 2,486 of 19,510 (12.7%) |

Notes:

- Dataset 1 labels are the more carefully produced (three annotators), but inter-annotator agreement was low and the number of positives is small.
- Dataset 2 labels are obtained by thresholding a probability score rather than from a multi-annotator vote, so treat them as **proxy labels**. How the probabilities were produced is not recorded in this repository. Results on the two datasets are therefore not directly comparable.
- The external Botometer score is used only as an independent diagnostic. It is never used to construct labels.
- Dataset checks (required columns, duplicate IDs and screen names, label and text coverage) are in `code/data_audit.py` and are stored in `results_main/metadata.json`.

## Features

| Feature set | Size | Module |
|---|---|---|
| Base behavioral (Katyal 2026): account-age rates, follow asymmetries, profile completeness, screen-name structure | 24 | `features.py` |
| Extra behavioral, dataset 1: tweet-type proportions, log-transformed rates, time-gap and burst features, new-account activity flag | +21 (45 total) | `extra_features_d1.py` |
| Extra behavioral, dataset 2 (same idea; no creation-day flag available) | +20 (44 total) | `extra_features_d2.py` |
| Simple content statistics of the sampled tweet text | 12 | `features.py` |
| Persian linguistic features (hazm-based adaptation of the LFC idea) | 17 | `linguistic_features.py` |
| Cross-account duplicate-content (coordination) features | 2 | `coordination_features.py`, `load_dataset2_text.py` |
| Semantic embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, frozen encoder) of profile description plus sampled tweets | 384 | `semantic_features.py` |

## Evaluation protocol

Implemented in `code/nested_cv.py`, configured in `code/experiment_config.py`.

- **Nested stratified cross-validation:** 5 outer folds, 4 inner folds, seed 42.
- **Fold-local preprocessing:** the median imputer (and the scaler for logistic regression) is fitted on the training part of each fold only, inside an sklearn `Pipeline`.
- **Threshold selection:** the decision threshold is chosen on the inner folds from the grid 0.05 to 0.95 (step 0.05) and applied once to the untouched outer test fold.
- **Model:** random forest with 300 trees and no class weighting; logistic regression as a linear reference.
- **Reported metrics:** ROC-AUC and average precision from pooled out-of-fold probabilities (`roc_auc_oof`, `average_precision_oof`), plus mean outer-fold F1, precision, recall and the standard deviation of fold AUC.
- **Reading F1:** F1 depends on class prevalence. A model that labels every account a bot would score F1 ≈ 0.13 on dataset 1 and ≈ 0.23 on dataset 2, so ROC-AUC is the primary comparison metric.

## Results

Main results from `results_main/` (nested CV; behavioral = 45 features on dataset 1 and 44 on dataset 2).

**Dataset 1, all labeled accounts (n = 1,089, 77 bots)**

| Model | Features | ROC-AUC | F1 |
|---|---|---|---|
| RF behavioral (base) | 24 | 0.701 | 0.260 |
| Logistic regression behavioral (base) | 24 | 0.714 | 0.286 |
| RF behavioral (extended) | 45 | 0.804 | 0.357 |

**Dataset 1, accounts with tweet text, identical rows for all three models (n = 1,086)**

| Model | Features | ROC-AUC | F1 |
|---|---|---|---|
| RF behavioral | 45 | 0.801 | 0.344 |
| RF content | 12 | 0.758 | 0.294 |
| RF fusion (behavioral + content) | 57 | 0.828 | 0.384 |

**Dataset 2 (n = 19,510, 2,486 bots)**

| Model | Features | ROC-AUC | F1 |
|---|---|---|---|
| RF behavioral (base) | 24 | 0.946 | 0.667 |
| RF behavioral (extended) | 44 | 0.981 | 0.801 |
| Botometer raw score (diagnostic, n = 12,438 with a valid score) | 1 | 0.454 | n/a |

How to read these numbers:

- Dataset 1 has few positives, so fold-to-fold variation is large (standard deviation of fold AUC between 0.04 and 0.11). Differences of a few hundredths in AUC, such as 0.801 versus 0.828, were not tested for statistical significance.
- Dataset 2 scores are much higher, but its labels are proxy labels; this does not imply that the model would perform equally well against independently verified labels.
- The extended activity features improve AUC over the 24-feature baseline on both datasets.
- Botometer's raw score is near chance on this population (0.454).

**Semantic-embedding experiment** (`results_semantic/`, dataset 1, n = 1,086, same rows for all models)

| Model | Features | ROC-AUC | F1 |
|---|---|---|---|
| RF behavioral (extended) | 45 | 0.808 | 0.289 |
| RF simple content | 12 | 0.752 | 0.188 |
| RF semantic (MiniLM) | 384 | 0.562 | 0.111 |
| RF behavioral + semantic | 429 | 0.714 | 0.347 |

> **Caution:** `results_semantic/` was generated from an earlier version of the dataset-1 file (SHA-256 `17562c38…`, 4.8% bots) than `results_main/` (SHA-256 `5630ae27…`, 7.1% bots). Its numbers are not comparable with the `results_main` tables above. Re-run `run_semantic_experiments.py` on the current file before citing them together.

## Repository layout

```
code/
  experiment_config.py             seed, CV settings, RF parameters, dataset and results paths
  nested_cv.py                     leakage-safe nested-CV evaluation (nested_cv_evaluate)
  data_audit.py                    dataset checks (columns, duplicates, label/text coverage)

  load_dataset1.py                 loader for dataset 1 (function load_custom_v2)
  load_dataset2.py                 loader for dataset 2 (labels, suspended accounts, Botometer)
  load_dataset2_text.py            tweet text aggregation and coordination features for dataset 2

  features.py                      24 base behavioral + 12 content features (+ helpers for the reference paper's English data)
  extra_features_d1.py             21 extra behavioral features, dataset 1
  extra_features_d2.py             20 extra behavioral features, dataset 2
  linguistic_features.py           17 Persian linguistic features (hazm)
  coordination_features.py         cross-account duplicate-content features
  semantic_features.py             MiniLM sentence embeddings with disk cache

  run_main_experiments.py          MAIN RUNNER: nested-CV experiments on both datasets -> results_main/
  run_semantic_experiments.py      semantic-embedding experiment on dataset 1 -> results_semantic/

  run_analysis_dataset1.py         legacy: reference-paper pipeline on dataset 1
  run_analysis_dataset2.py         legacy: reference-paper pipeline on dataset 2
  run_imbalance.py                 legacy: class-weighting, threshold tuning, SMOTE comparison
  run_shap.py                      legacy: SHAP interpretability of the behavioral model
  run_adversarial.py               legacy: heuristic text-rewriting robustness test
  run_linguistic_fusion_d1.py      legacy: linguistic + coordination fusion, dataset 1
  run_linguistic_fusion_d2.py      legacy: linguistic + coordination fusion, dataset 2
  run_cross_dataset.py             legacy: train on one dataset, test on the other

  run_analysis.py, run_extended.py, regen_figures.py
                                   reproduction of the reference paper on its own English dataset (not used for the thesis results)
  old_load_custom_dataset.py       superseded loader for the first file version (not imported)
  update_thesis.py                 one-off helper that inserted a section into the thesis .docx

results_main/                      current results: metrics, per-fold metrics, out-of-fold predictions, ROC/PR plots, metadata.json
results_semantic/                  semantic-embedding experiment results (see caution above)
results_legacy_d1/                 exploratory results from the pre-nested-CV pipeline, dataset 1
results_legacy_d2/                 exploratory results from the pre-nested-CV pipeline, dataset 2
papers/                            the two reference papers
```

The thesis itself is stored as `پایان_نامه.docx` in the repository root.

## How to run

Tested with Python 3.12.4, pandas 2.3.2, numpy 2.3.5 and scikit-learn 1.8.0.

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl
pip install sentence-transformers      # semantic experiment only
pip install hazm                       # linguistic fusion scripts only
pip install shap imbalanced-learn seaborn scipy python-docx   # legacy scripts only
```

From the repository root, with the datasets in place:

```bash
python code/run_main_experiments.py --dataset d1     # or d2, or both (default)
python code/run_semantic_experiments.py --dataset d1
```

- `run_main_experiments.py` writes everything to `results_main/`, including `metadata.json` (seed, package versions, SHA-256 of both dataset files, data audits). It hashes both dataset files even when only `--dataset d1` is requested, so both files must exist.
- `run_semantic_experiments.py` downloads the MiniLM model on its first run and caches embeddings under `results_semantic/cache/`.
- The legacy scripts take the dataset path as a command-line argument and several still default to old file names, so pass the path explicitly, for example `cd code && python run_shap.py ../datasets/dataset1.xlsx`. They write into `results_legacy_d1/` or `results_legacy_d2/`, so running them will overwrite the legacy outputs stored there.

## Limitations

- Dataset 1 contains only 77 bots among 1,089 labeled accounts, and human annotators agreed only weakly. Estimates are noisy and the labels themselves contain error.
- Dataset 2 labels are proxy labels obtained by thresholding a probability at 0.5; suspended accounts are used only to fill missing labels.
- Tweet text is available for a subset of accounts, so content, linguistic and semantic experiments are restricted to that subset.
- The Persian linguistic features are built without a part-of-speech tagger and are an adaptation, not a reproduction, of the reference paper's model.
- Coordination features count duplicate texts across all accounts in the dataset (a transductive setup); a strict inductive evaluation would compute them from training accounts only.
- Text-rewriting robustness was tested with a heuristic rewriting function, not with a real language model.
- Legacy results fit imputation on the full dataset and select thresholds on the predictions used for reporting. Treat them as exploratory.
- `results_main/metadata.json` was written by the most recent run, which covered dataset 1 only; the dataset-2 result files come from an earlier run. Re-run with `--dataset both` to make the metadata and all result files consistent.

## References

- Katyal, G. (2026). *Account-history features for social bot detection in the era of large language models.* [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6821398). PDF in `papers/`.
- Trokhymovych, M. et al. (2026). *Adversarial Creation and Detection of AI-Generated Social Bot Content.* PDF in `papers/`.
- Reimers, N. and Gurevych, I. Sentence-Transformers, model `paraphrase-multilingual-MiniLM-L12-v2`.
- hazm: Persian natural language processing toolkit.

## Author

Hanieh Abolhasani
