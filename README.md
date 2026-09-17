# Account-history features for social bot detection in the era of large language models
Paper: [katyal_2026_bot_detection.pdf](katyal_2026_bot_detection.pdf) (also on [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6821398))
Code, result tables, and figures accompanying the paper of the same title (Katyal 2026).

The paper evaluates whether account-history features (account age, follower and friend counts, profile completeness, screen-name structure) can substitute for content features in social bot detection when bot operators use language models to launder their text. On a publicly redistributed corpus of 2,432 labeled Twitter accounts, a random forest using only behavioral features achieves ROC-AUC 0.977 in five-fold cross-validation, against 0.830 for a content-only baseline. Behavioral performance is invariant under realistic text rewriting; content performance degrades from AUC 0.842 to 0.785.

## Repository contents

```
code/      Five scripts (Python + Node.js) that reproduce the full analysis
results/   Ten result tables (CSV) and two summary files (JSON)
figures/   Five paper figures (PNG)
```

## Headline results

| Model            | ROC-AUC (5-fold CV) | Brier | FPR   | FNR   |
| ---------------- | ------------------- | ----- | ----- | ----- |
| RF-Content       | 0.830 (0.017)       | 0.155 | 0.130 | 0.322 |
| RF-Behavioral    | 0.977 (0.003)       | 0.055 | 0.034 | 0.118 |
| RF-Fusion        | **0.981 (0.003)**   | 0.050 | 0.022 | 0.102 |

DeLong's test for paired AUC on the held-out test split:

* Behavioral vs Content: z = 9.36, p < 0.001
* Fusion vs Behavioral: z = 2.67, p = 0.008 (difference in AUC: 0.003)

Adversarial robustness, text-rewriting protocol at severity 1.0:

* Content: AUC 0.842 to 0.785
* Behavioral: 0.981 to 0.981 (invariant)
* Fusion: 0.984 to 0.984 (essentially invariant)

Adversarial robustness, feature-space protocol (upper bound on attack strength):

* Content: AUC 0.842 to 0.466 (below chance)
* Behavioral: 0.981 to 0.981 (invariant)

## Reproducing the analysis

Dependencies:

* Python 3.9+ with pandas, scikit-learn, scipy, matplotlib, seaborn
* Internet access on first run: both datasets are downloaded automatically into a
  local `data/` directory (git-ignored), so no manual data setup is required

The two analysis scripts are independent and can be run in either order:

```bash
cd code/
python3 run_analysis.py     # Tables 1-4, Figures 1-4, TwiBot-20 cross-dataset scoring
python3 run_extended.py     # Tables 5-10, Figure 5 (calibration)
```

`features.py` is imported by both and defines the 36-feature engineering pipeline. `regen_figures.py` regenerates the figures with the titles used in the paper. `build_paper.js` is the Node.js script that builds the manuscript itself from the result CSVs and is included for completeness; it is not needed to reproduce the analysis.

## Data sources

Primary dataset, labeled: 2,432 Twitter accounts, 43% bots, manually annotated. Downloaded from the `jubins/MachineLearning-Detecting-Twitter-Bots` repository on GitHub. Direct URL:

```
https://raw.githubusercontent.com/jubins/MachineLearning-Detecting-Twitter-Bots/master/FinalProjectAndCode/kaggle_data/training_data_2_csv_UTF.csv
```

Secondary dataset, unlabeled, used for qualitative cross-dataset validation: the 100-account public sample of TwiBot-20 (Feng et al. 2021). Downloaded from the `BunsenFeng/TwiBot-20` repository on GitHub.

The full labeled TwiBot-20 and TwiBot-22 benchmarks are gated behind an application process administered by the dataset maintainers and are not required to reproduce the results in this repository.

## Citation

If you use this code or build on the analysis, please cite the paper:

```
Katyal, G. 2026. Account-history features for social bot detection in the era
of large language models. [Journal of Online Trust and Safety, under review].
Preprint and code: [arXiv ID to be added on preprint posting].
```

A Zenodo DOI for this code release is available via the release tag on this repository.

## License

MIT License. See LICENSE file.

## Contact

Gaurang Katyal, Independent Researcher
gaurang.katyal@gmail.com
ORCID: 0009-0001-0850-1673
