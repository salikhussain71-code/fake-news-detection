# Fake News Detection

[![CI](https://github.com/YOUR-USERNAME/fake-news-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/fake-news-detection/actions)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An end-to-end, reproducible NLP project that classifies news articles as **REAL** or **FAKE**. It covers
preprocessing of title and content (stop-word removal, lemmatisation), **TF-IDF** vectorisation,
**Logistic Regression**, **Linear SVM** and **Naive Bayes** classifiers, evaluation with **accuracy
and F1-score**, **word clouds** of fake vs real news, and explainable predictions.

> **Task 3 - Level 2 (NLP Tasks).** Dataset: Fake and Real News (Kaggle). Includes the bonus word-cloud
> visualisation, plus a **data-leakage analysis** that most tutorials skip.

---

## Table of contents
1. [Highlights](#highlights)
2. [Task requirements checklist](#task-requirements-checklist)
3. [Important: the dataset has shortcuts](#important-the-dataset-has-shortcuts)
4. [Project structure](#project-structure)
5. [Quick start](#quick-start)
6. [Usage](#usage)
7. [Results](#results)
8. [Methodology](#methodology)
9. [Testing & code quality](#testing--code-quality)
10. [Limitations & responsible use](#limitations--responsible-use)
11. [License](#license)

## Highlights
* **Clean, modular package** (`src/fake_news_detection/`) - not just a notebook.
* **Leakage-aware**: removes source tells (`(Reuters)` datelines, "Featured image via", ...), never uses the
  `subject` / `date` columns, and can **quantify the shortcut** with `--leakage-ablation`.
* **No data leakage in evaluation**: duplicates and conflicting duplicates removed before an 80/20
  stratified split; TF-IDF fitted on training folds only; the best model is chosen by 5-fold CV, so the
  test score is unbiased.
* **Explainable**: `predict --explain` shows which terms pushed an article towards FAKE or REAL.
* Metrics: accuracy, precision, recall, **F1 (FAKE)**, ROC-AUC; confusion matrix; error-analysis CSV.
* Flexible input: Kaggle `Fake.csv` + `True.csv`, or any single labelled CSV (`--data ... --fake-label`).
* Title / body ablation (`--fields title|body|both`), optional tuning (`--tune`), notebook, tests, CI.

## Task requirements checklist

| Requirement | Where it is implemented |
|---|---|
| Dataset: Fake and Real News (Kaggle) | `data/README.md`, `src/fake_news_detection/data.py` |
| Classify articles as real or fake from their text | `models.py`, `train.py` |
| Preprocess title and content (stop-words, lemmatise, vectorise) | `preprocessing.py`, `features.py`, `data.py` (`--fields`) |
| Logistic Regression **or** SVM classifier | `models.py` - both, plus Naive Bayes |
| Evaluate with accuracy and F1-score | `evaluate.py` (+ precision, recall, ROC-AUC, CV) |
| **Bonus:** word cloud of common terms in fake vs real news | `visualize.py` -> `wordclouds_fake_vs_real.png` |

## Important: the dataset has shortcuts

On this dataset a naive model reaches ~99 % mostly because **every real article starts with
`WASHINGTON (Reuters) -`**, fake articles have their own boilerplate, and the `subject` column almost
perfectly separates the classes. A model can score high without learning anything about fakeness.

This project removes those tells by default and offers a one-command check:

```bash
python -m fake_news_detection.train --leakage-ablation
```
It reports accuracy / F1 for (1) text with tells **removed**, (2) text with tells **kept**, and (3) a
**`subject`-only** baseline, so you can see exactly how much of the score is shortcut. Details in
[`docs/methodology.md`](docs/methodology.md).

## Project structure

```
fake-news-detection/
├── README.md
├── LICENSE
├── Makefile                      # shortcuts: make train / ablation / test / lint
├── pyproject.toml                # package metadata, pytest & ruff config
├── requirements.txt              # runtime dependencies
├── requirements-dev.txt          # + pytest, ruff, jupyter
├── .gitignore
├── .github/workflows/ci.yml      # lint + tests on every push
├── data/
│   ├── README.md                 # how to download the dataset, known data problems
│   ├── raw/                      # put Fake.csv and True.csv here (git-ignored)
│   ├── processed/                # cleaned data written by the pipeline (git-ignored)
│   └── sample/                   # synthetic data for smoke tests only
├── docs/methodology.md           # design decisions, leakage analysis, limitations
├── notebooks/
│   └── 01_fake_news_detection.ipynb
├── scripts/                      # generate_sample_data.py, build_notebook.py
├── src/fake_news_detection/
│   ├── config.py                 # paths & constants
│   ├── data.py                   # loading, title+body, de-duplication
│   ├── preprocessing.py          # text cleaning + source-tell removal
│   ├── features.py               # TF-IDF
│   ├── models.py                 # models, scoring, per-prediction explanations
│   ├── evaluate.py               # metrics, cross-validation, error analysis
│   ├── visualize.py              # all plots (word clouds, ROC, ...)
│   ├── train.py                  # end-to-end training CLI (+ leakage ablation)
│   └── predict.py                # inference CLI / API
├── tests/                        # pytest unit & integration tests
├── models/                       # saved model (git-ignored)
└── reports/
    ├── figures/                  # PNG charts generated by training
    └── metrics/                  # CSV / JSON / Markdown metrics, misclassified examples
```

## Quick start

**Requirements:** Python 3.9+ and about 2 GB of free RAM.

```bash
# 1. Clone
git clone https://github.com/YOUR-USERNAME/fake-news-detection.git
cd fake-news-detection

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt
pip install -e .

# 4. Get the data (see data/README.md) -> data/raw/Fake.csv and data/raw/True.csv
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip

# 5. Train, evaluate and generate all figures
python -m fake_news_detection.train
```

NLTK data (`stopwords`, `wordnet`) is downloaded automatically on first run.
No Kaggle account? Download the CSVs in the browser and place them in `data/raw/`.

## Usage

### Train
```bash
python -m fake_news_detection.train                        # default (source tells removed)
python -m fake_news_detection.train --leakage-ablation     # + quantify the shortcut
python -m fake_news_detection.train --tune                 # grid-search the best model
python -m fake_news_detection.train --fields title         # headline-only model
python -m fake_news_detection.train --no-strip-leakage     # keep tells (inflated score, for comparison)
python -m fake_news_detection.train --help                 # all options
```

Outputs: `models/fake_news_model.joblib`, `reports/metrics/*`, `reports/figures/*.png`, and the
**Results** section of this README (updated automatically).

### Predict
```bash
python -m fake_news_detection.predict --explain "SHOCKING: the elites LIE about the truth!!! Share before it is deleted"
python -m fake_news_detection.predict --title "Senate passes budget" --text "The Senate voted on Tuesday ..."
python -m fake_news_detection.predict --file articles.txt        # one article per line
```
Example output (illustrative - values depend on your trained model):
```
[FAKE | confidence 9x.x%] SHOCKING: the elites LIE about the truth!!! Share before it is deleted
    pushes towards FAKE: share (+0.xx), delete (+0.xx), truth (+0.xx), ...
Note: this model detects writing patterns of the training data; it does not verify facts.
```

From Python:
```python
from fake_news_detection.predict import FakeNewsDetector

detector = FakeNewsDetector()
detector.predict(["Article text ..."], explain=True)
```

### Notebook
```bash
pip install -r requirements-dev.txt
jupyter notebook notebooks/01_fake_news_detection.ipynb
```

## Results

<!-- RESULTS:START -->
_Run `python -m fake_news_detection.train --leakage-ablation` on the Kaggle dataset - this section is then
filled in automatically with your metrics, the shortcut check and figures._
<!-- RESULTS:END -->

**What to expect:** with the source tells *kept*, linear models typically reach about 99 % or more.
With them removed, scores usually drop a little but stay high (roughly 97-99 %), because real (Reuters) and
fake (hyper-partisan) articles still differ strongly in style and topic. That remaining gap is a property of
the dataset - see [Limitations](#limitations--responsible-use). Your exact numbers appear in the table above.

## Methodology

A short summary - see [`docs/methodology.md`](docs/methodology.md) for the reasoning behind every step.

1. **Load** `Fake.csv` / `True.csv` (label 1 = FAKE), combine title + body, drop empty/duplicate/conflicting rows.
2. **Clean**: strip HTML/URLs and **source tells**, lowercase, tokenise, remove stop-words, lemmatise.
3. **Split** 80/20, stratified, fixed seed (`42`).
4. **Vectorise** with TF-IDF (sub-linear TF, uni+bi-grams, 50k features) inside a `Pipeline`.
5. **Train** Logistic Regression, Linear SVM and Naive Bayes; select the best by 5-fold CV F1.
6. **Evaluate once** on the test set (accuracy, F1, ROC-AUC); explain predictions; save everything.

## Testing & code quality

```bash
pip install -r requirements-dev.txt
pytest          # 24 tests: preprocessing, data loading, every model, explanations, end-to-end CLI
ruff check .    # linting
```
GitHub Actions runs both on Python 3.10, 3.11 and 3.12.

## Limitations & responsible use
* **The model does not check facts** - it recognises writing patterns of two specific groups of sources.
* Performance on other outlets, topics, languages or years is expected to be much lower (test it with
  `--data` on another dataset).
* Real misinformation is often well written and mixes true and false statements.
* Do not use the output as the only basis for moderation, publication or accusations.
* Ideas: cross-dataset evaluation, style features (capitalisation, punctuation), embeddings or a fine-tuned transformer.

## License
Released under the [MIT License](LICENSE). Dataset licensing is governed by its Kaggle page.
