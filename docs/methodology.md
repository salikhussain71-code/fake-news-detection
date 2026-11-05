# Methodology

## 1. Problem
Binary classification of news articles as **REAL (0)** or **FAKE (1)**. FAKE is the *positive* class,
so precision / recall / F1 describe how well fake articles are detected.

## 2. The shortcut problem (read this first)
On the Kaggle *Fake and Real News* dataset a naive TF-IDF + Logistic Regression model scores about
99 %. Much of that comes from **where the articles were collected**, not from anything about
truthfulness:

* Every real article is a Reuters story and starts with `WASHINGTON (Reuters) -`.
* Fake articles carry their own tells (`Featured image via Getty Images`, `21st Century Wire`, ...).
* The `subject` column (`politicsNews`, `worldnews` vs `News`, `left-news`, ...) separates the two
  classes almost perfectly.

A model that learns these artefacts will fail on any other news source. This project therefore:

1. **Removes the source tells** from the text by default (`PreprocessConfig.strip_leakage`).
2. **Never uses `subject` or `date`** as features.
3. Provides `--leakage-ablation`, which reports three numbers side by side:
   text with tells removed (default), text with tells kept, and a **`subject`-only baseline**.
   The gap quantifies how much of the headline accuracy is shortcut learning.

Even after removing the tells, the two sources still differ strongly in *style and topic*, so high
scores remain. That is a property of the dataset (see Limitations).

## 3. Data protocol (no leakage)
* Title and body are concatenated (`--fields title|body|both` lets you test each separately).
* Exact duplicates are dropped; identical text with conflicting labels is dropped entirely.
* **Stratified 80/20 split** (`random_state=42`) *after* de-duplication.
* The TF-IDF vectoriser lives inside a scikit-learn `Pipeline`, so it is fitted on training folds only.
* Models are compared by **5-fold CV F1 on the training split**; the test set is evaluated once.

## 4. Preprocessing (`preprocessing.py`)
| Step | Reason |
|---|---|
| HTML unescape + tag removal, URL removal (incl. `pic.twitter.com/...`) | Noise |
| **Source-tell removal** (datelines, "Featured image via", ...) | Prevents shortcut learning |
| Lower-casing | One feature per word form |
| Regex tokenisation (letters only) | Simple, dependency-free |
| Stop-word removal (NLTK English) | Removes uninformative function words |
| WordNet lemmatisation (cached) | `claims/claim`, `denied/deny` collapse into one feature |

Note: lower-casing discards ALL-CAPS and punctuation (`!!!`) style signals that often distinguish
sensational articles; character-level or style features are a possible extension.

## 5. Features and models
* **TF-IDF**, sub-linear TF, uni+bi-grams, `min_df=2`, `max_df=0.9`, 50,000 features.
* **Logistic Regression** (required baseline), **Linear SVM**, **Multinomial Naive Bayes**.
* `--tune` grid-searches n-gram range and the main regularisation parameter on the training split.

## 6. Evaluation
Accuracy and **F1 (FAKE class)** as required, plus precision, recall, macro-F1, ROC-AUC, 5-fold CV,
confusion matrix, ROC curves and a CSV of the most confident mistakes
(`reports/metrics/misclassified_examples.csv`).

## 7. Explainability (`predict --explain`)
For linear models and Naive Bayes the contribution of a term to one prediction is
`weight x tf-idf value`. The CLI lists the terms pushing towards FAKE and towards REAL, which makes
shortcut features easy to spot.

## 8. Limitations and responsible use
* **The model does not verify facts.** It recognises writing patterns of two specific sources.
* Reuters (real) vs a mixture of hyper-partisan sites (fake) is an easy contrast; performance on
  other outlets, topics, languages or time periods is expected to be much lower.
* Real-world misinformation is often well written and mixes true and false claims.
* Never use the output as the sole basis for moderation or accusations.

## 9. Extensions
Evaluate on a different dataset (cross-dataset generalisation), add style features (capitalisation,
punctuation), embeddings or a fine-tuned transformer, probability calibration, a Streamlit demo.
