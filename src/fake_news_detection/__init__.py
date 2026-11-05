"""Fake news detection.

Binary text classification: data loading -> leakage-aware preprocessing (stop-words,
lemmatisation) -> TF-IDF -> Logistic Regression / Linear SVM / Naive Bayes -> evaluation
(accuracy, F1) -> visualisation (word clouds) -> explainable inference.
"""

__version__ = "1.0.0"
