"""Model factory, scoring helpers and per-prediction explanations."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config
from .features import build_vectorizer

MODEL_NAMES = {"lr": "Logistic Regression", "svm": "Linear SVM", "nb": "Multinomial Naive Bayes"}

PARAM_GRIDS = {
    "lr": {"vec__ngram_range": [(1, 1), (1, 2)], "clf__C": [1, 5, 10, 30]},
    "svm": {"vec__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 0.3, 0.5, 1.0]},
    "nb": {"vec__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.03, 0.1, 0.3, 1.0]},
}


def build_classifier(kind: str):
    if kind == "lr":
        return LogisticRegression(C=10.0, solver="liblinear", max_iter=1000,
                                  random_state=config.RANDOM_STATE)
    if kind == "svm":
        return LinearSVC(C=0.5, random_state=config.RANDOM_STATE)
    if kind == "nb":
        return MultinomialNB(alpha=0.1)
    raise ValueError(f"Unknown model '{kind}'. Choose from {tuple(MODEL_NAMES)}.")


def build_pipeline(model: str, max_features: int = 50_000) -> Pipeline:
    return Pipeline([("vec", build_vectorizer(max_features=max_features)),
                     ("clf", build_classifier(model))])


def positive_scores(pipeline: Pipeline, texts) -> np.ndarray:
    """Continuous score for the FAKE class (probability if available, else the SVM margin)."""
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    return pipeline.decision_function(texts)


def fake_probability(pipeline: Pipeline, texts) -> np.ndarray:
    """P(fake) in [0, 1]. For the SVM this is a logistic squash of the margin (a *relative* score,
    not a calibrated probability)."""
    scores = positive_scores(pipeline, texts)
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        return scores
    return 1.0 / (1.0 + np.exp(-scores))


def _feature_weights(clf) -> np.ndarray:
    if hasattr(clf, "coef_"):
        return clf.coef_.ravel()
    if hasattr(clf, "feature_log_prob_"):
        return clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    raise ValueError("Classifier does not expose feature weights.")


def top_features(pipeline: Pipeline, n: int = 20):
    """``(real_indicators, fake_indicators)`` as lists of ``(term, weight)``."""
    terms = np.asarray(pipeline.named_steps["vec"].get_feature_names_out())
    weights = _feature_weights(pipeline.named_steps["clf"])
    order = np.argsort(weights)
    real = [(terms[i], float(weights[i])) for i in order[:n]]
    fake = [(terms[i], float(weights[i])) for i in order[::-1][:n]]
    return real, fake


def explain_prediction(pipeline: Pipeline, clean_text: str, n: int = 5):
    """Terms of ONE article that push the decision hardest towards FAKE and towards REAL.

    Contribution = model weight x TF-IDF value of the term in this article.
    Returns ``(fake_evidence, real_evidence)``: lists of ``(term, contribution)``.
    """
    vec, clf = pipeline.named_steps["vec"], pipeline.named_steps["clf"]
    row = vec.transform([clean_text])
    terms = np.asarray(vec.get_feature_names_out())
    idx = row.indices
    contrib = _feature_weights(clf)[idx] * row.data
    order = np.argsort(contrib)
    fake = [(str(terms[idx[i]]), float(contrib[i])) for i in order[::-1][:n] if contrib[i] > 0]
    real = [(str(terms[idx[i]]), float(contrib[i])) for i in order[:n] if contrib[i] < 0]
    return fake, real
