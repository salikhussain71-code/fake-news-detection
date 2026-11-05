"""Feature extraction: TF-IDF."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 50_000, ngram_range: tuple[int, int] = (1, 2),
                     min_df: int = 2, max_df: float = 0.9) -> TfidfVectorizer:
    """Unfitted TF-IDF vectoriser (sub-linear TF, uni+bi-grams).

    Always fitted inside a scikit-learn ``Pipeline`` on training data only, so no vocabulary or
    IDF information leaks from the test set.
    """
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df,
                           max_df=max_df, sublinear_tf=True, lowercase=False)
