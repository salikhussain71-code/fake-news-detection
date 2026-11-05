"""Text preprocessing: tokenisation, stop-word removal, lemmatisation, leakage removal.

The *Fake and Real News* dataset contains well-known **source tells** that let a model reach ~99 %
without learning anything about fakeness, e.g. every real article starts with
``WASHINGTON (Reuters) -``. With ``strip_leakage=True`` (default) these are removed so the model must
rely on the actual writing. Switch it off (``--no-strip-leakage``) to see the inflated score.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable

import nltk
import pandas as pd

logger = logging.getLogger(__name__)

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+)", re.IGNORECASE)
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")

# --- source tells (applied on the ORIGINAL-case text, before lower-casing) -------------------
# "WASHINGTON (Reuters) - ", "BERLIN/PARIS (Reuters) - ", "SAN FRANCISCO (Reuters) -"
_DATELINE = re.compile(r"^\s*[A-Z][A-Za-z .,'/\-]{0,60}\(\s*Reuters\s*\)\s*[-\u2013\u2014:]?\s*")
_REUTERS_TAG = re.compile(r"\(\s*Reuters\s*\)\s*[-\u2013\u2014:]?", re.IGNORECASE)
_SOURCE_PHRASES = re.compile(
    r"\b(?:reuters|featured image(?: via| by)?|image via|photo by|getty images?|screenshot|"
    r"21st century wire|read more)\b|\(\s*(?:video|image|photo)\s*\)",
    re.IGNORECASE,
)

SOURCE_TELLS_DESCRIPTION = (
    "Reuters datelines/tags, 'Featured image via', 'Getty Images', 'Image via', 'Photo by', "
    "'Screenshot', '21st Century Wire', 'Read more', '(VIDEO)', pic.twitter.com links"
)


def ensure_nltk_resources() -> None:
    """Download the NLTK corpora needed (stopwords, wordnet) if they are missing."""
    for resource, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource '%s' ...", resource)
            nltk.download(resource, quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
        nltk.data.find("corpora/stopwords")
    except LookupError as exc:  # pragma: no cover - only when offline
        raise RuntimeError(
            "NLTK data could not be downloaded. Run: python -m nltk.downloader stopwords wordnet"
        ) from exc


@lru_cache(maxsize=1)
def _lemmatizer():
    ensure_nltk_resources()
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("test")  # load the corpus now so failures surface early
    return lemmatizer


@lru_cache(maxsize=1)
def default_stopwords() -> frozenset[str]:
    """NLTK English stop words."""
    ensure_nltk_resources()
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=300_000)
def _lemmatize_word(word: str) -> str:
    lemmatizer = _lemmatizer()
    for pos in ("n", "v", "a"):  # noun, verb, adjective
        lemma = lemmatizer.lemmatize(word, pos=pos)
        if lemma != word:
            return lemma
    return word


@dataclass(frozen=True)
class PreprocessConfig:
    """Switches for the cleaning pipeline (stored inside the saved model for inference)."""

    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    strip_leakage: bool = True
    min_token_length: int = 2


class TextPreprocessor:
    """Callable text cleaner: ``TextPreprocessor()(raw_text) -> clean_text``."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self._stopwords = default_stopwords() if self.config.remove_stopwords else frozenset()
        if self.config.lemmatize:
            _lemmatizer()

    def tokenize(self, text: str) -> list[str]:
        """Clean one article and return its list of tokens."""
        cfg = self.config
        if not isinstance(text, str):
            return []
        if cfg.remove_html:
            text = _HTML_TAG.sub(" ", html.unescape(text))
        if cfg.remove_urls:
            text = _URL.sub(" ", text)
        if cfg.strip_leakage:
            text = _DATELINE.sub("", text)
            text = _REUTERS_TAG.sub(" ", text)
            text = _SOURCE_PHRASES.sub(" ", text)
        if cfg.lowercase:
            text = text.lower()
        text = _NON_ALPHA.sub(" ", text)
        text = _WHITESPACE.sub(" ", text).strip()

        tokens = [t for t in text.split(" ") if len(t) >= cfg.min_token_length]
        if cfg.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if cfg.lemmatize:
            tokens = [_lemmatize_word(t) for t in tokens]
        return tokens

    def __call__(self, text: str) -> str:
        return " ".join(self.tokenize(text))

    def transform(self, texts: Iterable[str]) -> list[str]:
        return [self(t) for t in texts]

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series.map(self)

    def to_dict(self) -> dict:
        return asdict(self.config)
