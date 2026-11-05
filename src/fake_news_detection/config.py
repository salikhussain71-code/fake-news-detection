"""Central configuration: paths and constants shared across the project."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("FN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_FAKE_PATH = DATA_DIR / "sample" / "Fake.csv"
SAMPLE_TRUE_PATH = DATA_DIR / "sample" / "True.csv"
# Kaggle: clmentbisaillon/fake-and-real-news-dataset  (Fake.csv, True.csv)
DEFAULT_FAKE_PATH = RAW_DIR / "Fake.csv"
DEFAULT_TRUE_PATH = RAW_DIR / "True.csv"
CLEAN_CACHE_PATH = PROCESSED_DIR / "clean_articles.csv"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "fake_news_model.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
README_PATH = PROJECT_ROOT / "README.md"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Label 1 = FAKE is the "positive" class (we are detecting fake news).
LABEL_NAMES = {0: "REAL", 1: "FAKE"}
