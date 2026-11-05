"""Dataset loading for the Kaggle *Fake and Real News* dataset (or any labelled CSV).

Label convention: **1 = FAKE (positive class), 0 = REAL**.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

FIELDS = ("both", "title", "body")


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Build the combined text, then remove empty rows, duplicates and label conflicts."""
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["body"] = df["body"].fillna("").astype(str).str.strip()
    df["text"] = (df["title"] + ". " + df["body"]).str.strip(" .")
    n_start = len(df)

    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "label"])
    # the same text labelled both FAKE and REAL is ambiguous: drop all copies
    df = df[~df["text"].duplicated(keep=False)]
    df = df.reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError("The dataset must contain both FAKE and REAL articles.")
    logger.info("Kept %d of %d articles (%d empty / duplicate / conflicting removed).",
                len(df), n_start, n_start - len(df))
    return df[["title", "body", "text", "subject", "label"]]


def load_dataset(fake_path: str | Path | None = None, true_path: str | Path | None = None) -> pd.DataFrame:
    """Load Kaggle's ``Fake.csv`` + ``True.csv`` (columns: title, text, subject, date)."""
    fake_path = Path(fake_path) if fake_path else config.DEFAULT_FAKE_PATH
    true_path = Path(true_path) if true_path else config.DEFAULT_TRUE_PATH
    for p in (fake_path, true_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset file not found: '{p}'.\n"
                "Download the Kaggle 'Fake and Real News Dataset' and place Fake.csv and True.csv in "
                f"'{config.RAW_DIR}' (see data/README.md), or pass --fake / --true."
            )
    frames = []
    for path, label in ((fake_path, 1), (true_path, 0)):
        raw = pd.read_csv(path)
        missing = {"title", "text"} - set(raw.columns)
        if missing:
            raise KeyError(f"Column(s) {sorted(missing)} not found in {path.name}. "
                           f"Available: {list(raw.columns)}.")
        frames.append(pd.DataFrame({
            "title": raw["title"], "body": raw["text"],
            "subject": raw["subject"] if "subject" in raw.columns else None,
            "label": label,
        }))
    return _finalize(pd.concat(frames, ignore_index=True))


def load_single_csv(path: str | Path, label_col: str, text_col: str, title_col: str | None = None,
                    fake_label: str = "1") -> pd.DataFrame:
    """Load one CSV that holds both classes. ``fake_label`` is the label value that means FAKE
    (e.g. ``1`` for most datasets, ``0`` for WELFake, or ``fake`` / ``FAKE``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: '{path}'.")
    raw = pd.read_csv(path)
    needed = {label_col, text_col} | ({title_col} if title_col else set())
    missing = needed - set(raw.columns)
    if missing:
        raise KeyError(f"Column(s) {sorted(missing)} not found in {path.name}. "
                       f"Available: {list(raw.columns)}. Use --label-col / --text-col / --title-col.")
    is_fake = raw[label_col].astype(str).str.strip().str.lower() == str(fake_label).strip().lower()
    return _finalize(pd.DataFrame({
        "title": raw[title_col] if title_col else "", "body": raw[text_col], "subject": None,
        "label": is_fake.astype(int),
    }))


def select_text(df: pd.DataFrame, fields: str = "both") -> pd.Series:
    """Choose which raw field(s) feed the model: title, body, or both."""
    if fields not in FIELDS:
        raise ValueError(f"fields must be one of {FIELDS}")
    if fields == "title":
        return df["title"]
    if fields == "body":
        return df["body"]
    return df["text"]
