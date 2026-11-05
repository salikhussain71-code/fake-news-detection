"""Evaluation helpers: metrics, cross-validation, error analysis, serialisation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Binary metrics with FAKE (1) as the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def detailed_report(y_true, y_pred) -> dict:
    names = [config.LABEL_NAMES[0], config.LABEL_NAMES[1]]
    return {
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def cross_validate_pipeline(pipeline, X, y, folds: int = config.CV_FOLDS) -> dict:
    """Stratified k-fold CV on the training data (vectoriser re-fitted inside every fold)."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    s = cross_validate(pipeline, X, y, cv=cv, scoring={"accuracy": "accuracy", "f1": "f1"}, n_jobs=-1)
    return {
        "cv_accuracy_mean": float(np.mean(s["test_accuracy"])),
        "cv_accuracy_std": float(np.std(s["test_accuracy"])),
        "cv_f1_mean": float(np.mean(s["test_f1"])),
        "cv_f1_std": float(np.std(s["test_f1"])),
    }


def misclassified(texts: pd.Series, y_true, y_pred, p_fake, limit: int = 300) -> pd.DataFrame:
    """Error-analysis table (``p_fake`` in [0, 1]): the most confidently wrong predictions first."""
    y_true, y_pred, p_fake = np.asarray(y_true), np.asarray(y_pred), np.asarray(p_fake)
    wrong = np.flatnonzero(y_true != y_pred)
    conf = np.where(y_pred[wrong] == 1, p_fake[wrong], 1 - p_fake[wrong])
    df = pd.DataFrame({
        "text": np.asarray(texts)[wrong].astype(str),
        "true": [config.LABEL_NAMES[int(i)] for i in y_true[wrong]],
        "predicted": [config.LABEL_NAMES[int(i)] for i in y_pred[wrong]],
        "confidence": conf,
    })
    df["text"] = df["text"].str.slice(0, 400)
    return df.sort_values("confidence", ascending=False).head(limit).reset_index(drop=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
