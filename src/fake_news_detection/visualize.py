"""All plotting code. Every function saves a PNG and returns the output path."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers / CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # noqa: E402

from . import config  # noqa: E402

FAKE_COLOR = "#C62828"
REAL_COLOR = "#1F4E79"
NEUTRAL = "#F9A825"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.25,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(labels: pd.Series, out_dir: Path) -> Path:
    counts = labels.map(config.LABEL_NAMES).value_counts().reindex(["REAL", "FAKE"])
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[REAL_COLOR, FAKE_COLOR])
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.set_title("Class distribution")
    ax.set_ylabel("Number of articles")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir / "class_distribution.png")


def plot_subject_by_class(subjects: pd.Series, labels: pd.Series, out_dir: Path) -> Path | None:
    """Shows the `subject` shortcut: subjects almost perfectly separate the classes."""
    if subjects.isna().all():
        return None
    ct = pd.crosstab(subjects.fillna("unknown"), labels.map(config.LABEL_NAMES))
    ct = ct.reindex(columns=["REAL", "FAKE"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct.plot(kind="barh", stacked=True, color=[REAL_COLOR, FAKE_COLOR], ax=ax)
    ax.set_title("Articles per `subject` and class (a data-leakage warning sign)")
    ax.set_xlabel("Number of articles")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir / "subject_by_class.png")


def plot_length_distribution(texts: pd.Series, labels: pd.Series, out_dir: Path) -> Path:
    lengths = texts.str.split().str.len()
    upper = float(np.percentile(lengths, 99))
    bins = np.linspace(0, max(upper, 1), 50)
    fig, ax = plt.subplots(figsize=(7, 4))
    for value, color in ((0, REAL_COLOR), (1, FAKE_COLOR)):
        ax.hist(lengths[labels == value].clip(upper=upper), bins=bins, alpha=0.55, color=color,
                label=config.LABEL_NAMES[value])
    ax.set_title("Article length (words, clipped at 99th percentile)")
    ax.set_xlabel("Words")
    ax.set_ylabel("Number of articles")
    ax.legend()
    return _save(fig, out_dir / "article_length_distribution.png")


def word_frequencies(clean_texts: pd.Series) -> Counter:
    counter: Counter = Counter()
    for text in clean_texts:
        counter.update(text.split())
    return counter


def plot_wordclouds(freq_real: Counter, freq_fake: Counter, out_dir: Path) -> Path:
    """Bonus: word clouds of common terms in REAL vs FAKE news."""
    from wordcloud import WordCloud

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, freq, cmap, title in ((axes[0], freq_real, "Blues", "REAL news"),
                                  (axes[1], freq_fake, "Reds", "FAKE news")):
        cloud = WordCloud(width=900, height=600, background_color="white", colormap=cmap,
                          max_words=150, random_state=config.RANDOM_STATE)
        cloud.generate_from_frequencies(dict(freq.most_common(500)))
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    return _save(fig, out_dir / "wordclouds_fake_vs_real.png")


def plot_top_words(freq_real: Counter, freq_fake: Counter, out_dir: Path, n: int = 20) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, freq, color, title in ((axes[0], freq_real, REAL_COLOR, "Most frequent words - REAL"),
                                   (axes[1], freq_fake, FAKE_COLOR, "Most frequent words - FAKE")):
        words, counts = zip(*freq.most_common(n)) if freq else ((), ())
        ax.barh(list(words)[::-1], list(counts)[::-1], color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Frequency")
        ax.grid(axis="y", visible=False)
    return _save(fig, out_dir / "top_words_fake_vs_real.png")


def plot_confusion_matrix(y_true, y_pred, title: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], display_labels=["REAL", "FAKE"],
                                            cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    return _save(fig, out_dir / "confusion_matrix.png")


def plot_roc_curves(curves: dict[str, tuple], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=1.8, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_dir / "roc_curves.png")


def plot_model_comparison(results: pd.DataFrame, out_dir: Path) -> Path:
    df = results.sort_values("accuracy")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(df) + 1.8))
    ax.barh(y - 0.2, df["accuracy"], height=0.38, color=REAL_COLOR, label="Accuracy")
    ax.barh(y + 0.2, df["f1"], height=0.38, color=NEUTRAL, label="F1 (FAKE class)")
    ax.set_yticks(y, df["model_name"])
    lo = max(0.0, float(min(df["accuracy"].min(), df["f1"].min())) - 0.05)
    hi = float(max(df["accuracy"].max(), df["f1"].max()))
    ax.set_xlim(lo, min(1.03, hi + 0.05))
    for i, (a, f) in enumerate(zip(df["accuracy"], df["f1"])):
        ax.text(a + 0.001, i - 0.2, f"{a:.4f}", va="center", fontsize=8)
        ax.text(f + 0.001, i + 0.2, f"{f:.4f}", va="center", fontsize=8)
    ax.set_title("Model comparison (held-out test set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir / "model_comparison.png")


def plot_top_features(real, fake, title: str, out_dir: Path, n: int = 20) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    for ax, feats, color, name in ((axes[0], real[:n], REAL_COLOR, "Strongest REAL indicators"),
                                   (axes[1], fake[:n], FAKE_COLOR, "Strongest FAKE indicators")):
        ax.barh([t for t, _ in feats][::-1], [abs(w) for _, w in feats][::-1], color=color)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("|weight|")
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, fontweight="bold")
    return _save(fig, out_dir / "top_model_features.png")
