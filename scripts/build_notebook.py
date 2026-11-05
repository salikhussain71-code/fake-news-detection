"""Builds notebooks/01_fake_news_detection.ipynb (kept as a script so the notebook is reproducible)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s.strip())  # noqa: E731
code = lambda s: nbf.v4.new_code_cell(s.strip())  # noqa: E731

nb.cells = [
md("""
# Fake News Detection
**Task 3 - NLP Level 2** | Dataset: Fake and Real News (Kaggle) | Label: 1 = FAKE, 0 = REAL

The reusable code lives in `src/fake_news_detection/`; run `python -m fake_news_detection.train` for the full
automated pipeline. This notebook walks through it:

1) load -> 2) explore (and spot the shortcuts) -> 3) preprocess -> 4) split -> 5) TF-IDF -> 6) train & compare ->
7) evaluate -> 8) word clouds (bonus) -> 9) why does it work? -> 10) explain a prediction
"""),
code("""
import sys
import tempfile
from pathlib import Path

try:
    import fake_news_detection  # noqa: F401  (works after `pip install -e .`)
except ImportError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from fake_news_detection import config, visualize
from fake_news_detection.data import load_dataset
from fake_news_detection.evaluate import compute_metrics
from fake_news_detection.models import MODEL_NAMES, build_pipeline, explain_prediction, positive_scores, top_features
from fake_news_detection.preprocessing import PreprocessConfig, TextPreprocessor

pd.set_option("display.max_colwidth", 140)
FIG_DIR = Path(tempfile.mkdtemp())   # notebook figures are temporary; the CLI saves the final ones
"""),
md("## 1. Load the data"),
code("""
if config.DEFAULT_FAKE_PATH.exists() and config.DEFAULT_TRUE_PATH.exists():
    df = load_dataset(config.DEFAULT_FAKE_PATH, config.DEFAULT_TRUE_PATH)
else:
    print("Kaggle CSVs not found in data/raw/ - using the SYNTHETIC sample (results are NOT meaningful).")
    print("See data/README.md to download the real dataset.")
    df = load_dataset(config.SAMPLE_FAKE_PATH, config.SAMPLE_TRUE_PATH)

print(f"{len(df):,} articles after removing empty / duplicate / conflicting rows")
df.sample(5, random_state=1)[["title", "subject", "label"]]
"""),
md("## 2. Explore - and notice the shortcuts\nThe `subject` column (and the Reuters dateline in the text) separate the classes almost perfectly. A model could cheat with them."),
code("""
display(df["label"].map(config.LABEL_NAMES).value_counts().to_frame("articles"))
display(Image(str(visualize.plot_class_distribution(df["label"], FIG_DIR))))
shown = visualize.plot_subject_by_class(df["subject"], df["label"], FIG_DIR)
if shown: display(Image(str(shown)))
print("Real article starts with:", df.loc[df.label == 0, "body"].iloc[0][:80])
"""),
md("## 3. Preprocess\nStrip HTML/URLs and **source tells**, lowercase, remove stop-words, lemmatise."),
code("""
pre = TextPreprocessor()                                        # source tells removed (default)
leaky = TextPreprocessor(PreprocessConfig(strip_leakage=False))  # for comparison
example = "WASHINGTON (Reuters) - The Senators voted on the bills. Featured image via Getty Images"
print("RAW  :", example)
print("CLEAN:", pre(example))
print("KEPT :", leaky(example))
"""),
code("""
df["clean_text"] = pre.transform_series(df["text"])
df = df[df["clean_text"].str.split().str.len() >= 3].reset_index(drop=True)
df[["text", "clean_text", "label"]].head()
"""),
md("## 4. Split\nStratified 80/20. The test set is used once, at the end."),
code("""
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.RANDOM_STATE)
print(f"train={len(X_train):,}  test={len(X_test):,}")
"""),
md("## 5-6. TF-IDF + classifiers\nThe vectoriser sits inside a `Pipeline`, so it is fitted on training data only."),
code("""
rows, fitted = [], {}
for kind in MODEL_NAMES:
    pipe = build_pipeline(kind).fit(X_train, y_train)
    m = compute_metrics(y_test, pipe.predict(X_test), positive_scores(pipe, X_test))
    rows.append({"model": MODEL_NAMES[kind], **m})
    fitted[kind] = pipe
comparison = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
comparison.round(4)
"""),
md("## 7. Evaluate (Logistic Regression)"),
code("""
lr = fitted["lr"]
pred = lr.predict(X_test)
print(classification_report(y_test, pred, target_names=["REAL", "FAKE"], digits=4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["REAL", "FAKE"], cmap="Blues")
plt.title("TF-IDF + Logistic Regression"); plt.grid(False); plt.show()
"""),
md("## 8. Word clouds: fake vs real news (bonus)"),
code("""
freq_real = visualize.word_frequencies(X_train[y_train == 0])
freq_fake = visualize.word_frequencies(X_train[y_train == 1])
display(Image(str(visualize.plot_wordclouds(freq_real, freq_fake, FIG_DIR))))
"""),
md("## 9. What did the model learn?\nCheck whether the strongest indicators are about *style/content* or look like leftover shortcuts."),
code("""
real, fake = top_features(lr, n=20)
display(Image(str(visualize.plot_top_features(real, fake, "Top terms: TF-IDF + Logistic Regression", FIG_DIR))))
"""),
md("### Leakage check: what if we keep the source tells?"),
code("""
raw_train, raw_test = df.loc[X_train.index, "text"], df.loc[X_test.index, "text"]
clean_leaky_train, clean_leaky_test = leaky.transform_series(raw_train), leaky.transform_series(raw_test)
leaky_pipe = build_pipeline("lr").fit(clean_leaky_train, y_train)
acc_leaky = (leaky_pipe.predict(clean_leaky_test) == y_test).mean()
acc_clean = (lr.predict(X_test) == y_test).mean()
print(f"accuracy with source tells removed: {acc_clean:.4f}")
print(f"accuracy with source tells KEPT   : {acc_leaky:.4f}")
"""),
md("## 10. Explain a prediction"),
code("""
def classify(text):
    clean = pre(text)
    p = lr.predict_proba([clean])[0, 1]
    fake_ev, real_ev = explain_prediction(lr, clean, n=4)
    label = "FAKE" if p >= 0.5 else "REAL"
    print(f"{label} (P(fake)={p:.2f})  <-  {text[:80]}")
    print("   towards FAKE:", [t for t, _ in fake_ev], "| towards REAL:", [t for t, _ in real_ev])

classify("SHOCKING: the elites LIE about the truth!!! Share before it is deleted")
classify("WASHINGTON (Reuters) - The Senate said on Tuesday that the budget bill would proceed after review.")
"""),
md("""
## Conclusions
* TF-IDF + a linear model is a strong, interpretable baseline for this dataset.
* High scores are partly due to dataset shortcuts - always check what the model actually uses.
* The model detects *writing patterns*, not truth. Do not use it as a fact-checker.
* **Next steps:** evaluate on another dataset, add style features, try embeddings or a fine-tuned transformer.

> If you ran this on the synthetic sample, re-run it on the real Kaggle dataset before drawing conclusions.
"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "notebooks/01_fake_news_detection.ipynb")
print("notebook written")
