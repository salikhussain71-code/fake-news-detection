"""End-to-end training pipeline for fake news detection.

    python -m fake_news_detection.train                       # Kaggle Fake.csv + True.csv
    python -m fake_news_detection.train --leakage-ablation    # + how much do source tells inflate scores?
    python -m fake_news_detection.train --tune                # grid-search the best model
    python -m fake_news_detection.train --data articles.csv --label-col label --text-col text --fake-label 0

Stages: load -> clean (stop-words, lemmatise, remove source tells) -> EDA plots -> stratified split ->
train LR / SVM / Naive Bayes -> 5-fold CV on train -> select best by CV F1 -> evaluate once on the
held-out test set -> save model, metrics, figures.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from . import __version__, config, visualize
from .data import FIELDS, load_dataset, load_single_csv, select_text
from .evaluate import (
    compute_metrics,
    cross_validate_pipeline,
    detailed_report,
    misclassified,
    save_json,
)
from .models import (
    MODEL_NAMES,
    PARAM_GRIDS,
    build_pipeline,
    fake_probability,
    positive_scores,
    top_features,
)
from .preprocessing import SOURCE_TELLS_DESCRIPTION, PreprocessConfig, TextPreprocessor

logger = logging.getLogger("fake_news_detection")

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"
MIN_TOKENS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate fake-news classifiers.")
    p.add_argument("--fake", type=Path, default=None, help="Fake.csv (default: data/raw/Fake.csv).")
    p.add_argument("--true", type=Path, default=None, help="True.csv (default: data/raw/True.csv).")
    p.add_argument("--data", type=Path, default=None,
                   help="Single CSV containing both classes (alternative to --fake/--true).")
    p.add_argument("--label-col", default="label")
    p.add_argument("--text-col", default="text")
    p.add_argument("--title-col", default="title", help="Title column ('' if none).")
    p.add_argument("--fake-label", default="1", help="Value of the label column that means FAKE.")
    p.add_argument("--fields", choices=FIELDS, default="both", help="Use title, body or both.")
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    p.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    p.add_argument("--cv-folds", type=int, default=config.CV_FOLDS,
                   help="Cross-validation folds on the training split (0 disables CV).")
    p.add_argument("--tune", action="store_true", help="Grid-search the best model (CV, train split only).")
    p.add_argument("--no-strip-leakage", action="store_true",
                   help="Keep source tells such as '(Reuters)' (inflates accuracy - for demonstration).")
    p.add_argument("--leakage-ablation", action="store_true",
                   help="Also measure scores with source tells kept, and a `subject`-only baseline.")
    p.add_argument("--no-lemmatize", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-update-readme", action="store_true")
    p.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    return p.parse_args(argv)


def run_experiments(X_train, y_train, X_test, y_test, kinds, cv_folds):
    rows, pipes, roc = [], {}, {}
    for kind in kinds:
        pipe = build_pipeline(kind)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        y_pred = pipe.predict(X_test)
        y_score = positive_scores(pipe, X_test)
        row = {"model": kind, "model_name": MODEL_NAMES[kind],
               **compute_metrics(y_test, y_pred, y_score), "fit_seconds": round(fit_s, 2)}
        if cv_folds >= 2:
            row.update(cross_validate_pipeline(build_pipeline(kind), X_train, y_train, cv_folds))
        rows.append(row)
        pipes[kind], roc[MODEL_NAMES[kind]] = pipe, (y_test, y_score)
        logger.info("%-26s acc=%.4f f1=%.4f auc=%.4f", MODEL_NAMES[kind], row["accuracy"], row["f1"],
                    row["roc_auc"])
    return pd.DataFrame(rows), pipes, roc


def leakage_ablation(raw_text, labels, subjects, train_idx, test_idx, kind, lemmatize, stripped_metrics):
    """Compare: source tells removed (default) vs kept vs a `subject`-only baseline."""
    pre = TextPreprocessor(PreprocessConfig(lemmatize=lemmatize, strip_leakage=False))
    clean = pre.transform_series(raw_text)
    ok = clean.str.split().str.len() >= MIN_TOKENS
    tr, te = [i for i in train_idx if ok[i]], [i for i in test_idx if ok[i]]
    pipe = build_pipeline(kind).fit(clean[tr], labels[tr])
    kept = compute_metrics(labels[te], pipe.predict(clean[te]), positive_scores(pipe, clean[te]))
    rows = [
        {"setup": "Text only, source tells REMOVED (default)", "accuracy": stripped_metrics["accuracy"],
         "f1": stripped_metrics["f1"]},
        {"setup": "Text only, source tells KEPT (leaky)", "accuracy": kept["accuracy"], "f1": kept["f1"]},
    ]
    if subjects.notna().any():
        grouped = labels[train_idx].groupby(subjects[train_idx].fillna("unknown"))
        majority = grouped.agg(lambda s: int(s.mean() >= 0.5))
        fallback = int(labels[train_idx].mean() >= 0.5)
        pred = subjects[test_idx].fillna("unknown").map(majority).fillna(fallback).astype(int)
        m = compute_metrics(labels[test_idx], pred, pred)
        rows.append({"setup": "`subject` column ONLY (no text)", "accuracy": m["accuracy"], "f1": m["f1"]})
    return pd.DataFrame(rows)


def results_markdown(results: pd.DataFrame, saved_name: str, has_cv: bool) -> str:
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"] + (["cv_f1_mean"] if has_cv else [])
    headers = ["Model", "Accuracy", "Precision (FAKE)", "Recall (FAKE)", "F1 (FAKE)", "ROC-AUC"] + \
        (["CV F1 (mean)"] if has_cv else [])
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"]
    for _, r in results.sort_values("f1", ascending=False).iterrows():
        name = f"**{r['model_name']}**" if r["model_name"] == saved_name else r["model_name"]
        lines.append("| " + " | ".join([name] + [f"{r[c]:.4f}" for c in cols]) + " |")
    return "\n".join(lines)


def ablation_markdown(df: pd.DataFrame) -> str:
    lines = ["| Setup | Accuracy | F1 (FAKE) |", "|---|---:|---:|"]
    lines += [f"| {r.setup} | {r.accuracy:.4f} | {r.f1:.4f} |" for r in df.itertuples()]
    return "\n".join(lines)


def update_readme(table_md: str, summary: dict, ablation_md: str | None, figures: list[str]) -> bool:
    path = config.README_PATH
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        return False
    m = summary["test_metrics"]
    block = [
        RESULTS_START,
        f"_Generated automatically on {summary['trained_at']} - {summary['n_train']:,} training / "
        f"{summary['n_test']:,} test articles, seed {summary['seed']}, "
        f"source tells {'removed' if summary['preprocessing']['strip_leakage'] else 'KEPT'}._",
        "", table_md, "",
        f"**Saved model:** {summary['best_model']} - accuracy **{m['accuracy']:.2%}**, "
        f"F1 (FAKE) **{m['f1']:.2%}**, ROC-AUC **{m['roc_auc']:.4f}** on the held-out test set.",
    ]
    if ablation_md:
        block += ["", "#### Shortcut / leakage check", "", ablation_md]
    if figures:
        block += ["", "| | |", "|:---:|:---:|"]
        cells = [f"![{f}](reports/figures/{f})" for f in figures]
        for i in range(0, len(cells), 2):
            pair = cells[i:i + 2] + [""] * (2 - len(cells[i:i + 2]))
            block.append("| " + " | ".join(pair) + " |")
    block.append(RESULTS_END)
    pattern = re.compile(re.escape(RESULTS_START) + r".*?" + re.escape(RESULTS_END), re.DOTALL)
    path.write_text(pattern.sub(lambda _: "\n".join(block), text, count=1), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    # 1. Load -------------------------------------------------------------------------------
    try:
        if args.data:
            df = load_single_csv(args.data, args.label_col, args.text_col, args.title_col or None,
                                 args.fake_label)
        else:
            df = load_dataset(args.fake, args.true)
    except (FileNotFoundError, KeyError, ValueError) as err:
        logger.error("%s", err.args[0] if err.args else err)
        return 1

    # 2. Clean ------------------------------------------------------------------------------
    pre_cfg = PreprocessConfig(lemmatize=not args.no_lemmatize, strip_leakage=not args.no_strip_leakage)
    preprocessor = TextPreprocessor(pre_cfg)
    raw_text = select_text(df, args.fields)
    logger.info("Cleaning %d articles (stop-words, lemmatise%s) ...", len(df),
                ", remove source tells" if pre_cfg.strip_leakage else "; source tells KEPT")
    df["clean_text"] = preprocessor.transform_series(raw_text)
    short = df["clean_text"].str.split().str.len() < MIN_TOKENS
    if short.any():
        logger.info("Dropping %d articles with fewer than %d tokens after cleaning.",
                    int(short.sum()), MIN_TOKENS)
    keep = ~short
    raw_text = raw_text[keep].reset_index(drop=True)
    df = df[keep].reset_index(drop=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_CACHE_PATH, index=False)
    balance = {config.LABEL_NAMES[k]: v for k, v in sorted(df["label"].value_counts().items())}
    logger.info("Class balance: %s", balance)

    # 3. Split ------------------------------------------------------------------------------
    train_idx, test_idx = train_test_split(df.index.to_numpy(), test_size=args.test_size,
                                           stratify=df["label"], random_state=args.seed)
    X_train, X_test = df.loc[train_idx, "clean_text"], df.loc[test_idx, "clean_text"]
    y_train, y_test = df.loc[train_idx, "label"], df.loc[test_idx, "label"]
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    fig_dir, figures = config.FIGURES_DIR, []
    if not args.no_plots:
        visualize.plot_class_distribution(df["label"], fig_dir)
        visualize.plot_length_distribution(raw_text, df["label"], fig_dir)
        if visualize.plot_subject_by_class(df["subject"], df["label"], fig_dir):
            figures.append("subject_by_class.png")
        freq_real = visualize.word_frequencies(X_train[y_train == 0])
        freq_fake = visualize.word_frequencies(X_train[y_train == 1])
        visualize.plot_top_words(freq_real, freq_fake, fig_dir)
        visualize.plot_wordclouds(freq_real, freq_fake, fig_dir)

    # 4. Train + evaluate every model ---------------------------------------------------------
    results, pipes, roc = run_experiments(X_train, y_train, X_test, y_test, args.models, args.cv_folds)
    has_cv = "cv_f1_mean" in results.columns

    # 5. Select the best model WITHOUT looking at the test set --------------------------------
    key = "cv_f1_mean" if has_cv else "f1"
    best_row = results.sort_values(key, ascending=False).iloc[0]
    best_kind, best_pipe = best_row["model"], pipes[best_row["model"]]
    saved_name, tuning_info = MODEL_NAMES[best_kind], None
    if args.tune:
        logger.info("Tuning %s ...", saved_name)
        search = GridSearchCV(
            build_pipeline(best_kind), PARAM_GRIDS[best_kind], scoring="f1",
            cv=StratifiedKFold(max(args.cv_folds, 3), shuffle=True, random_state=args.seed),
            n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        tuning_info = {"best_params": {k: str(v) for k, v in search.best_params_.items()},
                       "cv_f1": float(search.best_score_)}
        saved_name = MODEL_NAMES[best_kind] + " (tuned)"
        tuned = compute_metrics(y_test, best_pipe.predict(X_test), positive_scores(best_pipe, X_test))
        results = pd.concat([results, pd.DataFrame([{"model": best_kind, "model_name": saved_name, **tuned,
                                                     "fit_seconds": float("nan"),
                                                     "cv_f1_mean": float(search.best_score_)}])],
                            ignore_index=True)
        roc[saved_name] = (y_test, positive_scores(best_pipe, X_test))
        logger.info("Best params: %s", search.best_params_)

    y_pred = best_pipe.predict(X_test)
    y_score = positive_scores(best_pipe, X_test)
    p_fake = fake_probability(best_pipe, X_test)
    test_metrics = compute_metrics(y_test, y_pred, y_score)
    report = detailed_report(y_test, y_pred)
    logger.info("SAVED MODEL: %s | test accuracy=%.4f F1=%.4f", saved_name, test_metrics["accuracy"],
                test_metrics["f1"])

    # 6. Optional leakage ablation --------------------------------------------------------------
    ablation_df = None
    if args.leakage_ablation:
        logger.info("Running leakage ablation (%s) ...", MODEL_NAMES[best_kind])
        is_untuned = (results["model"] == best_kind) & ~results["model_name"].str.contains("tuned")
        untuned = results[is_untuned].iloc[0]
        ablation_df = leakage_ablation(raw_text, df["label"], df["subject"], train_idx, test_idx, best_kind,
                                       not args.no_lemmatize, untuned.to_dict())
        for r in ablation_df.itertuples():
            logger.info("  %-45s acc=%.4f f1=%.4f", r.setup, r.accuracy, r.f1)

    # 7. Persist artefacts ----------------------------------------------------------------------
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    table_md = results_markdown(results, saved_name, has_cv)
    (config.METRICS_DIR / "model_comparison.md").write_text(table_md + "\n", encoding="utf-8")
    ablation_md = ablation_markdown(ablation_df) if ablation_df is not None else None
    if ablation_df is not None:
        ablation_df.to_csv(config.METRICS_DIR / "leakage_ablation.csv", index=False)
        (config.METRICS_DIR / "leakage_ablation.md").write_text(ablation_md + "\n", encoding="utf-8")
    misclassified(raw_text[test_idx].reset_index(drop=True), y_test, y_pred, p_fake).to_csv(
        config.METRICS_DIR / "misclassified_examples.csv", index=False)

    summary = {
        "version": __version__, "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "seed": args.seed, "fields": args.fields, "preprocessing": preprocessor.to_dict(),
        "source_tells_removed": SOURCE_TELLS_DESCRIPTION if pre_cfg.strip_leakage else None,
        "best_model": saved_name, "tuning": tuning_info, "test_metrics": test_metrics, **report,
    }
    save_json(summary, config.METRICS_DIR / "best_model_metrics.json")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "preprocess_config": preprocessor.to_dict(),
                 "metadata": {"model": saved_name, "fields": args.fields, "test_metrics": test_metrics,
                              "trained_at": summary["trained_at"]}},
                config.BEST_MODEL_PATH, compress=3)
    logger.info("Model saved -> %s", config.BEST_MODEL_PATH)

    # 8. Evaluation figures -----------------------------------------------------------------------
    if not args.no_plots:
        visualize.plot_confusion_matrix(y_test, y_pred, f"Confusion matrix\n{saved_name}", fig_dir)
        visualize.plot_roc_curves(roc, fig_dir)
        visualize.plot_model_comparison(results, fig_dir)
        real_f, fake_f = top_features(best_pipe, n=20)
        visualize.plot_top_features(real_f, fake_f, f"Top terms learned by {saved_name}", fig_dir)
        figures = ["wordclouds_fake_vs_real.png", "top_words_fake_vs_real.png", "model_comparison.png",
                   "confusion_matrix.png", "roc_curves.png", "top_model_features.png"] + figures
        logger.info("Figures saved -> %s", fig_dir)

    if not args.no_update_readme and update_readme(table_md, summary, ablation_md, figures):
        logger.info("README.md results section updated.")

    print("\n" + table_md + f"\n\nSaved model: {saved_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
