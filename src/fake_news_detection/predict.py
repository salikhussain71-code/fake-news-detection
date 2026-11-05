"""Inference with explanations.

    python -m fake_news_detection.predict "Article text ..."
    python -m fake_news_detection.predict --title "Headline" --text "Body ..." --explain
    python -m fake_news_detection.predict --file articles.txt        # one article per line

IMPORTANT: the model recognises *writing patterns* learned from its training data. It does NOT check
facts, so treat the output as a signal, not a verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from . import config
from .models import explain_prediction, fake_probability
from .preprocessing import PreprocessConfig, TextPreprocessor

DISCLAIMER = ("Note: this model detects writing patterns of the training data; "
              "it does not verify facts.")


class FakeNewsDetector:
    """Loads the bundle written by ``train`` and classifies raw article text."""

    def __init__(self, model_path: str | Path = config.BEST_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at '{model_path}'. Run `python -m fake_news_detection.train` first.")
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.metadata = bundle.get("metadata", {})
        self.preprocessor = TextPreprocessor(PreprocessConfig(**bundle["preprocess_config"]))

    def predict(self, texts: list[str], explain: bool = False, n_terms: int = 5) -> list[dict]:
        clean = self.preprocessor.transform(texts)
        p_fake = fake_probability(self.pipeline, clean)
        results = []
        for text, c, p in zip(texts, clean, p_fake):
            is_fake = bool(p >= 0.5)
            item = {"text": text, "label": config.LABEL_NAMES[int(is_fake)], "p_fake": float(p),
                    "confidence": float(p if is_fake else 1 - p)}
            if explain:
                fake_ev, real_ev = explain_prediction(self.pipeline, c, n_terms)
                item["fake_evidence"], item["real_evidence"] = fake_ev, real_ev
            results.append(item)
        return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify news articles as REAL or FAKE.")
    parser.add_argument("text", nargs="*", help="Article text(s) to classify.")
    parser.add_argument("--title", help="Optional headline, prepended to a single --text / text.")
    parser.add_argument("--text", dest="body", help="Article body (use with --title).")
    parser.add_argument("--file", type=Path, help="Text file with one article per line.")
    parser.add_argument("--explain", action="store_true", help="Show the terms that drove the decision.")
    parser.add_argument("--model", type=Path, default=config.BEST_MODEL_PATH)
    args = parser.parse_args(argv)

    try:
        detector = FakeNewsDetector(args.model)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    texts = list(args.text)
    if args.body:
        texts.append(args.body)
    if args.title and texts:
        texts[-1] = f"{args.title}. {texts[-1]}"
    elif args.title:
        texts.append(args.title)
    if args.file:
        lines = args.file.read_text(encoding="utf-8").splitlines()
        texts += [ln.strip() for ln in lines if ln.strip()]
    if not texts:
        print("Enter an article (empty line to quit):")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            texts.append(line)

    for r in detector.predict(texts, explain=args.explain):
        snippet = r["text"] if len(r["text"]) <= 80 else r["text"][:77] + "..."
        print(f"[{r['label']} | confidence {r['confidence']:.1%}] {snippet}")
        if args.explain:
            if r["fake_evidence"]:
                print("    pushes towards FAKE:", ", ".join(f"{t} ({w:+.2f})" for t, w in r["fake_evidence"]))
            if r["real_evidence"]:
                print("    pushes towards REAL:", ", ".join(f"{t} ({w:+.2f})" for t, w in r["real_evidence"]))
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
