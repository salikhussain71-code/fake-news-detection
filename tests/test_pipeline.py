import json

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from fake_news_detection import config, train
from fake_news_detection.data import load_dataset
from fake_news_detection.models import (
    MODEL_NAMES,
    build_pipeline,
    explain_prediction,
    fake_probability,
    top_features,
)
from fake_news_detection.predict import FakeNewsDetector
from fake_news_detection.preprocessing import TextPreprocessor


@pytest.fixture(scope="module")
def split(sample_paths):
    df = load_dataset(*sample_paths)
    df["clean"] = TextPreprocessor().transform_series(df["text"])
    return train_test_split(df["clean"], df["label"], test_size=0.25, stratify=df["label"], random_state=0)


@pytest.mark.parametrize("model", list(MODEL_NAMES))
def test_every_model_trains_and_scores(split, model):
    X_tr, X_te, y_tr, y_te = split
    pipe = build_pipeline(model).fit(X_tr, y_tr)
    assert (pipe.predict(X_te) == y_te).mean() > 0.9
    p = fake_probability(pipe, X_te)
    assert p.shape == (len(X_te),) and np.all((p >= 0) & (p <= 1))
    real, fake = top_features(pipe, n=5)
    assert len(real) == len(fake) == 5


def test_explain_prediction_points_the_right_way(split):
    X_tr, _, y_tr, _ = split
    pipe = build_pipeline("lr").fit(X_tr, y_tr)
    clean = TextPreprocessor()("Share this shocking truth before it is deleted")
    fake_ev, real_ev = explain_prediction(pipe, clean)
    assert fake_ev and all(w > 0 for _, w in fake_ev)
    assert all(w < 0 for _, w in real_ev)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_pipeline("bert")


def _isolate(monkeypatch, tmp_path):
    root = config.PROJECT_ROOT
    for name in ("PROCESSED_DIR", "CLEAN_CACHE_PATH", "MODELS_DIR", "BEST_MODEL_PATH", "REPORTS_DIR",
                 "FIGURES_DIR", "METRICS_DIR", "README_PATH"):
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).relative_to(root))
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)


def test_end_to_end_with_ablation_and_predict(tmp_path, monkeypatch, sample_paths):
    _isolate(monkeypatch, tmp_path)
    body = f"# t\n{train.RESULTS_START}\nPLACEHOLDER_TEXT\n{train.RESULTS_END}\n"
    (tmp_path / "README.md").write_text(body, encoding="utf-8")

    code = train.main(["--fake", str(sample_paths[0]), "--true", str(sample_paths[1]),
                       "--cv-folds", "3", "--leakage-ablation"])
    assert code == 0
    m = json.loads((tmp_path / "reports/metrics/best_model_metrics.json").read_text())
    assert m["test_metrics"]["accuracy"] > 0.9
    for f in ("confusion_matrix.png", "wordclouds_fake_vs_real.png", "subject_by_class.png"):
        assert (tmp_path / "reports/figures" / f).exists()
    assert (tmp_path / "reports/metrics/leakage_ablation.csv").exists()
    readme = (tmp_path / "README.md").read_text()
    assert "PLACEHOLDER_TEXT" not in readme and "leakage" in readme.lower()

    detector = FakeNewsDetector(tmp_path / "models/fake_news_model.joblib")
    out = detector.predict(["SHOCKING: the elites LIE about the truth!!! Share before it is deleted",
                            "WASHINGTON (Reuters) - The Senate said on Tuesday the bill would proceed."],
                           explain=True)
    assert [o["label"] for o in out] == ["FAKE", "REAL"]
    assert all(0.5 <= o["confidence"] <= 1 for o in out)
    assert out[0]["fake_evidence"]


def test_tuning_and_title_only(tmp_path, monkeypatch, sample_paths):
    _isolate(monkeypatch, tmp_path)
    code = train.main(["--fake", str(sample_paths[0]), "--true", str(sample_paths[1]), "--models", "nb",
                       "--fields", "title", "--tune", "--no-plots", "--no-update-readme"])
    assert code == 0
    m = json.loads((tmp_path / "reports/metrics/best_model_metrics.json").read_text())
    assert m["tuning"]["best_params"] and m["fields"] == "title"


def test_missing_dataset_returns_error_code(tmp_path):
    assert train.main(["--fake", str(tmp_path / "a.csv"), "--true", str(tmp_path / "b.csv")]) == 1
