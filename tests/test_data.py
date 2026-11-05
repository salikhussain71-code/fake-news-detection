import pandas as pd
import pytest

from fake_news_detection.data import load_dataset, load_single_csv, select_text


def _write(path, rows):
    pd.DataFrame(rows, columns=["title", "text", "subject", "date"]).to_csv(path, index=False)


def test_load_sample(sample_paths):
    df = load_dataset(*sample_paths)
    assert set(df["label"]) == {0, 1}
    assert df["label"].value_counts().min() > 100
    assert df["text"].is_unique
    assert {"title", "body", "text", "subject", "label"} == set(df.columns)


def test_labels_fake_is_one(tmp_path):
    _write(tmp_path / "Fake.csv", [("f1", "fake body", "News", "d")])
    _write(tmp_path / "True.csv", [("t1", "true body", "worldnews", "d")])
    df = load_dataset(tmp_path / "Fake.csv", tmp_path / "True.csv").set_index("title")
    assert df.loc["f1", "label"] == 1 and df.loc["t1", "label"] == 0


def test_duplicates_empty_and_conflicts_removed(tmp_path):
    _write(tmp_path / "Fake.csv", [("a", "same", "x", "d"), ("a", "same", "x", "d"), ("", "", "x", "d"),
                                   ("c", "conflict", "x", "d"), ("f", "unique fake", "x", "d")])
    _write(tmp_path / "True.csv", [("c", "conflict", "y", "d"), ("t", "unique true", "y", "d")])
    df = load_dataset(tmp_path / "Fake.csv", tmp_path / "True.csv")
    assert sorted(df["title"]) == ["a", "f", "t"]  # empty row, duplicate and conflicting pair dropped


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_dataset(tmp_path / "Fake.csv", tmp_path / "True.csv")


def test_missing_column_message(tmp_path):
    pd.DataFrame({"headline": ["a"], "body": ["b"]}).to_csv(tmp_path / "Fake.csv", index=False)
    pd.DataFrame({"headline": ["a"], "body": ["b"]}).to_csv(tmp_path / "True.csv", index=False)
    with pytest.raises(KeyError, match="title"):
        load_dataset(tmp_path / "Fake.csv", tmp_path / "True.csv")


def test_single_csv_with_custom_fake_label(tmp_path):
    f = tmp_path / "s.csv"
    pd.DataFrame({"label": [0, 1, 1], "text": ["fake one", "real one", "real two"]}).to_csv(f, index=False)
    df = load_single_csv(f, "label", "text", None, fake_label="0")  # WELFake style: 0 = fake
    assert df.sort_values("text")["label"].tolist() == [1, 0, 0]


def test_single_csv_wrong_column(tmp_path):
    f = tmp_path / "s.csv"
    pd.DataFrame({"a": [1], "b": ["x"]}).to_csv(f, index=False)
    with pytest.raises(KeyError, match="--label-col"):
        load_single_csv(f, "label", "text")


def test_select_text_fields(sample_paths):
    df = load_dataset(*sample_paths)
    assert select_text(df, "title").equals(df["title"])
    assert select_text(df, "body").equals(df["body"])
    assert select_text(df, "both").equals(df["text"])
    with pytest.raises(ValueError):
        select_text(df, "headline")
