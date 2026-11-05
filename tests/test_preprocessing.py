from fake_news_detection.preprocessing import PreprocessConfig, TextPreprocessor

pre = TextPreprocessor()
keep = TextPreprocessor(PreprocessConfig(strip_leakage=False))


def test_basic_cleaning():
    tokens = pre("The Senators were VOTING on <b>bills</b> at http://x.com").split()
    assert "the" not in tokens and "http" not in tokens and "b" not in tokens
    assert "senator" in tokens and "bill" in tokens and "vote" in tokens


def test_reuters_dateline_removed_by_default():
    out = pre("WASHINGTON (Reuters) - The Senate voted on Tuesday.").split()
    assert "reuters" not in out and "washington" not in out
    assert "senate" in out


def test_multi_city_dateline_removed():
    out = pre("BERLIN/PARIS (Reuters) - Talks continue.").split()
    assert "reuters" not in out and "berlin" not in out and "paris" not in out


def test_other_source_tells_removed():
    out = pre("Big news. Featured image via Getty Images pic.twitter.com/abc123 (VIDEO) 21st Century Wire")
    assert out.split() == ["big", "news"]


def test_leakage_can_be_kept_for_demonstration():
    assert "reuters" in keep("WASHINGTON (Reuters) - Talks continue.").split()


def test_ordinary_use_of_words_is_not_damaged():
    # only the listed source tells are removed, normal vocabulary stays
    assert "washington" in pre("Officials in Washington said the plan failed.").split()


def test_lemmatization_can_be_disabled():
    raw = TextPreprocessor(PreprocessConfig(lemmatize=False))
    assert "bills" in raw("new bills").split()


def test_non_string_is_safe():
    assert pre(None) == ""  # type: ignore[arg-type]
    assert pre("") == ""
