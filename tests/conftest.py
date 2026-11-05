import pytest

from fake_news_detection import config


@pytest.fixture(scope="session")
def sample_paths():
    assert config.SAMPLE_FAKE_PATH.exists(), "Run scripts/generate_sample_data.py"
    return config.SAMPLE_FAKE_PATH, config.SAMPLE_TRUE_PATH
