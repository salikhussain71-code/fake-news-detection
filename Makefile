.PHONY: install install-dev train train-fast ablation tune train-sample test lint predict clean

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Full run on Kaggle Fake.csv + True.csv (data/raw/)
train:
	python -m fake_news_detection.train

# Also measure how much accuracy depends on source "tells" (Reuters datelines, etc.)
ablation:
	python -m fake_news_detection.train --leakage-ablation

tune:
	python -m fake_news_detection.train --tune

# Smoke run on the bundled synthetic sample (never report these numbers)
train-sample:
	python -m fake_news_detection.train --fake data/sample/Fake.csv --true data/sample/True.csv --no-update-readme

test:
	pytest

lint:
	ruff check .

predict:
	python -m fake_news_detection.predict --explain "BREAKING: You won't BELIEVE what the elites are hiding from you!!!"

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
