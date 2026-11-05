# Data

Large datasets are **not committed** to Git (see `.gitignore`). Follow one of the options below.

## Recommended dataset: Fake and Real News Dataset (Kaggle)

* Kaggle page: <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>
* Two files: `Fake.csv` (fake articles) and `True.csv` (real articles, collected from Reuters)
* Columns: `title`, `text`, `subject`, `date`
* Roughly 23k fake and 21k real articles, mostly US political news from 2016-2018

### Option A - manual download
1. Open the Kaggle link and click **Download**.
2. Unzip and place **both** files here:
   ```
   data/raw/Fake.csv
   data/raw/True.csv
   ```

### Option B - Kaggle CLI
```bash
pip install kaggle                       # put your API token in ~/.kaggle/kaggle.json
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
```

Then run `python -m fake_news_detection.train`.

## Known dataset problems (handled by this project)

| Problem | What the pipeline does |
|---|---|
| Real articles start with `CITY (Reuters) -` | Removed by default (`strip_leakage`), see `docs/methodology.md` |
| Fake articles contain tells such as `Featured image via`, `Getty Images`, `21st Century Wire`, `pic.twitter.com` | Removed by default |
| The `subject` column almost perfectly separates the classes | **Not used as a feature**; plotted as a leakage warning (`subject_by_class.png`) |
| Many exact duplicate articles | Dropped **before** the train/test split |
| Identical text labelled both real and fake | Dropped (ambiguous) |
| Some articles have an empty body | Title is still used; articles with fewer than 3 tokens after cleaning are dropped |

## Using another dataset

Any CSV containing both classes works, e.g. a single file with a label column:

```bash
# label value 1 means FAKE
python -m fake_news_detection.train --data articles.csv --label-col label --text-col text --title-col title

# WELFake-style files use 0 = fake:
python -m fake_news_detection.train --data WELFake_Dataset.csv --label-col label --text-col text \
    --title-col title --fake-label 0
```
`--fake-label` is the value of the label column that means FAKE (also accepts words such as `fake`).
Use `--title-col ""` if there is no title column.

## Bundled sample (`data/sample/`)

`Fake.csv` and `True.csv` (250 rows each) are **synthetic**, template-generated articles in the Kaggle
format, created by `scripts/generate_sample_data.py`. They exist only for unit tests, CI and a quick
smoke run (`make train-sample`). **Never report results obtained on them.**

## Folders

| Path | Purpose | Tracked by Git |
|---|---|---|
| `data/raw/` | Original Kaggle downloads | no |
| `data/processed/` | `clean_articles.csv` written by the pipeline | no |
| `data/sample/` | Synthetic smoke-test data | yes |
