"""Generate SYNTHETIC Fake.csv / True.csv (Kaggle format) for smoke tests and CI.

Columns: title, text, subject, date. The text is template based and deliberately imitates the
dataset's source tells (Reuters datelines for REAL, 'Featured image via' for FAKE) so the leakage
handling can be tested. Never report results obtained on this data.

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

CITIES = ["WASHINGTON", "LONDON", "BERLIN", "NEW YORK", "BEIJING", "PARIS", "MOSCOW", "TOKYO"]
REAL_TOPICS = ["the budget bill", "trade talks", "the tax proposal", "the infrastructure plan",
               "central bank rates", "the defense agreement", "the healthcare measure", "the census report"]
REAL_VERBS = ["said on Tuesday", "told reporters", "announced", "confirmed", "proposed", "agreed", "reported"]
REAL_ACTORS = ["The Senate", "A spokesman", "The finance minister", "Officials", "The committee", "Lawmakers"]
FAKE_HOOKS = ["BREAKING", "SHOCKING", "EXPOSED", "UNBELIEVABLE", "WATCH", "BOMBSHELL", "HILARIOUS"]
FAKE_SUBJ = ["the elites", "the mainstream media", "liberal snowflakes", "the deep state",
             "a corrupt politician", "Hollywood hypocrites", "the globalists"]
FAKE_ACT = ["DESTROYS", "EXPOSES", "SECRETLY plots against", "LIES about", "PANICS over", "MELTS DOWN over"]
FAKE_OBJ = ["the truth", "your freedom", "everything you know", "the election", "the border wall", "patriots"]
FAKE_SUBJECTS = ["News", "politics", "left-news", "Government News", "US_News", "Middle-east"]
REAL_SUBJECTS = ["politicsNews", "worldnews"]


def real_article(rng: random.Random):
    topic = rng.choice(REAL_TOPICS)
    title = f"{rng.choice(REAL_ACTORS)} {rng.choice(REAL_VERBS).split()[0]} on {topic}"
    body = " ".join(
        f"{rng.choice(REAL_ACTORS)} {rng.choice(REAL_VERBS)} that {rng.choice(REAL_TOPICS)} would proceed "
        f"after review, according to a statement." for _ in range(rng.randint(2, 4)))
    text = f"{rng.choice(CITIES)} (Reuters) - {body}"
    return title, text, rng.choice(REAL_SUBJECTS)


def fake_article(rng: random.Random):
    who, act, obj = rng.choice(FAKE_SUBJ).title(), rng.choice(FAKE_ACT), rng.choice(FAKE_OBJ)
    title = f"{rng.choice(FAKE_HOOKS)}: {who} {act} {obj}!!!"
    body = " ".join(
        f"You will NOT believe how {rng.choice(FAKE_SUBJ)} {rng.choice(FAKE_ACT).lower()} "
        f"{rng.choice(FAKE_OBJ)} "
        f"- they do not want you to know this, share before it is deleted!" for _ in range(rng.randint(2, 4)))
    tail = rng.choice(["", "", " Featured image via Getty Images", " 21st Century Wire says this is big.",
                       " Read more: pic.twitter.com/abc123", " (VIDEO)"])
    return title, body + tail, rng.choice(FAKE_SUBJECTS)


def write(path: Path, maker, n: int, seed: int) -> None:
    rng = random.Random(seed)
    seen, rows = set(), []
    while len(rows) < n:
        title, text, subject = maker(rng)
        if (title, text) in seen:
            continue
        seen.add((title, text))
        rows.append((title, text, subject, f"January {rng.randint(1, 28)}, 2017"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["title", "text", "subject", "date"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} synthetic articles -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--n", type=int, default=250, help="Articles per class.")
    args = parser.parse_args()
    write(args.out_dir / "Fake.csv", fake_article, args.n, seed=1)
    write(args.out_dir / "True.csv", real_article, args.n, seed=2)


if __name__ == "__main__":
    main()
