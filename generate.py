"""Bootstrap training set: lexicon.json -> data/train.csv.

The system has no users yet, so there is no labelled data to learn from. This
generator expands the hand-built merchant lexicon into realistic M-Pesa-style
descriptions. The model is then evaluated on REAL labelled statements
(data/test_real.csv), which tells us whether this bootstrap generalises.

Each row also carries the source `keyword`, so train.py can cross-validate with
GroupKFold — a fold never sees a merchant it is tested on, which keeps the CV
score from being inflated by near-duplicate rows.

Usage:  python generate.py
"""

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
LEXICON = ROOT / "lexicon.json"
OUT = ROOT / "data" / "train.csv"

ROWS_PER_CATEGORY = 600  # ~12 rows per keyword at ~50 keywords — every template used
MIN_ROWS_PER_KEYWORD = 6

# Plausible KES ranges per category, sampled log-uniformly so small amounts are
# more common than large ones (how student spending actually looks). Ranges are
# deliberately wide where a category mixes cheap and expensive items — Rent
# covers KPLC tokens and water as well as monthly rent; Academic covers
# printing as well as fees.
AMOUNT_RANGES = {
    "Food": (50, 5000),
    "Rent": (300, 45000),
    "Transport": (50, 3000),
    "Academic": (20, 60000),
    "Airtime & Data": (10, 3000),
    "Entertainment": (100, 5000),
    "Health": (100, 10000),
    "Personal Care": (100, 5000),
    "Other": (50, 10000),
}

PLACES = [
    "westlands", "kilimani", "ngong road", "rongai", "juja", "thika road",
    "cbd", "karen", "langata", "kasarani", "ruiru",
]

# Patterns seen in real M-Pesa statement "Details" columns. Every template is
# used for every category, so template words ("pay bill", "ltd") carry no
# category signal — only the keyword does.
TEMPLATES = [
    "{kw}",
    "{kw} ltd",
    "{kw} limited",
    "pay bill to {kw}",
    "buy goods {kw}",
    "{kw} branch {place}",
    "merchant payment to {kw}",
    "customer transfer to {kw}",
    "pay bill online to {kw}",
    "{kw} {place}",
    "merchant payment to {kw} {place}",
    "lipa na mpesa {kw}",
]


def sample_amount(category: str) -> int:
    low, high = AMOUNT_RANGES[category]
    return int(round(math.exp(random.uniform(math.log(low), math.log(high)))))


def main() -> None:
    random.seed(42)
    lexicon: dict[str, list[str]] = json.loads(LEXICON.read_text(encoding="utf-8"))

    empty = [c for c, kws in lexicon.items() if not kws]
    if empty:
        raise SystemExit(f"lexicon.json has empty categories: {', '.join(empty)}")
    unknown = set(lexicon) - set(AMOUNT_RANGES)
    if unknown:
        raise SystemExit(f"No amount range for: {', '.join(sorted(unknown))}")
    seen: dict[str, str] = {}
    for category, keywords in lexicon.items():
        for kw in keywords:
            key = kw.strip().lower()
            if key in seen and seen[key] != category:
                raise SystemExit(f'"{kw}" is listed under both {seen[key]} and {category}')
            seen[key] = category

    rows = []
    for category, keywords in lexicon.items():
        per_keyword = max(MIN_ROWS_PER_KEYWORD, math.ceil(ROWS_PER_CATEGORY / len(keywords)))
        for kw in keywords:
            kw = kw.strip().lower()
            # Cycle through a shuffled template list so each keyword gets the
            # full variety before any template repeats.
            templates = random.sample(TEMPLATES, len(TEMPLATES))
            for i in range(per_keyword):
                template = templates[i % len(templates)]
                description = template.format(kw=kw, place=random.choice(PLACES))
                rows.append((description, sample_amount(category), category, kw))

    random.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "amount", "category", "keyword"])
        writer.writerows(rows)

    counts = Counter(r[2] for r in rows)
    print(f"Wrote {len(rows)} rows to {OUT.relative_to(ROOT)}")
    for category in lexicon:
        print(f"  {category:<16} {counts[category]:>5}  ({len(lexicon[category])} keywords)")


if __name__ == "__main__":
    main()
