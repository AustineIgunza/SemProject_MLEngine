"""Suggest an Akiba category for one expense with out/model.json. No scikit-learn needed.

Pure Python: it builds the same 6 features as features.py and runs Multinomial
Naive Bayes on the probabilities exported by export.py.

Usage:
    python scripts/predict.py --amount 5 --date 2025-06-14 --account acct_1 --tag tag_1
    python scripts/predict.py --check     # compare with scikit-learn on out/parity.json
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "out" / "model.json"
PARITY = ROOT / "out" / "parity.json"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LOW_CONFIDENCE = 0.6
TOLERANCE = 1e-4


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run python scripts/train.py and python scripts/export.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def features(amount: float, day: date, account: str, tag: str, model: dict) -> dict:
    """The same 6 features as build_features() in features.py."""
    edges, labels = model["amount_bin_edges"], model["amount_bin_labels"]
    amount_bin = next((label for label, upper in zip(labels, edges[1:]) if amount <= upper), labels[-1])
    return {
        "amount_bin": amount_bin,
        "day_of_week": DAY_NAMES[day.weekday()],
        "month": day.month,
        "is_weekend": int(day.weekday() >= 5),
        "account": account,
        "tags": tag,
    }


def predict(values: dict, model: dict) -> tuple[list, list]:
    """Return [(category, probability)] from most to least likely, and the unseen features."""
    scores = list(model["class_log_prior"])
    column, unseen = 0, []
    for name in model["features"]:
        seen = model["categories"][name]
        if values[name] in seen:
            j = column + seen.index(values[name])
            scores = [s + row[j] for s, row in zip(scores, model["feature_log_prob"])]
        else:
            unseen.append(name)  # like OneHotEncoder(handle_unknown="ignore")
        column += len(seen)

    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    probs = [(c, w / total) for c, w in zip(model["classes"], weights)]
    return sorted(probs, key=lambda cp: cp[1], reverse=True), unseen


def show(args, model: dict) -> None:
    day = date.fromisoformat(args.date)
    values = features(args.amount, day, args.account, args.tag, model)
    probs, unseen = predict(values, model)
    category, confidence = probs[0]

    print(f"Expense:  {args.amount:g} BYN on {DAY_NAMES[day.weekday()]} {day}, {args.account}, {args.tag}")
    print("Features:", ", ".join(f"{k}={v}" for k, v in values.items()))
    if unseen:
        print("Not seen in training, so ignored:", ", ".join(unseen))
    print(f"\nPredicted category: {category} ({confidence:.1%} confident)")
    if confidence < LOW_CONFIDENCE:
        print("Low confidence: Akiba should ask the student to confirm the category.")
    print("\nTop 4 alternatives:")
    for name, p in probs[1:5]:
        print(f"  {name:<15}{p:6.1%}")


def check(model: dict) -> None:
    rows = load(PARITY)
    worst, label_matches = 0.0, 0
    for row in rows:
        values = features(row["amount"], date.fromisoformat(row["date"]), row["account"], row["tag"], model)
        probs, _ = predict(values, model)
        mine = dict(probs)
        worst = max(worst, max(abs(mine[c] - p) for c, p in row["probabilities"].items()))
        label_matches += probs[0][0] == row["predicted"]

    passed = worst <= TOLERANCE and label_matches == len(rows)
    print(f"Parity with scikit-learn on {len(rows)} real rows: largest probability difference "
          f"{worst:.1e} (tolerance {TOLERANCE:g}), same prediction on {label_matches}/{len(rows)}")
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--amount", type=float, help="amount in BYN, above 0")
    parser.add_argument("--date", help="date of the expense, YYYY-MM-DD")
    parser.add_argument("--account", help="account, e.g. acct_1")
    parser.add_argument("--tag", help="tag, e.g. tag_1")
    parser.add_argument("--check", action="store_true", help="compare with out/parity.json")
    args = parser.parse_args()

    model = load(MODEL)
    if args.check:
        check(model)
        return
    if None in (args.amount, args.date, args.account, args.tag):
        parser.error("--amount, --date, --account and --tag are all required")
    if args.amount <= 0:
        parser.error("--amount must be above 0")
    try:
        date.fromisoformat(args.date)
    except ValueError:
        parser.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
    show(args, model)


if __name__ == "__main__":
    main()
