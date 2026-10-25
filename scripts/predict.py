"""Classify a transaction with the exported model — no sklearn required.

Loads out/model.json (the same file the TypeScript runtime uses) and runs
Multinomial Naive Bayes inference directly, so you can try the shipped model
on any description without retraining.

Usage:
    python scripts/predict.py "naivas westlands" 1240
    python scripts/predict.py              # interactive mode
"""

import json
import math
import sys
from pathlib import Path

MODEL = Path(__file__).resolve().parent.parent / "out" / "model.json"


def load():
    if not MODEL.exists():
        raise SystemExit(f"{MODEL} not found. Run `python scripts/train.py && python scripts/export.py` first.")
    return json.loads(MODEL.read_text(encoding="utf-8"))


def to_tokens(description: str, amount: float, model: dict) -> list[str]:
    """Mirrors features.py: clean -> amount bucket -> unigrams + bigrams."""
    text = "".join(ch if "a" <= ch <= "z" or ch.isspace() else " " for ch in description.lower())
    words = text.split()

    value = abs(float(amount or 0))
    bucket = model["amount_buckets"][-1][2]
    for low, high, token in model["amount_buckets"]:
        if high is None or value < high:
            bucket = token
            break
    words.append(bucket)

    lo, hi = model["ngram_range"]
    tokens = []
    for n in range(lo, hi + 1):
        tokens += [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return tokens


def predict(description: str, amount: float, model: dict):
    vocab, log_prob = model["vocabulary"], model["feature_log_prob"]
    scores = list(model["class_log_prior"])
    hits = []
    for token in to_tokens(description, amount, model):
        idx = vocab.get(token)
        if idx is not None:
            hits.append(token)
            for c in range(len(scores)):
                scores[c] += log_prob[c][idx]

    top = max(scores)
    exp = [math.exp(s - top) for s in scores]
    total = sum(exp)
    probs = sorted(zip(model["classes"], [e / total for e in exp]),
                   key=lambda kv: kv[1], reverse=True)
    return probs, hits


def show(description: str, amount: float, model: dict) -> None:
    probs, hits = predict(description, amount, model)
    label, confidence = probs[0]
    print(f'\n  "{description}"   KES {amount:,.0f}')
    print(f"  -> {label}   ({confidence:.1%} confident)\n")
    for name, p in probs[:4]:
        bar = "#" * round(p * 36)
        print(f"    {name:<16}{p:6.1%}  {bar}")
    known = [h for h in hits if " " not in h and not h.startswith("amt_")]
    print(f"\n    matched terms: {', '.join(known) if known else '(none — unknown merchant)'}")
    if confidence < 0.60:
        print("    LOW CONFIDENCE — this one should be confirmed by the user.")


def main() -> None:
    model = load()
    print(f"Model v{model['version']}  trained {model['trained_at']}  "
          f"{len(model['vocabulary']):,} terms  {len(model['classes'])} classes")

    if len(sys.argv) >= 2:
        show(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0, model)
        return

    print("Enter a description and amount. Blank line to quit.")
    while True:
        try:
            line = input("\ndescription> ").strip()
        except EOFError:
            break
        if not line:
            break
        try:
            amount = float(input("amount (KES)> ").strip() or 0)
        except ValueError:
            amount = 0
        show(line, amount, model)


if __name__ == "__main__":
    main()
