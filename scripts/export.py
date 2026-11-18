"""out/categoriser.pkl -> out/model.json and out/parity.json.

model.json holds everything Naive Bayes inference needs, so predict.py can
classify an expense without scikit-learn:

  classes            the Akiba categories the model can predict
  class_log_prior    log P(category), one value per class
  features           the 6 feature names, in one-hot column order
  categories         for each feature, the values seen in training (one column each)
  feature_log_prob   log P(column | category), one row per class
  amount_bin_edges   bin edges in BYN; a bin includes its upper edge and the
  amount_bin_labels  last bin (above the last edge) has no upper limit

parity.json holds 25 real rows with scikit-learn's predicted probabilities, so
`python scripts/predict.py --check` can show that the pure-Python inference
gives the same answers.

Usage:  python scripts/export.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from features import AMOUNT_BIN_EDGES, AMOUNT_BIN_LABELS, FEATURES, build_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Expenses_clean.csv"
PKL = ROOT / "out" / "categoriser.pkl"
MODEL = ROOT / "out" / "model.json"
PARITY = ROOT / "out" / "parity.json"


def main() -> None:
    pipe = joblib.load(PKL)
    onehot, nb = pipe.named_steps["onehot"], pipe.named_steps["model"]
    assert list(onehot.feature_names_in_) == FEATURES

    categories = {name: values.tolist() for name, values in zip(FEATURES, onehot.categories_)}
    assert sum(len(v) for v in categories.values()) == nb.feature_log_prob_.shape[1]

    model = {
        "version": "2.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "MultinomialNB",
        "alpha": float(nb.alpha),
        "classes": nb.classes_.tolist(),
        "class_log_prior": nb.class_log_prior_.tolist(),
        "features": FEATURES,
        "categories": categories,
        "feature_log_prob": nb.feature_log_prob_.tolist(),
        "amount_bin_edges": AMOUNT_BIN_EDGES[:-1],  # drop inf, which JSON cannot store
        "amount_bin_labels": AMOUNT_BIN_LABELS,
    }
    MODEL.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {MODEL.relative_to(ROOT)}  ({len(model['classes'])} classes, "
          f"{nb.feature_log_prob_.shape[1]} one-hot columns from {len(FEATURES)} features)")
    write_parity(pipe)


def write_parity(pipe) -> None:
    """scikit-learn's own probabilities for 25 real rows, for predict.py --check."""
    raw = pd.read_csv(DATA)
    sample = raw[raw["amount"] > 0].sample(n=25, random_state=7)
    X, y = build_features(sample)
    proba = pipe.predict_proba(X)
    classes = pipe.classes_.tolist()
    rows = [
        {"date": when[:10], "amount": float(amount), "account": account, "tag": tag,
         "true_category": label, "predicted": classes[p.argmax()],
         "probabilities": dict(zip(classes, [round(float(v), 6) for v in p]))}
        for when, amount, account, tag, label, p in zip(
            sample["date_time"], sample["amount"], sample["account"], sample["tags"], y, proba)
    ]
    PARITY.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {PARITY.relative_to(ROOT)}  ({len(rows)} real rows)")


if __name__ == "__main__":
    main()
