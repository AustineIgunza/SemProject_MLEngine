"""out/categoriser.pkl -> out/model.json.

Writes everything Multinomial Naive Bayes inference needs, so the Akiba
Express server can classify in-process with no Python and no extra dependency.
Copy out/model.json (and out/parity.json) into the app's
server/src/modules/categoriser/ to ship a retrained model.

Usage:  python export.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import AMOUNT_BUCKETS, NGRAM_RANGE, to_document

ROOT = Path(__file__).parent
PKL = ROOT / "out" / "categoriser.pkl"
TRAIN = ROOT / "data" / "train.csv"
OUT = ROOT / "out" / "model.json"
PARITY = ROOT / "out" / "parity.json"

MAX_BYTES = 2 * 1024 * 1024
DECIMALS = 4


def build(vocab: dict, log_prob: np.ndarray, pipe) -> dict:
    clf = pipe.named_steps["clf"]
    return {
        "version": "1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alpha": float(clf.alpha),
        "classes": [str(c) for c in clf.classes_],
        "class_log_prior": np.round(clf.class_log_prior_, DECIMALS).tolist(),
        "vocabulary": vocab,
        "feature_log_prob": np.round(log_prob, DECIMALS).tolist(),
        "ngram_range": list(NGRAM_RANGE),
        "amount_buckets": [[lo, hi, tok] for lo, hi, tok in AMOUNT_BUCKETS],
    }


def main() -> None:
    pipe = joblib.load(PKL)
    vocab = {str(t): int(i) for t, i in pipe.named_steps["vec"].vocabulary_.items()}
    log_prob = pipe.named_steps["clf"].feature_log_prob_

    model = build(vocab, log_prob, pipe)
    payload = json.dumps(model, separators=(",", ":"))

    if len(payload.encode()) > MAX_BYTES:
        # Prune terms that barely discriminate between classes: if a term's
        # log-prob is nearly the same in every class it cannot move the argmax.
        spread = log_prob.max(axis=0) - log_prob.min(axis=0)
        threshold = 0.1
        while True:
            keep = np.flatnonzero(spread >= threshold)
            terms = sorted(((t, i) for t, i in vocab.items() if i in set(keep)), key=lambda x: x[1])
            new_vocab = {t: n for n, (t, _) in enumerate(terms)}
            pruned = build(new_vocab, log_prob[:, [i for _, i in terms]], pipe)
            payload = json.dumps(pruned, separators=(",", ":"))
            if len(payload.encode()) <= MAX_BYTES:
                break
            threshold *= 1.5
        print(f"Pruned {len(vocab) - len(new_vocab)} of {len(vocab)} terms (spread < {threshold:.3f})")

    OUT.write_text(payload, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}  ({len(payload.encode()) / 1024:.1f} KB, "
          f"{len(json.loads(payload)['vocabulary'])} terms, {len(model['classes'])} classes)")
    write_parity_fixture(pipe)


def write_parity_fixture(pipe) -> None:
    """sklearn's own predictions on sample rows, for the TS parity test.

    categoriser.test.ts runs the same rows through the TypeScript inference and
    asserts the probabilities match — proof that features.py and text.ts agree.
    """
    sample = pd.read_csv(TRAIN).sample(n=25, random_state=7)
    docs = [to_document(d, a) for d, a in zip(sample["description"], sample["amount"])]
    proba = pipe.predict_proba(docs)
    classes = [str(c) for c in pipe.classes_]
    rows = [
        {"description": d, "amount": float(a), "probabilities": dict(zip(classes, np.round(p, 4).tolist()))}
        for d, a, p in zip(sample["description"], sample["amount"], proba)
    ]
    PARITY.parent.mkdir(exist_ok=True)
    PARITY.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"Wrote {PARITY.relative_to(ROOT)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
