"""Train on the lexicon bootstrap; evaluate on real labelled rows when available.

  train: data/train.csv      (generate.py)
  test:  data/test_real.csv  (optional — hand-labelled M-Pesa rows, never used
                              for fitting or tuning; alpha is tuned by CV on train)

Ships CountVectorizer + MultinomialNB: integer counts port to TypeScript in a
few lines. TF-IDF + LogReg is trained only as a comparison and its gap is
recorded in metrics.json so it can be reported honestly.

Usage:  python train.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from features import to_document, vectorizer_kwargs

ROOT = Path(__file__).parent
TRAIN, TEST, OUT = ROOT / "data/train.csv", ROOT / "data/test_real.csv", ROOT / "out"
CV = GroupKFold(n_splits=5)  # grouped by keyword: no merchant in both train and val


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["category"])
    df["doc"] = [to_document(d, a) for d, a in zip(df["description"], df["amount"])]
    return df


def evaluate(model, test: pd.DataFrame, classes: list[str]) -> dict:
    pred = model.predict(test["doc"])
    return {
        "accuracy": round(accuracy_score(test["category"], pred), 4),
        "macro_f1": round(f1_score(test["category"], pred, labels=classes, average="macro", zero_division=0), 4),
        "report": classification_report(test["category"], pred, labels=classes, output_dict=True, zero_division=0),
    }


def top_terms(pipe: Pipeline, n: int = 10) -> dict:
    vocab = np.array(pipe.named_steps["vec"].get_feature_names_out())
    log_prob = pipe.named_steps["clf"].feature_log_prob_
    out = {}
    for i, cls in enumerate(pipe.classes_):
        # How much more likely the term is in this class than in the rest.
        others = np.delete(log_prob, i, axis=0).mean(axis=0)
        order = np.argsort(log_prob[i] - others)[::-1]
        out[cls] = [t for t in vocab[order] if not t.startswith("amt_") and " " not in t][:n]
    return out


def main() -> None:
    train = load(TRAIN)
    classes = sorted(train["category"].unique())
    test = load(TEST) if TEST.exists() else None
    if test is None:
        print("No data/test_real.csv — reporting cross-validation only.")
    else:
        unknown = set(test["category"]) - set(classes)
        if unknown:
            raise SystemExit(f"test_real.csv has unknown categories: {sorted(unknown)}")

    X, y, groups = train["doc"], train["category"], train["keyword"]
    nb = Pipeline([("vec", CountVectorizer(**vectorizer_kwargs())), ("clf", MultinomialNB())])
    search = GridSearchCV(nb, {"clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}, cv=CV, scoring="f1_macro")
    search.fit(X, y, groups=groups)
    nb = search.best_estimator_

    models = {
        "baseline_most_frequent": Pipeline([("vec", CountVectorizer(**vectorizer_kwargs())),
                                            ("clf", DummyClassifier(strategy="most_frequent"))]),
        "count_nb": nb,
        "tfidf_logreg": Pipeline([("vec", TfidfVectorizer(**vectorizer_kwargs())),
                                  ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    }

    metrics = {
        "shipped_model": "count_nb",
        "nb_alpha": search.best_params_["clf__alpha"],
        "train_rows": len(train),
        "test_rows": 0 if test is None else len(test),
        "train_source": "synthetic bootstrap from lexicon.json (generate.py)",
        "test_source": "real hand-labelled M-Pesa rows (test_real.csv)" if test is not None else None,
        "classes": classes,
        "models": {},
    }
    for name, model in models.items():
        cv = cross_val_score(model, X, y, groups=groups, cv=CV, scoring="f1_macro")
        model.fit(X, y)
        entry = {"cv_macro_f1_train": round(float(cv.mean()), 4)}
        if test is not None:
            entry["real_test"] = evaluate(model, test, classes)
        metrics["models"][name] = entry
        print(f"\n== {name}  CV macro-F1 (train, grouped by keyword): {entry['cv_macro_f1_train']}")
        if test is not None:
            rt = entry["real_test"]
            print(f"   REAL test accuracy {rt['accuracy']}  macro-F1 {rt['macro_f1']}")
            print(classification_report(test["category"], model.predict(test["doc"]),
                                        labels=classes, zero_division=0))

    if test is not None:
        nb_f1 = metrics["models"]["count_nb"]["real_test"]["macro_f1"]
        lr_f1 = metrics["models"]["tfidf_logreg"]["real_test"]["macro_f1"]
        metrics["logreg_minus_nb_macro_f1"] = round(lr_f1 - nb_f1, 4)
        cm = confusion_matrix(test["category"], nb.predict(test["doc"]), labels=classes)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT / "confusion_matrix.csv")

    metrics["top_terms"] = top_terms(nb)
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(nb, OUT / "categoriser.pkl")
    print(f"\nbest NB alpha = {metrics['nb_alpha']}. Wrote out/metrics.json, out/categoriser.pkl"
          + (", out/confusion_matrix.csv" if test is not None else ""))
    for cls, terms in metrics["top_terms"].items():
        print(f"  {cls:<16} {', '.join(terms)}")


if __name__ == "__main__":
    main()
