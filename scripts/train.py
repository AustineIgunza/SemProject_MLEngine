"""Train on the lexicon bootstrap; evaluate on real labelled rows when available.

  train: data/train.csv      (generate.py)
  test:  data/test_real.csv  (optional — hand-labelled M-Pesa rows, never used
                              for fitting)

Ships CountVectorizer + MultinomialNB with default smoothing: integer counts
port to TypeScript in a few lines. The other taught models (Logistic Regression,
Decision Tree, K-Nearest Neighbours, Random Forest, Gradient Boosting) go
through the same pipeline and the same folds as a comparison, and their scores
are recorded in metrics.json so they can be reported honestly.

Usage:  python scripts/train.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from features import to_document, vectorizer_kwargs

ROOT = Path(__file__).resolve().parent.parent
TRAIN, TEST, OUT = ROOT / "data/train.csv", ROOT / "data/test_real.csv", ROOT / "out"
# Same 5 folds as notebooks/NaiveBayes_Base.ipynb. The nine classes are near-balanced,
# so accuracy is a fair metric; random guessing would score about 1/9 = 11%.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["category"])
    df["doc"] = [to_document(d, a) for d, a in zip(df["description"], df["amount"])]
    return df


def pipeline(clf) -> Pipeline:
    return Pipeline([("vec", CountVectorizer(**vectorizer_kwargs())), ("clf", clf)])


def per_class(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "precision": precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0),
        "rows": pd.Series(y_true).value_counts().reindex(classes, fill_value=0).values,
    }, index=classes).round(4)


def evaluate(model, test: pd.DataFrame, classes: list[str]) -> dict:
    pred = model.predict(test["doc"])
    return {
        "accuracy": round(accuracy_score(test["category"], pred), 4),
        "per_class": per_class(test["category"], pred, classes).to_dict(orient="index"),
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

    X, y = train["doc"], train["category"]
    nb = pipeline(MultinomialNB())  # default smoothing (alpha = 1)

    models = {
        "count_nb": nb,
        "logistic_regression": pipeline(LogisticRegression(max_iter=1000)),
        "decision_tree": pipeline(DecisionTreeClassifier(random_state=42)),
        "knn": pipeline(KNeighborsClassifier()),
        "random_forest": pipeline(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        "gradient_boosting": pipeline(GradientBoostingClassifier(random_state=42)),
    }

    metrics = {
        "shipped_model": "count_nb",
        "cv": "5-fold stratified k-fold, shuffled, random_state=42",
        "train_rows": len(train),
        "test_rows": 0 if test is None else len(test),
        "train_source": "synthetic bootstrap from lexicon.json (generate.py)",
        "test_source": "real hand-labelled M-Pesa rows (test_real.csv)" if test is not None else None,
        "classes": classes,
        "models": {},
    }
    for name, model in models.items():
        cv = cross_val_score(model, X, y, cv=CV, scoring="accuracy", n_jobs=-1)
        entry = {"cv_accuracy_train": round(float(cv.mean()), 4), "cv_accuracy_std": round(float(cv.std()), 4)}
        if test is not None:
            model.fit(X, y)
            entry["real_test"] = evaluate(model, test, classes)
        metrics["models"][name] = entry
        print(f"\n== {name:<20} CV accuracy (train, 5-fold stratified): {entry['cv_accuracy_train']}")
        if test is not None:
            print(f"   REAL test accuracy {entry['real_test']['accuracy']}")
            print(per_class(test["category"], model.predict(test["doc"]), classes).to_string())

    nb.fit(X, y)
    if test is not None:
        cm = confusion_matrix(test["category"], nb.predict(test["doc"]), labels=classes)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT / "confusion_matrix.csv")

    metrics["top_terms"] = top_terms(nb)
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(nb, OUT / "categoriser.pkl")
    print("\nShipped count_nb (MultinomialNB, default smoothing). Wrote out/metrics.json, out/categoriser.pkl"
          + (", out/confusion_matrix.csv" if test is not None else ""))
    for cls, terms in metrics["top_terms"].items():
        print(f"  {cls:<16} {', '.join(terms)}")


if __name__ == "__main__":
    main()
