"""Train the Naive Bayes categoriser on the real Kaggle expenses.

Loads data/Expenses_clean.csv, builds the 6 features with features.py, and
scores Naive Bayes and the five comparison models with 5-fold stratified cross
validation (the same folds for every model). Then fits Naive Bayes on all rows.

Writes:  out/metrics.json     accuracy and macro-F1 (mean and std) for each model
         out/categoriser.pkl  the fitted Naive Bayes pipeline, read by export.py

Usage:   python scripts/train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from features import FEATURES, build_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Expenses_clean.csv"
OUT = ROOT / "out"

# Same folds and scoring as notebooks/NaiveBayes_Base.ipynb. A category the model
# never predicts gets precision 0 rather than a warning.
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {"accuracy": "accuracy",
           "macro_f1": make_scorer(f1_score, average="macro", zero_division=0)}

MODELS = {
    "naive_bayes": MultinomialNB(),  # the shipped model, default smoothing (alpha = 1)
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "random_forest": RandomForestClassifier(random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def pipeline(model) -> Pipeline:
    return Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore")), ("model", model)])


def main() -> None:
    raw = pd.read_csv(DATA)
    X, y = build_features(raw)
    metrics = {
        "data": "data/Expenses_clean.csv (Kaggle Financial Transactions Dataset, expenses)",
        "rows_raw": len(raw),
        "rows_used": len(X),
        "zero_amount_rows_dropped": int((raw["amount"] == 0).sum()),
        "features": FEATURES,
        "classes": sorted(y.unique()),
        "class_counts": {c: int(n) for c, n in y.value_counts().items()},
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "majority_class_accuracy": round(float(y.value_counts(normalize=True).max()), 4),
        "shipped_model": "naive_bayes",
        "models": {},
    }
    print(f"{len(raw)} raw rows -> {len(X)} rows, {X.shape[1]} features, {y.nunique()} classes")
    print(f"Always predicting {y.value_counts().idxmax()}: accuracy {metrics['majority_class_accuracy']:.4f}\n")
    print(f"{'model':<22}{'accuracy':<18}macro-F1")

    for name, model in MODELS.items():
        cv = cross_validate(pipeline(model), X, y, cv=FOLDS, scoring=SCORING)
        entry = {
            "cv_accuracy": round(float(cv["test_accuracy"].mean()), 4),
            "cv_accuracy_std": round(float(cv["test_accuracy"].std()), 4),
            "cv_macro_f1": round(float(cv["test_macro_f1"].mean()), 4),
            "cv_macro_f1_std": round(float(cv["test_macro_f1"].std()), 4),
        }
        metrics["models"][name] = entry
        print(f"{name:<22}{entry['cv_accuracy']:.4f} +/- {entry['cv_accuracy_std']:.4f}   "
              f"{entry['cv_macro_f1']:.4f} +/- {entry['cv_macro_f1_std']:.4f}")

    nb = pipeline(MODELS["naive_bayes"]).fit(X, y)
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    joblib.dump(nb, OUT / "categoriser.pkl")
    print("\nWrote out/metrics.json and out/categoriser.pkl (Naive Bayes fitted on all rows)")


if __name__ == "__main__":
    main()
