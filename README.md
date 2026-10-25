# SemProject_MLEngine — Akiba transaction categoriser

ML engine for Akiba, a personal-finance app. It trains a Multinomial Naive Bayes model that suggests an
expense category from an M-Pesa transaction description and amount. Training runs on a laptop; the output
is one JSON file, `out/model.json`, which the Akiba Express server loads and runs in pure TypeScript.

## Files

| File | Purpose |
|---|---|
| `notebooks/Expenses_Exploration.ipynb` | Deliverable 1: exploration of the Kaggle `Expenses_clean.csv` dataset |
| `notebooks/NaiveBayes_Base.ipynb` | Deliverable 2: data cleaning, EDA, the Naive Bayes base model and the model comparison |
| `data/Expenses_clean.csv` | Kaggle *Financial Transactions Dataset*: 938 expense rows, 14 categories, BYN. No description column, so not used for training |
| `data/Income_clean.csv` | Same Kaggle dataset: 349 income rows, 7 categories, BYN. Kept for reference |
| `data/train.csv` | Training data generated from the lexicon (5,587 rows) |
| `lexicon.json` | Hand-built merchant/keyword list per category. The quality of the whole model rests on this |
| `scripts/generate.py` | Expands the lexicon into M-Pesa-style training rows (seeded, reproducible) |
| `scripts/features.py` | Text preprocessing. Must match `text.ts` in the Akiba app's categoriser module |
| `scripts/train.py` | Trains the shipped Naive Bayes model and scores it against Logistic Regression, Decision Tree, KNN, Random Forest and Gradient Boosting |
| `scripts/export.py` | Writes the trained model to `out/model.json`, plus `out/parity.json` for the TypeScript parity test |
| `scripts/predict.py` | Classifies a description and amount with `out/model.json`, no scikit-learn needed |
| `scripts/extract.py` | Pulls money-out rows from M-Pesa PDF statements and anonymises them for labelling |
| `out/metrics.json` | Cross-validation accuracy for every model |

## Rebuilding the model

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate.py        # lexicon.json -> data/train.csv
python scripts/train.py           # out/metrics.json, out/categoriser.pkl
python scripts/export.py          # out/model.json, out/parity.json
python scripts/predict.py "chicken inn" 650
```

To run the notebooks, also `pip install matplotlib jupyter`.

## Reading the numbers

`out/metrics.json` reports 5-fold stratified cross-validation accuracy, with the same folds for every
model. The nine classes are near-balanced (about 11% each), so accuracy is a fair measure; random
guessing would score about 11%.

Read the scores as optimistic. The 5,587 training rows come from only 457 merchant keywords, so the same
merchant usually lands in both the training and the test folds, and the model is partly tested on
merchants it has already seen. That is why the shipped model scores about 97%. A realistic figure needs
real, hand-labelled M-Pesa rows, which have not been collected yet.

## Categories

Exactly the nine expense defaults seeded by the Akiba app: Food, Rent, Transport, Academic,
Airtime & Data, Entertainment, Health, Personal Care, Other. Income is not classified in v1.
