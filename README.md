# SemProject_MLEngine: Akiba expense categoriser

ML engine for Akiba, a personal finance app for university students. It trains a Multinomial Naive Bayes
model that suggests one expense category from the amount, date, account and tag of a transaction. The
trained model is exported to `out/model.json`, which `scripts/predict.py` runs in pure Python.

## Data

`data/Expenses_clean.csv` is the expenses file of the Kaggle Financial Transactions Dataset: 938 real
expenses from one person, 7 January to 30 November 2025, in Belarusian roubles (BYN). It has no
description or merchant text. Cleaning drops the currency column (always BYN) and the 14 zero-amount rows,
leaving 924 rows. `data/Income_clean.csv` is the income side of the same dataset and is not used, because
Akiba only categorises spending.

## Category mapping

| Dataset category | Akiba category |
|---|---|
| Public transport, Taxi | Transport |
| Cafe, Food | Food |
| Leisure | Entertainment |
| Health | Health |
| University | Academic |
| Clothes, Bought for myself | Personal Care |
| Gifts, Loan given, Fines, Job, Other | Other |

Akiba's Rent and Airtime & Data have no examples in this dataset, so the model predicts 7 of the 9 categories.

## Files

| File | Purpose |
|---|---|
| `notebooks/Expenses_Exploration.ipynb` | Deliverable 1: first exploration of the dataset |
| `notebooks/NaiveBayes_Base.ipynb` | Cleaning, EDA, the Naive Bayes base model, model comparison and regularisation |
| `scripts/features.py` | Cleaning steps 2 to 4 and 7, and the 6 features. Used by the notebook and the scripts |
| `scripts/train.py` | 5-fold comparison of six models, then fits Naive Bayes on all rows |
| `scripts/export.py` | Writes the model as JSON, plus 25 real rows with scikit-learn's probabilities |
| `scripts/predict.py` | Classifies one expense from `out/model.json`, no scikit-learn needed |
| `scripts/extract.py` | Future work: anonymises donated M-Pesa statement PDFs for labelling, as in the proposal |

The 6 features are the amount bin ([0,1], (1,2], (2,5], (5,10], (10,25], (25,60], (60,∞) BYN), day of
week, month, is-weekend, account and tag, one-hot encoded.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py             # out/metrics.json, out/categoriser.pkl
python scripts/export.py            # out/model.json, out/parity.json
python scripts/predict.py --check   # pure-Python inference matches scikit-learn on 25 rows
python scripts/predict.py --amount 5 --date 2025-06-14 --account acct_1 --tag tag_1
python scripts/extract.py <pdf-password>   # future work: data/raw/*.pdf -> data/to_label.csv
```

## Results

From 5-fold stratified cross validation (`out/metrics.json`), Naive Bayes scores 77.5% accuracy and a
macro-F1 of 0.31. Both numbers matter. Accuracy looks good, but always predicting Transport would
already score 51.4%. Macro-F1 averages the F1 score of each category, so it shows what accuracy hides:
the model handles Transport and Food well but misses Entertainment, Health, Academic and Personal Care,
which have very few rows. The other five models score 73% to 78% accuracy and 0.30 to 0.35 macro-F1,
so the limit is the data, not the algorithm.

The Akiba app's TypeScript categoriser still expects the old text-based `model.json` and must be updated
to the new format before this model can run inside the app.
