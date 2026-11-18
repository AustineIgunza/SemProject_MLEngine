"""Cleaning and features shared by the notebook, train.py and export.py.

The data is the Kaggle expenses file data/Expenses_clean.csv: one person's
spending in BYN, with a date, a category, an account, an amount and a tag.
build_features() applies cleaning steps 2, 3, 4 and 7 from the notebook:

  2. drop the currency column (it is always BYN)
  3. drop expenses with an amount of 0
  4. map the dataset's 14 categories onto Akiba's categories
  7. bin the amount and turn the date into day of week, month and is-weekend

Steps 1, 5 and 6 change no rows: the duplicates are genuine repeat purchases,
nothing is missing, and the large amounts are real purchases.
"""

import math

import pandas as pd

AKIBA_CATEGORY = {
    "Public transport": "Transport",
    "Taxi": "Transport",
    "Cafe": "Food",
    "Food": "Food",
    "Leisure": "Entertainment",
    "Health": "Health",
    "University": "Academic",
    "Clothes": "Personal Care",
    "Bought for myself": "Personal Care",
    "Gifts": "Other",
    "Loan given": "Other",
    "Fines": "Other",
    "Job": "Other",
    "Other": "Other",
}

# Amount bins in BYN: [0,1], (1,2], (2,5], (5,10], (10,25], (25,60], (60,inf)
AMOUNT_BIN_EDGES = [0, 1, 2, 5, 10, 25, 60, math.inf]
AMOUNT_BIN_LABELS = ["0-1", "1-2", "2-5", "5-10", "10-25", "25-60", "60+"]

FEATURES = ["amount_bin", "day_of_week", "month", "is_weekend", "account", "tags"]


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Raw expense rows -> (6 categorical features, Akiba category label)."""
    df = raw.drop(columns="currency")                 # step 2
    df = df[df["amount"] > 0]                          # step 3

    label = df["category"].map(AKIBA_CATEGORY)         # step 4
    unmapped = sorted(df.loc[label.isna(), "category"].unique())
    if unmapped:
        raise ValueError(f"No Akiba category for: {unmapped}")

    dates = pd.to_datetime(df["date_time"])            # step 7
    X = pd.DataFrame({
        "amount_bin": pd.cut(df["amount"], bins=AMOUNT_BIN_EDGES, labels=AMOUNT_BIN_LABELS,
                             include_lowest=True).astype(str),
        "day_of_week": dates.dt.day_name(),
        "month": dates.dt.month,
        "is_weekend": (dates.dt.dayofweek >= 5).astype(int),
        "account": df["account"],
        "tags": df["tags"],
    })
    return X.reset_index(drop=True), label.rename("akiba_category").reset_index(drop=True)
