"""Inter-labeller agreement on the shared overlap rows.

Both labellers label the same ~60 rows independently, saved as two CSVs with
identical description/amount columns in the same order.

Usage:  python agreement.py data/overlap_a.csv data/overlap_b.csv
"""

import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python agreement.py <labeller_a.csv> <labeller_b.csv>")
    a, b = (pd.read_csv(p) for p in sys.argv[1:])
    if len(a) != len(b) or not (a["description"] == b["description"]).all():
        raise SystemExit("The two files must contain the same rows in the same order.")

    agree = (a["category"] == b["category"]).mean()
    kappa = cohen_kappa_score(a["category"], b["category"])
    print(f"rows: {len(a)}   raw agreement: {agree:.1%}   Cohen's kappa: {kappa:.3f}")
    disagreements = a[a["category"] != b["category"]].assign(other=b["category"])
    for _, row in disagreements.iterrows():
        print(f"  {row['description']!r:<50} A={row['category']:<15} B={row['other']}")


if __name__ == "__main__":
    main()
