"""Real M-Pesa statements -> data/to_label.csv (anonymised, ready to label).

Reads password-protected statement PDFs from data/raw/, pulls the transaction
table, keeps money-out rows (income is not classified in v1), anonymises the
description and drops everything else (receipt no., time, balance).

Usage:  python extract.py <pdf-password>

Then label the `category` column by hand and save the result as
data/test_real.csv. Use exactly the nine category names in lexicon.json.
"""

import csv
import re
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "to_label.csv"

PHONE = re.compile(r"(?:\+?254|0)\s?[17]\d{2}\s?\d{3}\s?\d{3}")
MASKED_PHONE = re.compile(r"(?:\+?254|0)[17]\*{3,}\d+|\b[17]\d{2}\*{3,}\d+\b")
REF = re.compile(r"\b[A-Z]{2}\d{8,}\b|\b(?=[A-Z0-9]{10}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10}\b")
LONG_NUMBER = re.compile(r"\d{7,}")
# "Customer Transfer to [PHONE] - JANE WANJIKU" — the trailing name is a private
# person, not a merchant, so it goes too.
NAME_AFTER_PHONE = re.compile(r"(\[PHONE\])\s*-\s*[^-]+$")


def anonymise(description: str) -> str:
    text = " ".join(description.split())
    text = PHONE.sub("[PHONE]", text)
    text = MASKED_PHONE.sub("[PHONE]", text)
    text = REF.sub("[REF]", text)
    text = LONG_NUMBER.sub("[NUM]", text)
    text = NAME_AFTER_PHONE.sub(r"\1 - [NAME]", text)
    return text.strip()


def parse_amount(value) -> float | None:
    if value is None:
        return None
    cleaned = str(value).replace(",", "").replace("-", "").strip()
    try:
        amount = float(cleaned)
    except ValueError:
        return None
    return amount if amount > 0 else None


def find_columns(header: list) -> tuple[int, int] | None:
    """Index of the Details and Withdrawn columns, or None if not a txn table."""
    names = [str(h or "").strip().lower() for h in header]
    try:
        details = next(i for i, n in enumerate(names) if "details" in n)
        withdrawn = next(i for i, n in enumerate(names) if "withdrawn" in n)
    except StopIteration:
        return None
    return details, withdrawn


def extract(pdf_path: Path, password: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    columns = None
    with pdfplumber.open(pdf_path, password=password) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for raw in table:
                    found = find_columns(raw)
                    if found:
                        columns = found  # header row (repeats on some pages)
                        continue
                    if not columns:
                        continue
                    details_i, withdrawn_i = columns
                    if max(details_i, withdrawn_i) >= len(raw):
                        continue
                    amount = parse_amount(raw[withdrawn_i])
                    details = raw[details_i]
                    if amount is None or not details:
                        continue
                    rows.append((anonymise(str(details)), amount))
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python extract.py <pdf-password>")
    password = sys.argv[1]

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {RAW_DIR.relative_to(ROOT)}/")

    rows: list[tuple[str, float]] = []
    for pdf in pdfs:
        found = extract(pdf, password)
        print(f"  {pdf.name}: {len(found)} money-out rows")
        rows.extend(found)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "amount", "category"])
        for description, amount in rows:
            writer.writerow([description, amount, ""])

    print(f"\nWrote {len(rows)} rows to {OUT.relative_to(ROOT)}")
    print("Sample anonymised descriptions — check nothing personal survived:")
    for description, _ in rows[:10]:
        print(f"  {description}")


if __name__ == "__main__":
    main()
