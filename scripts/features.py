"""Text preprocessing shared by train.py and export.py.

This MUST stay byte-for-byte equivalent to the TypeScript side
(server/src/modules/categoriser/text.ts + categoriser.service.ts), because the
exported model is only valid if inference builds exactly the same tokens:

  1. lowercase
  2. replace every character that is not a-z or whitespace with a space
     (strips digits, punctuation, '&', etc.)
  3. collapse whitespace
  4. append one amount-bucket token
  5. unigrams + bigrams over whitespace-separated tokens (sklearn joins with ' ')
"""

import re

# [upper_bound_exclusive, token]. The last bucket has no upper bound.
AMOUNT_BUCKETS = [
    (0, 100, "amt_xs"),
    (100, 500, "amt_s"),
    (500, 2000, "amt_m"),
    (2000, 10000, "amt_l"),
    (10000, None, "amt_xl"),
]

NGRAM_RANGE = (1, 2)

_NON_ALPHA = re.compile(r"[^a-z\s]+")
_SPACES = re.compile(r"\s+")


def clean(text) -> str:
    text = "" if text is None else str(text)
    text = _NON_ALPHA.sub(" ", text.lower())
    return _SPACES.sub(" ", text).strip()


def amount_bucket(amount) -> str:
    value = abs(float(amount or 0))
    for low, high, token in AMOUNT_BUCKETS:
        if high is None or value < high:
            return token
    return AMOUNT_BUCKETS[-1][2]


def to_document(description, amount) -> str:
    cleaned = clean(description)
    bucket = amount_bucket(amount)
    return f"{cleaned} {bucket}" if cleaned else bucket


def vectorizer_kwargs() -> dict:
    """CountVectorizer settings that mirror the TS tokeniser exactly.

    lowercase=False and token_pattern=r"\\S+" because to_document() has already
    normalised the text; sklearn's default pattern would drop 1-char tokens and
    the TS side would not.
    """
    return {"lowercase": False, "token_pattern": r"\S+", "ngram_range": NGRAM_RANGE}
