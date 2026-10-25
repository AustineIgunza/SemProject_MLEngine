# SemProject_MLEngine — Akiba transaction categoriser

ML engine for Akiba, a personal-finance app. It trains a Multinomial Naive Bayes
model that suggests an expense category from a transaction description and
amount. **Nothing here is deployed.** Training runs on a laptop; the output is
one JSON file (`out/model.json`) that the Akiba Express server loads and runs in
pure TypeScript — no Python service, no external API.

`Expenses_Exploration.ipynb` is the Deliverable 1 dataset exploration.

## Rebuild the model from scratch

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python generate.py                # lexicon.json  -> data/train.csv
# optional: real-data evaluation
python extract.py <pdf-password>  # data/raw/*.pdf -> data/to_label.csv (anonymised)
#   ...label the category column, save as data/test_real.csv...
python train.py                   # -> out/metrics.json, out/confusion_matrix.csv, out/categoriser.pkl
python export.py                  # -> out/model.json, out/parity.json
```

To ship it, copy `out/model.json` to the app's
`server/src/modules/categoriser/model.json` and `out/parity.json` to
`server/src/modules/categoriser/__fixtures__/parity.json`, then restart the
server. Commit `out/model.json` and `out/metrics.json` together so the shipped
model and its reported numbers always match.

## Files

| File | Purpose |
|---|---|
| `lexicon.json` | Hand-built merchant/keyword list per category — **the quality of the whole model rests on this** |
| `generate.py` | Expands the lexicon into M-Pesa-style training rows (bootstrap; seeded, reproducible) |
| `extract.py` | Pulls money-out rows from M-Pesa PDF statements and anonymises them |
| `train.py` | Baseline vs Count+NB vs TF-IDF+LogReg; tunes NB alpha by CV; evaluates on real data |
| `export.py` | Serialises the NB model to `out/model.json` for the TypeScript runtime |
| `agreement.py` | Cohen's kappa between two labellers on the overlap rows |
| `features.py` | Text preprocessing — **must match** the app's `server/src/modules/categoriser/text.ts` |

## Categories

Exactly the nine expense defaults seeded by the app
(`server/src/db/migrations/006_seed_default_categories.sql`):
Food, Rent, Transport, Academic, Airtime & Data, Entertainment, Health,
Personal Care, Other. Income is not classified in v1.

## Labelling the real test set

1. Put statement PDFs in `data/raw/` (git-ignored — never commit them).
2. `python extract.py <password>` and read the 10 printed samples: no phone
   numbers, names or receipt codes may survive.
3. Fill the `category` column of `data/to_label.csv` with one of the nine names
   above. Save as `data/test_real.csv`. Target ~250 rows.
4. Both labellers label the same 60 rows independently (`data/overlap_a.csv`,
   `data/overlap_b.csv`) and run `python agreement.py data/overlap_a.csv data/overlap_b.csv`.

`test_real.csv` is never used for fitting or tuning — NB alpha is chosen by
cross-validation on the bootstrap set only.

## Reading the numbers

- **CV macro-F1 (train)** uses `GroupKFold` grouped by source keyword, so each
  validation fold contains only merchants the model has never seen. It is a
  deliberately pessimistic bound on "unknown merchant" performance, not the
  headline number.
- **Real test accuracy / macro-F1** is the headline number: how well a
  lexicon-trained model labels genuine M-Pesa descriptions.
- Without `data/test_real.csv`, `train.py` still trains and exports; it just reports cross-validation only.

## Keep Python and TypeScript in sync

`features.py` and the app's `text.ts` implement the same steps — lowercase, replace
anything not `a-z`/whitespace with a space, collapse spaces, append one amount
bucket token (`amt_xs` <100, `amt_s` <500, `amt_m` <2000, `amt_l` <10000,
`amt_xl`), then unigrams + bigrams. If one changes, change the other and
retrain.
