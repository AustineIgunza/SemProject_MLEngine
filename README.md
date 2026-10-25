# SemProject_MLEngine — Akiba transaction categoriser

ML engine for Akiba, a personal-finance app. It trains a Multinomial Naive Bayes
model that suggests an expense category from a transaction description and
amount. Training runs on a laptop; the output is
one JSON file named out/model.json that the Akiba Express server loads and runs in
pure TypeScript.

`Expenses_Exploration.ipynb` is the dataset exploration.




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




