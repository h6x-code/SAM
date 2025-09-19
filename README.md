# SAM — Sentiment Analysis Machine (Yelp Edition)

**Goal:** train a compact Yelp-review sentiment model offline (TF-IDF + multinomial logistic regression), then serve **client-side, Python-in-the-browser** inference via PyScript/Pyodide on GitHub Pages. No servers, no APIs.

## What you get
- **Trainer** (`/trainer`): Python pipeline to ingest Yelp Open Dataset (`review.json`), clean, train TF-IDF + softmax LR, evaluate, and export a small `model.json`.
- **Static site** (`/site`): PyScript app that loads `model.json` + `taxonomy.json` and runs pure-Python inference entirely in the browser.
  - Single-review analysis with probabilities + short explanation.
  - CSV upload → choose text column (and optional id) → chunked processing with progress bar → download augmented CSV.
  - Simple aspect tagging via keyword taxonomy.

## Design choices & tradeoffs
- **Portable artifacts:** export only what the browser needs (vocab, IDF, weights, intercept, config). No sklearn in the browser.
- **Simple tokenizer:** lowercase + `\b\w+\b` word tokens; 1–2 n-grams; `max_features` defaults to 50k (tune in `trainer/config.yml`).
- **Baseline model:** TF-IDF + multinomial LR (LBFGS). Fast, robust, and small.
- **Privacy:** all inference is client-side; CSVs never leave the browser.

---

## Quickstart

### 0) Prereqs
- Python **3.11.9**
- Yelp Open Dataset (the `data/review.json` file). See: https://www.yelp.com/dataset

### 1) Create env & install trainer deps
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r trainer/requirements.txt
```

### 2) Create Train/Val/Test splits
```bash
python trainer/split.py \
  --json app/data/review.json \
  --outdir data/full \
  --val-size 0.10 \
  --test-size 0.10 \
  --seed 1337 \
  --stratify-by label
