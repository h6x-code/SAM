#!/usr/bin/env python3
"""
Train TF-IDF + multinomial Logistic Regression on Yelp reviews and export a compact model.json
that the browser (PyScript/Pyodide) can load.

Features:
- Load from: (a) JSONL splits via --train-json/--test-json, OR (b) SQLite, OR (c) raw Yelp NDJSON (review.json).
- Config-driven preprocessing to match browser inference (token_pattern, ngram_range, lowercase, sublinear_tf, norm).
- Exports only pure-Python types: vocab, idf, W, b, labels, preprocess config.

Usage:
  # A) Stream straight from big JSON (configured in trainer/config.yml)
  python trainer/train.py --config trainer/config.yml

  # B) Train on pre-made splits (recommended)
  python trainer/train.py --config trainer/config.yml \
         --train-json data/splits/train.jsonl \
         --test-json  data/splits/val.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ----------------------------
# Config & defaults
# ----------------------------
@dataclass
class Config:
    # paths
    reviews_json: str
    db_path: str
    table: str

    # training
    max_docs: Optional[int]
    test_size: float
    stratify: bool
    max_features: int
    ngram_range: Tuple[int, int]
    min_df: int
    max_df: float
    seed: int

    # mapping
    map_stars: Dict[int, str]

    # export
    out_path: str

    # preprocess (mirrors browser)
    lowercase: bool
    token_pattern: str
    sublinear_tf: bool
    norm: str


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    yelp = y.get("yelp", {})
    train = y.get("train", {})
    labels = y.get("labels", {})
    export = y.get("export", {})
    preprocess = y.get("preprocess", {})

    return Config(
        reviews_json=yelp.get("reviews_json", "data/review.json"),
        db_path=yelp.get("db_path", "data/yelp.db"),
        table=yelp.get("table", "reviews"),
        max_docs=train.get("max_docs", None),
        test_size=float(train.get("test_size", 0.2)),
        stratify=bool(train.get("stratify", False)),
        max_features=int(train.get("max_features", 50000)),
        ngram_range=tuple(train.get("ngram_range", [1, 2])),
        min_df=int(train.get("min_df", 5)),
        max_df=float(train.get("max_df", 0.9)),
        seed=int(train.get("seed", 123)),
        map_stars=labels.get("map_stars", {1: "neg", 2: "neg", 3: "neu", 4: "pos", 5: "pos"}),
        out_path=export.get("out_path", "site/models/model.json"),
        lowercase=bool(preprocess.get("lowercase", True)),
        token_pattern=str(preprocess.get("token_pattern", r"\b\w+\b")),
        sublinear_tf=bool(preprocess.get("sublinear_tf", False)),
        norm=str(preprocess.get("norm", "l2")),
    )


# ----------------------------
# Data loading
# ----------------------------
def iter_yelp_json(path: str) -> Iterable[Dict[str, Any]]:
    import os
    if os.path.isdir(path):
        raise ValueError(
            f"[train] reviews_json points to a DIRECTORY, not a file: {path}\n"
            "→ Fix trainer/config.yml to the actual file path (e.g. /abs/path/yelp_academic_dataset_review.json)."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[train] reviews_json file not found: {path}\n"
            "→ Ensure the path exists and is readable."
        )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_row(rec: Dict[str, Any]) -> Optional[Tuple[str, str, int]]:
    try:
        text = (rec.get("text") or "").strip()
        stars = int(rec.get("stars", 0))
        if not (1 <= stars <= 5 and len(text) >= 5):
            return None
        return rec.get("review_id"), text, stars
    except Exception:
        return None


def read_from_json(path: str, max_docs: Optional[int]) -> pd.DataFrame:
    rows: List[Tuple[str, str, int]] = []
    n = 0
    for rec in iter_yelp_json(path):
        row = clean_row(rec)
        if row is None:
            continue
        rows.append(row)
        n += 1
        if max_docs is not None and n >= max_docs:
            break
        if n and n % 500_000 == 0:
            print(f"[train] streamed {n:,} rows…")
    return pd.DataFrame(rows, columns=["review_id", "text", "stars"])


def read_from_sqlite(db_path: str, table: str, max_docs: Optional[int]) -> Optional[pd.DataFrame]:
    if not (db_path and os.path.exists(db_path)):
        return None
    try:
        conn = sqlite3.connect(db_path)
        limit = f" LIMIT {int(max_docs)}" if max_docs is not None else ""
        q = f"""
            SELECT review_id, text, stars
            FROM {table}
            WHERE LENGTH(text) >= 5 AND stars BETWEEN 1 AND 5
            {limit}
        """
        df = pd.read_sql_query(q, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[train] SQLite read failed: {e}")
        return None


def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            r = clean_row(rec)
            if r:
                rows.append(r)
    return pd.DataFrame(rows, columns=["review_id", "text", "stars"])


# ----------------------------
# Label mapping
# ----------------------------
KEY_TO_ID = {"neg": 0, "neu": 1, "pos": 2}
ID_TO_NAME = ["negative", "neutral", "positive"]


def align_xy(df: pd.DataFrame, map_stars: Dict[int, str]) -> Tuple[List[str], np.ndarray]:
    # map stars -> key -> id
    label_keys = [map_stars.get(int(s), "neu") for s in df["stars"].astype(int).tolist()]
    y = np.array([KEY_TO_ID.get(k, 1) for k in label_keys], dtype=np.int64)
    X = df["text"].astype(str).tolist()
    return X, y


# ----------------------------
# Training / Export
# ----------------------------
def export_model(
    out_path: str,
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    preprocess_cfg: Dict[str, Any],
) -> None:
    vocab = vectorizer.vocabulary_                       # dict: token -> idx
    idf = vectorizer.idf_.tolist()                       # list[float], len = V
    W = clf.coef_.tolist()                               # shape [3, V] for multinomial
    b = clf.intercept_.tolist()                          # shape [3]

    payload = {
        "version": "sam-1.0.0",
        "preprocess": {
            "lowercase": bool(preprocess_cfg.get("lowercase", True)),
            "token_pattern": str(preprocess_cfg.get("token_pattern", r"\\b\\w+\\b")),
            "ngram_range": list(preprocess_cfg.get("ngram_range", (1, 2))),
            "sublinear_tf": bool(preprocess_cfg.get("sublinear_tf", False)),
            "norm": str(preprocess_cfg.get("norm", "l2")),
        },
        "vocab": vocab,
        "idf": idf,
        "W": W,
        "b": b,
        "labels": ID_TO_NAME,
        "max_features": int(preprocess_cfg.get("max_features", len(vocab))),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[export] wrote {out_path} (V={len(vocab):,})")


def train_and_export(cfg: Config, train_json: Optional[str], test_json: Optional[str], out_override: Optional[str], max_docs_override: Optional[int]) -> None:
    rng = np.random.RandomState(cfg.seed)

    # 1) Load data
    if train_json:
        print(f"[train] Loading TRAIN from {train_json}")
        df_train = read_jsonl(train_json)
        if test_json:
            print(f"[train] Loading TEST  from {test_json}")
            df_test = read_jsonl(test_json)
        else:
            df_test = pd.DataFrame(columns=["review_id", "text", "stars"])
    else:
        df = read_from_sqlite(cfg.db_path, cfg.table, max_docs_override or cfg.max_docs)
        if df is None or df.empty:
            print("[train] No SQLite found or table empty; streaming JSON directly …")
            df = read_from_json(cfg.reviews_json, max_docs_override or cfg.max_docs)
        if df.empty:
            raise RuntimeError("No data loaded. Check paths in trainer/config.yml")
        print(f"[train] Loaded {len(df):,} rows")

        X_all, y_all = align_xy(df, cfg.map_stars)
        if cfg.max_docs is not None and len(X_all) > cfg.max_docs:
            X_all = X_all[: cfg.max_docs]
            y_all = y_all[: cfg.max_docs]

        stratify = y_all if cfg.stratify else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify
        )
        df_train = pd.DataFrame({"review_id": range(len(X_tr)), "text": X_tr, "stars": 3})
        df_test = pd.DataFrame({"review_id": range(len(X_te)), "text": X_te, "stars": 3})

    # 2) Vectorizer
    vec = TfidfVectorizer(
        lowercase=cfg.lowercase,
        token_pattern=cfg.token_pattern,
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        norm=cfg.norm,
        analyzer="word",
    )

    X_text_tr, y_tr = align_xy(df_train, cfg.map_stars)
    X_text_te, y_te = align_xy(df_test, cfg.map_stars) if not df_test.empty else ([], np.array([], dtype=np.int64))

    Xtr = vec.fit_transform(X_text_tr)
    Xte = vec.transform(X_text_te) if X_text_te else None

    # 3) Classifier
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=200,
        random_state=cfg.seed,
        n_jobs=None,
    )
    clf.fit(Xtr, y_tr)

    # 4) Eval
    if Xte is not None and X_text_te:
        yhat = clf.predict(Xte)
        acc = float(accuracy_score(y_te, yhat))
        f1 = float(f1_score(y_te, yhat, average="macro"))
        print(f"[eval] accuracy: {acc:.4f} | macro-F1: {f1:.4f} | n_test={len(y_te):,}")
    else:
        print("[eval] (no explicit test set)")

    # 5) Export
    out_path = out_override or cfg.out_path
    export_model(
        out_path,
        vectorizer=vec,
        clf=clf,
        preprocess_cfg={
            "lowercase": cfg.lowercase,
            "token_pattern": cfg.token_pattern,
            "ngram_range": cfg.ngram_range,
            "sublinear_tf": cfg.sublinear_tf,
            "norm": cfg.norm,
            "max_features": cfg.max_features,
        },
    )


def main():
    ap = argparse.ArgumentParser(description="Train TF-IDF + multinomial LR on Yelp reviews")
    ap.add_argument("--config", default="trainer/config.yml", help="Path to YAML config")
    ap.add_argument("--train-json", default=None, help="Path to train.jsonl (optional)")
    ap.add_argument("--test-json", default=None, help="Path to val/test jsonl (optional)")
    ap.add_argument("--out", default=None, help="Override output model path (optional)")
    ap.add_argument("--max-docs", type=int, default=None, help="Override max_docs from config (optional)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_and_export(cfg, train_json=args.train_json, test_json=args.test_json, out_override=args.out, max_docs_override=args.max_docs)


if __name__ == "__main__":
    main()
