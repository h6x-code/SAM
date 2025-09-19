#!/usr/bin/env python3
"""
train.py
- Load Yelp reviews from SQLite (preferred) or stream JSON directly.
- Map stars → {neg, neu, pos}, split, train TF-IDF + multinomial logistic regression.
- Evaluate and export compact model.json for browser inference.

Usage:
  python trainer/train.py --config trainer/config.yml

Important:
- The browser inference layer expects lowercase + token pattern \\b\\w+\\b and (1,2)-grams by default.
- Keep 'max_features' reasonable (<= 50k) to keep model.json lightweight for the web.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@dataclass
class Config:
    reviews_json: str
    db_path: str
    table: str
    max_docs: Optional[int]
    test_size: float
    stratify: bool
    max_features: int
    ngram_range: Tuple[int, int]
    min_df: int
    max_df: float
    seed: int
    map_stars: Dict[int, str]
    out_path: str
    lowercase: bool
    token_pattern: str
    sublinear_tf: bool
    norm: str


NEG, NEU, POS = "negative", "neutral", "positive"
LABEL_ORDER = [NEG, NEU, POS]
LABEL_TO_ID = {"neg": 0, "neu": 1, "pos": 2}
ID_TO_LABEL = {0: NEG, 1: NEU, 2: POS}


def iter_yelp_json(path: str) -> Any:
    # Helpful guardrails
    if os.path.isdir(path):
        raise ValueError(
            f"[train] reviews_json points to a DIRECTORY, not a file: {path}\n"
            "→ Fix trainer/config.yml to the actual review.json file path.\n"
            "  In Docker, that is typically /app/data/review.json"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[train] reviews_json file not found: {path}\n"
            "→ Ensure you mounted the file into the container and the config path matches."
        )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(
        reviews_json=cfg["yelp"]["reviews_json"],
        db_path=cfg["yelp"]["db_path"],
        table=cfg["yelp"]["table"],
        max_docs=cfg["train"]["max_docs"],
        test_size=float(cfg["train"]["test_size"]),
        stratify=bool(cfg["train"]["stratify"]),
        max_features=int(cfg["train"]["max_features"]),
        ngram_range=tuple(cfg["train"]["ngram_range"]),
        min_df=int(cfg["train"]["min_df"]),
        max_df=float(cfg["train"]["max_df"]),
        seed=int(cfg["train"]["seed"]),
        map_stars={int(k): str(v) for k, v in cfg["labels"]["map_stars"].items()},
        out_path=cfg["export"]["out_path"],
        lowercase=bool(cfg["preprocess"]["lowercase"]),
        token_pattern=str(cfg["preprocess"]["token_pattern"]),
        sublinear_tf=bool(cfg["preprocess"]["sublinear_tf"]),
        norm=str(cfg["preprocess"]["norm"]),
    )


def ensure_dir_for(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_from_sqlite(db_path: str, table: str, max_docs: Optional[int]) -> Optional[pd.DataFrame]:
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    # Efficient sample if max_docs set
    if max_docs is None:
        df = pd.read_sql_query(f"SELECT review_id, text, stars FROM {table}", conn)
    else:
        # Random sample using rowid; fallback to LIMIT if no large table
        try:
            cnt = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {table}", conn).iloc[0]["n"]
            frac = min(1.0, max_docs / max(1, cnt))
            df = pd.read_sql_query(
                f"SELECT review_id, text, stars FROM {table} TABLESAMPLE SYSTEM (100*{frac})", conn
            )
            if len(df) < max_docs:
                df = pd.read_sql_query(
                    f"SELECT review_id, text, stars FROM {table} LIMIT {max_docs}", conn
                )
        except Exception:
            df = pd.read_sql_query(
                f"SELECT review_id, text, stars FROM {table} LIMIT {max_docs}", conn
            )
    conn.close()
    return df


def iter_yelp_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_from_json(path: str, max_docs: Optional[int]) -> pd.DataFrame:
    rows = []
    n = 0
    for rec in iter_yelp_json(path):
        try:
            stars = int(rec.get("stars", 0))
            text = (rec.get("text") or "").strip()
            if 1 <= stars <= 5 and len(text) >= 5:
                rows.append((rec.get("review_id"), text, stars))
                n += 1
                if max_docs is not None and n >= max_docs:
                    break
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["review_id", "text", "stars"])


def map_stars_to_ids(stars: pd.Series, mapping: Dict[int, str]) -> np.ndarray:
    # mapping values are 'neg'/'neu'/'pos' strings → map to IDs 0/1/2
    labels = []
    for s in stars:
        key = mapping.get(int(s))
        if key not in LABEL_TO_ID:
            labels.append(None)
        else:
            labels.append(LABEL_TO_ID[key])
    arr = np.array([x for x in labels if x is not None], dtype=np.int64)
    return arr


def align_xy(df: pd.DataFrame, mapping: Dict[int, str]) -> Tuple[List[str], np.ndarray]:
    # Return texts and label-ids aligned (drop rows with missing map)
    texts, y = [], []
    for _, row in df.iterrows():
        s = int(row["stars"])
        key = mapping.get(s)
        if key in LABEL_TO_ID:
            texts.append(str(row["text"]))
            y.append(LABEL_TO_ID[key])
    return texts, np.array(y, dtype=np.int64)


def train_and_export(cfg: Config) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 1) Load data
    df = read_from_sqlite(cfg.db_path, cfg.table, cfg.max_docs)
    if df is None or df.empty:
        print("[train] No SQLite found or table empty; streaming JSON directly …")
        df = read_from_json(cfg.reviews_json, cfg.max_docs)

    if df.empty:
        raise RuntimeError("No data loaded. Check paths in trainer/config.yml")

    print(f"[train] Loaded {len(df):,} rows")

    # 2) Prepare X, y
    X_texts, y_ids = align_xy(df, cfg.map_stars)
    if cfg.max_docs is not None and len(X_texts) > cfg.max_docs:
        X_texts = X_texts[: cfg.max_docs]
        y_ids = y_ids[: cfg.max_docs]
    print(f"[train] After mapping, usable rows: {len(X_texts):,}")

    stratify = y_ids if cfg.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_texts, y_ids, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify
    )

    # 3) Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=cfg.lowercase,
        token_pattern=cfg.token_pattern,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        sublinear_tf=cfg.sublinear_tf,
        norm=cfg.norm,
        dtype=np.float32,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # 4) Classifier (multinomial LR)
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=cfg.seed,
        max_iter=500,
        n_jobs=None,  # sklearn 1.4 l-bfgs ignores n_jobs; leaving for clarity
    )
    clf.fit(Xtr, y_train)

    # 5) Eval
    y_pred = clf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f"[eval] accuracy: {acc:.4f} | macro-F1: {f1_macro:.4f}")

    # sanity: ensure class order 0,1,2
    if not np.array_equal(clf.classes_, np.array([0, 1, 2])):
        raise RuntimeError(f"Unexpected class order from sklearn: {clf.classes_}")

    # 6) Export compact model.json
    ensure_dir_for(cfg.out_path)

    vocab: Dict[str, int] = vectorizer.vocabulary_  # token -> index
    # Align arrays to column order (sklearn already aligned)
    idf = vectorizer.idf_.astype(float).tolist()
    W = clf.coef_.astype(float).tolist()       # shape (3, V)
    b = clf.intercept_.astype(float).tolist()  # shape (3,)

    export = {
        "version": "sam-1.0.0",
        "preprocess": {
            "lowercase": cfg.lowercase,
            "token_pattern": cfg.token_pattern,
            "ngram_range": list(cfg.ngram_range),
            "sublinear_tf": cfg.sublinear_tf,
            "norm": cfg.norm,
        },
        "vocab": vocab,  # {token: idx}
        "idf": idf,      # [V]
        "W": W,          # [3, V] order: negative, neutral, positive
        "b": b,          # [3]
        "labels": [NEG, NEU, POS],
        "max_features": cfg.max_features,
    }

    with open(cfg.out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False)

    print(f"[export] wrote {cfg.out_path} ({os.path.getsize(cfg.out_path)/1024:.1f} KB)")

    # 7) Minimal label distribution snapshot
    uniq, cnts = np.unique(y_train, return_counts=True)
    dist = {ID_TO_LABEL[int(u)]: int(c) for u, c in zip(uniq, cnts)}
    print(f"[train] train label counts: {dist}")


def main():
    ap = argparse.ArgumentParser(description="Train TF-IDF + multinomial LR on Yelp reviews")
    ap.add_argument("--config", default="trainer/config.yml", help="Path to YAML config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_and_export(cfg)


if __name__ == "__main__":
    main()
