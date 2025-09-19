#!/usr/bin/env python3
"""
split.py — create a reproducible train/val/test split from Yelp's review.json (NDJSON).

- Streams review.json (so it can handle large files).
- Filters: stars in {1..5}, len(text) >= 5
- Maps stars → labels {neg, neu, pos} for stratified split (recommended).
- Outputs three JSONL files: data/splits/train.jsonl, val.jsonl, test.jsonl

Usage (from repo root):
  python trainer/split.py --json /path/to/review.json \
      --outdir data/splits \
      --val-size 0.10 --test-size 0.10 \
      --seed 123 --max-docs 200000  --stratify-by label

Then train against a split:
  python trainer/train.py --config trainer/config.yml \
      --train-json data/splits/train.jsonl \
      --test-json  data/splits/val.jsonl   # or test.jsonl for final eval
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, Iterator, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

# ----- config defaults (aligns with trainer/config.yml mapping) -----
STAR_TO_KEY = {1: "neg", 2: "neg", 3: "neu", 4: "pos", 5: "pos"}
KEY_TO_ID = {"neg": 0, "neu": 1, "pos": 2}


def ensure_dir(path: str) -> None:
    d = os.path.abspath(path)
    os.makedirs(d, exist_ok=True)


def iter_yelp_json(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_row(rec: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return a normalized row dict or None to skip."""
    try:
        stars = int(rec.get("stars", 0))
        text = (rec.get("text") or "").strip()
        if not (1 <= stars <= 5 and len(text) >= 5):
            return None
        return {
            "review_id": rec.get("review_id"),
            "text": text,
            "stars": stars,
            "business_id": rec.get("business_id"),
            "date": rec.get("date"),
            "label_key": STAR_TO_KEY[stars],            # 'neg'|'neu'|'pos'
            "label_id": KEY_TO_ID[STAR_TO_KEY[stars]],  # 0|1|2
        }
    except Exception:
        return None


def load_rows(json_path: str, max_docs: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = 0
    for rec in iter_yelp_json(json_path):
        row = clean_row(rec)
        if row is None:
            continue
        rows.append(row)
        n += 1
        if max_docs is not None and n >= max_docs:
            break
        if n % 500000 == 0:
            print(f"[split] streamed {n:,} rows…")
    print(f"[split] loaded {len(rows):,} usable rows")
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[split] wrote {path} ({len(rows):,} rows)")


def do_split(
    rows: List[Dict[str, Any]],
    val_size: float,
    test_size: float,
    seed: int,
    strat_key: str | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Two-step split:
      1) out test
      2) split remaining into train/val
    Supports stratification by 'label_id' or 'stars' (or None).
    """
    if strat_key not in (None, "label", "stars"):
        raise ValueError("--stratify-by must be one of: label | stars | none")

    strat_arr = None
    if strat_key == "label":
        strat_arr = np.array([r["label_id"] for r in rows])
    elif strat_key == "stars":
        strat_arr = np.array([r["stars"] for r in rows])

    # Step 1: split out test
    rest, test = train_test_split(
        rows,
        test_size=test_size,
        random_state=seed,
        stratify=strat_arr if strat_arr is not None else None,
    )

    # Step 2: split rest into train/val (val_size is relative to original total)
    val_rel = val_size / max(1e-9, (1.0 - test_size))
    if strat_arr is not None:
        strat_rest = np.array([r["label_id"] for r in rest]) if strat_key == "label" else np.array([r["stars"] for r in rest])
    else:
        strat_rest = None

    train, val = train_test_split(
        rest,
        test_size=val_rel,
        random_state=seed,
        stratify=strat_rest if strat_rest is not None else None,
    )
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description="Create train/val/test splits from review.json")
    ap.add_argument("--json", required=True, help="Path to Yelp review.json (NDJSON)")
    ap.add_argument("--outdir", default="data/splits", help="Output directory for JSONL splits")
    ap.add_argument("--val-size", type=float, default=0.10, help="Validation fraction of total (default 0.10)")
    ap.add_argument("--test-size", type=float, default=0.10, help="Test fraction of total (default 0.10)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--max-docs", type=int, default=None, help="Cap docs for speed (optional)")
    ap.add_argument("--stratify-by", default="label", choices=["label", "stars", "none"], help="Stratify by mapped label, raw stars, or disable")
    args = ap.parse_args()

    if not (0.0 < args.val_size < 1.0) or not (0.0 < args.test_size < 1.0) or (args.val_size + args.test_size >= 1.0):
        raise ValueError("Choose val/test sizes in (0,1) and ensure val_size + test_size < 1.0")

    rows = load_rows(args.json, max_docs=args.max_docs)
    strat_key = None if args.stratify_by == "none" else args.stratify_by

    train, val, test = do_split(rows, args.val_size, args.test_size, args.seed, strat_key)

    ensure_dir(args.outdir)
    write_jsonl(os.path.join(args.outdir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.outdir, "val.jsonl"), val)
    write_jsonl(os.path.join(args.outdir, "test.jsonl"), test)

    # Tiny distribution report
    def dist(rows):
        from collections import Counter
        c = Counter([r["label_key"] for r in rows])
        total = max(len(rows), 1)
        return {k: f"{v} ({v/total:.1%})" for k, v in sorted(c.items())}

    print("[split] label distribution:")
    print("  train:", dist(train))
    print("  val  :", dist(val))
    print("  test :", dist(test))


if __name__ == "__main__":
    main()
