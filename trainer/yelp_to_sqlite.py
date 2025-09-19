#!/usr/bin/env python3
"""
yelp_to_sqlite.py
- Stream Yelp Open Dataset `review.json` (NDJSON) and normalize into SQLite.
- Filters: stars in {1..5}, len(text) >= 5.
- Table schema: (review_id TEXT PRIMARY KEY, text TEXT, stars INT, business_id TEXT, date TEXT)

Usage:
  python trainer/yelp_to_sqlite.py --json /path/to/review.json --db data/yelp.db --table reviews --max-docs 2000000

You can skip this and let trainer/train.py stream JSON directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Iterator, Dict, Any

import pandas as pd


def iter_yelp_json(path: str) -> Iterator[Dict[str, Any]]:
    """Yield dicts from Yelp's NDJSON review file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def make_sqlite(db_path: str, table: str) -> sqlite3.Connection:
    ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            review_id   TEXT PRIMARY KEY,
            text        TEXT NOT NULL,
            stars       INTEGER NOT NULL,
            business_id TEXT,
            date        TEXT
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_stars ON {table}(stars)")
    return conn


def valid_record(rec: Dict[str, Any]) -> bool:
    try:
        stars = int(rec.get("stars", 0))
        text = (rec.get("text") or "").strip()
        return 1 <= stars <= 5 and len(text) >= 5
    except Exception:
        return False


def upsert_batch(conn: sqlite3.Connection, table: str, batch: list[tuple]) -> None:
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO {table} (review_id, text, stars, business_id, date)
        VALUES (?, ?, ?, ?, ?)
        """,
        batch,
    )


def json_to_sqlite(json_path: str, db_path: str, table: str, max_docs: int | None = None, report_every: int = 100000) -> int:
    conn = make_sqlite(db_path, table)
    cur = conn.cursor()
    n = 0
    batch: list[tuple] = []
    for rec in iter_yelp_json(json_path):
        if not valid_record(rec):
            continue
        batch.append(
            (
                rec.get("review_id"),
                (rec.get("text") or "").strip(),
                int(rec.get("stars")),
                rec.get("business_id"),
                rec.get("date"),
            )
        )
        if len(batch) >= 5000:
            upsert_batch(conn, table, batch)
            conn.commit()
            n += len(batch)
            batch.clear()
            if report_every and n % report_every == 0:
                print(f"[yelp_to_sqlite] inserted {n:,} rows...")
        if max_docs is not None and n >= max_docs:
            break
    if batch:
        upsert_batch(conn, table, batch)
        conn.commit()
        n += len(batch)
        batch.clear()
    # Vacuum/optimize lightly
    try:
        cur.execute("ANALYZE")
        cur.execute("VACUUM")
    except Exception:
        pass
    conn.close()
    print(f"[yelp_to_sqlite] done. total rows: {n:,}")
    return n


def quick_stats(db_path: str, table: str) -> None:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT stars, COUNT(*) as n FROM {table} GROUP BY stars ORDER BY stars", conn)
    print(df)
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Normalize Yelp review.json to SQLite")
    ap.add_argument("--json", required=True, help="Path to Yelp review.json (NDJSON)")
    ap.add_argument("--db", required=True, help="Output SQLite path (e.g., data/yelp.db)")
    ap.add_argument("--table", default="reviews", help="Table name (default: reviews)")
    ap.add_argument("--max-docs", type=int, default=None, help="Cap rows for speed/debug (optional)")
    args = ap.parse_args()

    json_to_sqlite(args.json, args.db, args.table, max_docs=args.max_docs)
    quick_stats(args.db, args.table)


if __name__ == "__main__":
    main()
