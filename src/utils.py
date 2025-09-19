"""
utils.py — small helpers for tokenization, n-grams, CSV IO, chunking, and DOM updates.
Pure Python; compatible with PyScript/Pyodide.
"""
from __future__ import annotations
import csv
import io
import math
import re
from typing import Iterable, List, Tuple, Dict, Any

# --- tokenization & n-grams ---
_token_re_cache: Dict[str, re.Pattern] = {}

def compile_token_re(pattern: str) -> re.Pattern:
    if pattern not in _token_re_cache:
        _token_re_cache[pattern] = re.compile(pattern, flags=re.IGNORECASE)
    return _token_re_cache[pattern]

def tokenize(text: str, pattern: str, lowercase: bool=True) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if lowercase:
        text = text.lower()
    rx = compile_token_re(pattern)
    return rx.findall(text)

def make_ngrams(tokens: List[str], ngram_range: Tuple[int,int]) -> List[str]:
    lo, hi = ngram_range
    feats: List[str] = []
    N = len(tokens)
    for n in range(lo, hi+1):
        for i in range(N - n + 1):
            feats.append(" ".join(tokens[i:i+n]))
    return feats

# --- math helpers ---
def l2_norm(vec: Dict[int, float]) -> float:
    s = 0.0
    for _, v in vec.items():
        s += v*v
    return math.sqrt(s) if s > 0 else 1.0

def softmax(z: List[float]) -> List[float]:
    m = max(z) if z else 0.0
    exps = [math.exp(v - m) for v in z]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

# --- csv helpers (browser-friendly) ---
def sniff_header(sample: str) -> Tuple[List[str], str]:
    """
    Try to detect delimiter; return header columns and delimiter.
    """
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=[",",";","|","\t"])
        delim = dialect.delimiter
    except csv.Error:
        delim = ","
    reader = csv.reader(io.StringIO(sample), delimiter=delim)
    try:
        header = next(reader)
    except StopIteration:
        header = []
    return header, delim

def parse_csv(text: str) -> Tuple[List[Dict[str,str]], List[str], str]:
    """
    Return (rows, header, delimiter) as list of dicts.
    """
    header, delim = sniff_header(text[:5000])
    if not header:
        # fallback: try one more line
        header, delim = sniff_header(text.splitlines(True)[:3] and "".join(text.splitlines(True)[:3]) or "")
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    rows = [dict(r) for r in reader]
    if not header and rows:
        header = list(rows[0].keys())
    return rows, header, delim

def to_csv(rows: List[Dict[str,Any]], fieldnames: List[str], delimiter: str=",") -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})
    return buf.getvalue()

# --- chunking ---
def chunked(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# --- explanation helpers ---
def top_k_contrib(contribs: List[Tuple[str, float]], k: int=5) -> List[Tuple[str,float]]:
    # contribs: (token/ngram, signed weight * tfidf)
    contribs_sorted = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)
    return contribs_sorted[:k]
