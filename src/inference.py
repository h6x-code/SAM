"""
inference.py — pure-Python TF-IDF + multinomial softmax inference for the browser.

Loads site/models/model.json and provides:
  - await load_model()
  - analyze(text: str, neutral_band: float=0.10, topk_tokens: int=5) -> dict

No sklearn required in-browser.
"""
from __future__ import annotations
import json
import math
import re
from typing import Dict, Any, List, Tuple, Optional

from pyodide.http import pyfetch
from .utils import tokenize, make_ngrams, l2_norm, softmax, top_k_contrib

_MODEL_CACHE: Dict[str, Any] = {}
VEC: Dict[str, Any] = {}
CLS: Dict[str, Any] = {}

async def _fetch_json(path: str) -> Any:
    resp = await pyfetch(path)
    if resp.status != 200:
        raise RuntimeError(f"Failed to load {path}: HTTP {resp.status}")
    return await resp.json()

async def load_model(path: str = "models/model.json") -> None:
    global _MODEL_CACHE, VEC, CLS
    if _MODEL_CACHE:
        return
    data = await _fetch_json(path)
    _MODEL_CACHE = data

    # Vectorizer params
    VEC = {
        "lowercase": bool(data["preprocess"].get("lowercase", True)),
        "token_pattern": str(data["preprocess"].get("token_pattern", r"\b\w+\b")),
        "ngram_range": tuple(data["preprocess"].get("ngram_range", [1, 1])),
        "sublinear_tf": bool(data["preprocess"].get("sublinear_tf", False)),
        "norm": str(data["preprocess"].get("norm", "l2")),
        "vocab": dict(data["vocab"]),
        "idf": list(map(float, data["idf"])),
        "V": int(len(data["idf"])),
    }

    # Classifier params
    CLS = {
        "W": [list(map(float, row)) for row in data["W"]],  # shape [3,V]
        "b": list(map(float, data["b"])),                   # shape [3]
        "labels": list(data.get("labels", ["negative","neutral","positive"])),
    }

def _tfidf_vector(tokens: List[str]) -> Tuple[Dict[int, float], List[Tuple[str, float]]]:
    """
    Build sparse TF or TF-IDF vector aligned to vocab indices.
    Returns:
      vec: {idx: value}
      used_terms: [(term, tfidf_value)] for explanation
    """
    vocab: Dict[str, int] = VEC["vocab"]
    idf: List[float] = VEC["idf"]
    sublinear = VEC["sublinear_tf"]

    # term frequency
    counts: Dict[int, int] = {}
    tf_raw: Dict[int, float] = {}
    terms_used: List[Tuple[str, float]] = []

    for t in tokens:
        idx = vocab.get(t)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1

    if not counts:
        return {}, []

    for idx, c in counts.items():
        tf = 1.0 + math.log(c) if sublinear else float(c)
        tfidf = tf * idf[idx]
        tf_raw[idx] = tfidf
        # store string term for explanation
        # (We need term string; reverse-lookup is O(V) if we build map on load; do it once)
    # Build reverse vocab lazily
    if "_rev_vocab" not in VEC:
        VEC["_rev_vocab"] = {v: k for k, v in vocab.items()}
    rev = VEC["_rev_vocab"]
    terms_used = [(rev[i], tf_raw[i]) for i in tf_raw.keys()]
    # normalization (L2) if requested
    if VEC["norm"] == "l2":
        norm = l2_norm(tf_raw)
        vec = {i: (v / norm) for i, v in tf_raw.items()}
    else:
        vec = tf_raw
    return vec, terms_used

def _logits(vec: Dict[int, float]) -> List[float]:
    W: List[List[float]] = CLS["W"]  # [3,V]
    b: List[float] = CLS["b"]        # [3]
    out = []
    for k in range(3):
        acc = b[k]
        row = W[k]
        for idx, val in vec.items():
            acc += row[idx] * val
        out.append(acc)
    return out

def _neutral_override(probs: List[float], neutral_band: float) -> Optional[int]:
    # If top-2 close within band, return neutral index (1) to hedge
    pairs = sorted([(p, i) for i, p in enumerate(probs)], reverse=True)
    if len(pairs) >= 2 and (pairs[0][0] - pairs[1][0]) <= neutral_band:
        return 1
    return None

def _explain(terms_used: List[Tuple[str,float]], topk: int=5) -> Dict[str, Any]:
    # Estimate contribution as max over class weights * tfidf per term
    # For a simpler, intuitive story we’ll report tokens with largest |sum_k w_kj * tfidf_j|,
    # and bucket by sign using (w_pos - w_neg) as a crude polarity indicator.
    W = CLS["W"]
    rev_contribs: List[Tuple[str, float]] = []
    pos_contribs: List[Tuple[str, float]] = []
    neg_contribs: List[Tuple[str, float]] = []
    for term, val in terms_used:
        j = VEC["vocab"].get(term)
        if j is None:
            continue
        w_neg, w_neu, w_pos = W[0][j], W[1][j], W[2][j]
        raw = (w_neg + w_neu + w_pos) * val  # magnitude for ranking
        rev_contribs.append((term, raw))
        polarity = (w_pos - w_neg) * val
        if polarity >= 0:
            pos_contribs.append((term, polarity))
        else:
            neg_contribs.append((term, polarity))
    return {
        "top_overall": top_k_contrib(rev_contribs, topk),
        "top_positive": top_k_contrib(pos_contribs, topk),
        "top_negative": top_k_contrib(neg_contribs, topk),
    }

def _preprocess(text: str) -> List[str]:
    toks = tokenize(text, VEC["token_pattern"], lowercase=VEC["lowercase"])
    ngrams = make_ngrams(toks, VEC["ngram_range"])
    return ngrams

def label_name(idx: int) -> str:
    return CLS["labels"][idx] if 0 <= idx < len(CLS["labels"]) else ["negative","neutral","positive"][idx]

def polarity_score(probs: List[float]) -> float:
    # simple scalar for sorting: pos - neg
    return float(probs[2] - probs[0])

def analyze(text: str, neutral_band: float=0.10, topk_tokens: int=5) -> Dict[str, Any]:
    ngrams = _preprocess(text or "")
    vec, terms_used = _tfidf_vector(ngrams)
    if not vec:
        # empty or OOV text
        probs = [1/3, 1/3, 1/3]
        return {
            "label_idx": 1,
            "label": label_name(1),
            "probs": {"negative": probs[0], "neutral": probs[1], "positive": probs[2]},
            "polarity_score": 0.0,
            "explanation": {"top_overall":[], "top_positive":[], "top_negative":[]},
            "used_terms": [],
        }
    z = _logits(vec)
    probs = softmax(z)
    override = _neutral_override(probs, neutral_band)
    best_idx = 1 if override is not None else int(max(range(3), key=lambda i: probs[i]))
    out = {
        "label_idx": best_idx,
        "label": label_name(best_idx),
        "probs": {"negative": probs[0], "neutral": probs[1], "positive": probs[2]},
        "polarity_score": polarity_score(probs),
        "explanation": _explain(terms_used, topk_tokens),
        "used_terms": terms_used,
    }
    return out
