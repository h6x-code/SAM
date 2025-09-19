"""
aspects.py — simple keyword tagging with confidence = hits / tokens.
Loads /site/models/taxonomy.json if present; otherwise uses an empty taxonomy.
"""
from __future__ import annotations
from typing import List, Dict, Any
from pyodide.http import pyfetch
from .utils import tokenize

_TAXONOMY: Dict[str, List[str]] = {}
_LOADED = False

async def load_taxonomy(path: str="models/taxonomy.json") -> None:
    global _TAXONOMY, _LOADED
    try:
        resp = await pyfetch(path)
        if resp.ok:
            _TAXONOMY = await resp.json()
            _LOADED = True
        else:
            _TAXONOMY = {}
            _LOADED = True
    except Exception:
        _TAXONOMY = {}
        _LOADED = True

def tag_aspects(text: str, token_pattern: str=r"\b\w+\b", lowercase: bool=True) -> List[Dict[str, Any]]:
    toks = tokenize(text or "", token_pattern, lowercase=lowercase)
    total = max(len(toks), 1)
    hits_by_aspect: List[Dict[str, Any]] = []
    for aspect, keywords in _TAXONOMY.items():
        hits = 0
        # naive membership check; multiword keywords are allowed
        text_joined = " ".join(toks)
        for kw in keywords:
            if " " in kw:
                # phrase match
                if kw in text_joined:
                    hits += 1
            else:
                if kw in toks:
                    hits += 1
        conf = hits / total
        if hits > 0:
            hits_by_aspect.append({"aspect": aspect, "confidence": conf, "hits": hits})
    # sort by confidence desc
    hits_by_aspect.sort(key=lambda x: x["confidence"], reverse=True)
    return hits_by_aspect
