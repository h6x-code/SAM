"""
ui.py — binds the DOM to the inference engine and aspect tagger.
- Single Review tab: runs analyze() and renders probabilities, explanation, aspects.
- CSV Upload tab: reads CSV in browser, chunked processing with a progress bar, then enables download.

Assumes:
  - models/model.json exists (exported by trainer)
  - models/taxonomy.json exists (Step 5; optional — handled gracefully if absent)
"""
from __future__ import annotations
import json
from typing import List, Dict, Any

from js import document, FileReader, Blob, URL
from pyodide.ffi import to_js
from pyodide.http import pyfetch

from .inference import load_model, analyze
from .aspects import load_taxonomy, tag_aspects
from .utils import parse_csv, to_csv, chunked

# --- DOM helpers ---
def qs(sel: str):
    return document.querySelector(sel)

def qsa(sel: str):
    return document.querySelectorAll(sel)

def set_html(el, html: str):
    setattr(el, "innerHTML", html)

def set_progress(pct: float):
    bar = qs("#progress-bar")
    if bar:
        bar.style.width = f"{max(0, min(100, pct))}%"

def set_status(msg: str):
    qs("#csv-status").textContent = msg

def format_prob_row(label: str, p: float) -> str:
    pct = f"{p*100:.1f}%"
    return f"<div class='row'><strong style='min-width:120px'>{label}</strong><span>{pct}</span></div>"

def render_single_output(result: Dict[str, Any], aspects: List[Dict[str, Any]], show_details: bool):
    html = []
    label = result["label"].capitalize()
    probs = result["probs"]
    html.append(f"<h3>Overall: {label}</h3>")
    html.append("<div class='card'>")
    html.append(format_prob_row("Negative", probs["negative"]))
    html.append(format_prob_row("Neutral", probs["neutral"]))
    html.append(format_prob_row("Positive", probs["positive"]))
    html.append("</div>")
    if show_details:
        exp = result["explanation"]
        def contrib_list(items):
            if not items: return "<em>(none)</em>"
            return ", ".join([f"{t} ({v:+.3f})" for t, v in items])
        html.append("<div class='card'>")
        html.append("<h4>Top tokens</h4>")
        html.append(f"<div><strong>Overall:</strong> {contrib_list(exp['top_overall'])}</div>")
        html.append(f"<div><strong>Positive:</strong> {contrib_list(exp['top_positive'])}</div>")
        html.append(f"<div><strong>Negative:</strong> {contrib_list(exp['top_negative'])}</div>")
        if aspects:
            html.append("<h4 style='margin-top:10px'>Aspects</h4>")
            for a in aspects:
                html.append(f"<div class='row'><strong style='min-width:180px'>{a['aspect']}</strong>"
                            f"<span>conf: {(a['confidence']*100):.1f}%</span>"
                            f"<span style='margin-left:8px'>hits: {a['hits']}</span></div>")
        html.append("</div>")
    set_html(qs("#single-output"), "\n".join(html))

# --- init model/taxonomy on page load ---
_initialized = False
async def _ensure_init():
    global _initialized
    if _initialized:
        return
    # Load model + taxonomy (taxonomy optional)
    await load_model("models/model.json")
    await load_taxonomy("models/taxonomy.json")
    _initialized = True

# --- Single Review handlers ---
def on_analyze_click(evt=None):
    async def run():
        await _ensure_init()
        text = qs("#single-text").value or ""
        band = float(qs("#neutral-band").value or "0.10")
        show_details = bool(qs("#show-details").checked)
        result = analyze(text, neutral_band=band, topk_tokens=5)
        aspects = tag_aspects(text)
        render_single_output(result, aspects, show_details)
    import asyncio
    asyncio.ensure_future(run())

qs("#analyze-btn").addEventListener("click", on_analyze_click)

# --- CSV: file selection → show column choices ---
def _read_file_as_text(file) -> None:
    """
    Return text via a JS Promise-like callback; here we set a temporary attribute.
    """
    reader = FileReader.new()
    def onload(evt):
        # store CSV text on a dataset field of #csv-file input for simplicity
        qs("#csv-file").dataset.fileText = reader.result
        # sniff columns and populate selects
        text = reader.result or ""
        rows, header, delim = parse_csv(text)
        # populate selects
        text_sel = qs("#text-col"); id_sel = qs("#id-col")
        # clear
        text_sel.innerHTML = ""; id_sel.innerHTML = "<option value=''>(none)</option>"
        for h in header:
            opt1 = document.createElement("option"); opt1.value = h; opt1.textContent = h
            opt2 = document.createElement("option"); opt2.value = h; opt2.textContent = h
            text_sel.appendChild(opt1); id_sel.appendChild(opt2)
        qs("#csv-columns").classList.remove("hidden")
        set_status(f"Loaded CSV with {len(rows):,} rows, {len(header)} columns (delim inferred).")
    reader.onload = onload
    reader.readAsText(file)

def on_file_change(evt=None):
    file_input = qs("#csv-file")
    if file_input.files.length == 0:
        return
    _read_file_as_text(file_input.files.item(0))

qs("#csv-file").addEventListener("change", on_file_change)

# --- CSV processing ---
_processed_cache = None

def on_process_click(evt=None):
    async def run():
        await _ensure_init()
        text_col = qs("#text-col").value
        id_col = qs("#id-col").value
        band = float(qs("#csv-neutral-band").value or "0.10")
        include_topk = bool(qs("#show-topk").checked)
        chunk_size = int(qs("#chunk-size").value or "1000")

        csv_text = qs("#csv-file").dataset.get("fileText", "")
        if not csv_text:
            set_status("No CSV loaded.")
            return
        rows, header, delim = parse_csv(csv_text)
        if not rows:
            set_status("CSV appears empty.")
            return
        if text_col not in header:
            set_status(f"Column '{text_col}' not found.")
            return

        # Output columns
        out_rows: List[Dict[str, Any]] = []
        base_fields = list(header)
        new_fields = [
            "overall_label", "p_negative", "p_neutral", "p_positive", "polarity_score"
        ]
        aspect_fields_seen = set()

        total = len(rows)
        done = 0
        set_progress(0)
        set_status("Starting analysis…")

        # Process in chunks to keep UI responsive
        for chunk in chunked(rows, chunk_size):
            # allow browser to breathe
            import asyncio
            await asyncio.sleep(0)
            for r in chunk:
                text = r.get(text_col, "") or ""
                result = analyze(text, neutral_band=band, topk_tokens=5)
                aspects = tag_aspects(text)
                out = dict(r)  # keep original fields
                out["overall_label"] = result["label"]
                out["p_negative"] = f"{result['probs']['negative']:.6f}"
                out["p_neutral"]  = f"{result['probs']['neutral']:.6f}"
                out["p_positive"] = f"{result['probs']['positive']:.6f}"
                out["polarity_score"] = f"{result['polarity_score']:.6f}"
                # per-aspect
                for a in aspects:
                    key_conf = f"aspect_{a['aspect']}_conf"
                    key_hits = f"aspect_{a['aspect']}_hits"
                    out[key_conf] = f"{a['confidence']:.6f}"
                    out[key_hits] = str(a["hits"])
                    aspect_fields_seen.add(key_conf); aspect_fields_seen.add(key_hits)
                # optional top tokens
                if include_topk:
                    top_overall = "; ".join([t for t,_ in result["explanation"]["top_overall"]])
                    out["top_tokens"] = top_overall
                out_rows.append(out)
            done += len(chunk)
            pct = 100.0 * done / total
            set_progress(pct)
            set_status(f"Processed {done:,} / {total:,} rows ({pct:.1f}%)")

        # Assemble final fieldnames
        fieldnames = base_fields + new_fields + sorted(aspect_fields_seen)
        if include_topk:
            fieldnames.append("top_tokens")

        csv_out = to_csv(out_rows, fieldnames)
        _blob = Blob.new([csv_out], {"type": "text/csv;charset=utf-8"})
        url = URL.createObjectURL(_blob)
        btn = qs("#download-btn")
        btn.disabled = False
        # attach a one-time click handler that triggers download
        def trigger_download(ev):
            a = document.createElement("a")
            a.href = url
            a.download = "sam_output.csv"
            a.style.display = "none"
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)
            btn.removeEventListener("click", trigger_download)
        btn.addEventListener("click", trigger_download)

        set_status("Done. Click “Download processed CSV”.")
        set_progress(100.0)
    import asyncio
    asyncio.ensure_future(run())

qs("#process-btn").addEventListener("click", on_process_click)

# Ensure model/taxonomy begin loading asap so first action feels instant
async def _warm_start():
    try:
        await _ensure_init()
    except Exception as e:
        # Render a friendly message into both output/status areas
        set_html(qs("#single-output"), f"<div class='notice'>Model load failed: {e}</div>")
        set_status(f"Model load failed: {e}")

import asyncio
asyncio.ensure_future(_warm_start())
