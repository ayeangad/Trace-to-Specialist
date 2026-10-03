"""Tool implementations over the synthetic filing store. Stdlib only.

These are plain functions with type hints + Google-style docstrings so they
can be passed directly to TRL GRPOTrainer(tools=[...]) in V1-GRPO, and also
driven by the local baseline harness / executable env.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_DOCS = Path(__file__).resolve().parent.parent / "data" / "documents" / "filings.json"
# Override for Experiment 2 (real-SEC transfer): point at a different store in
# the SAME schema with zero code changes, e.g.
#   HYDE_FILINGS=data/real_sec/filings.json python evals/run_model_baseline.py ...
_DOCS_PATH = Path(os.environ.get("HYDE_FILINGS", _DEFAULT_DOCS))
_FILINGS: dict | None = None


def _load() -> dict:
    global _FILINGS
    if _FILINGS is None:
        _FILINGS = json.loads(_DOCS_PATH.read_text()) if _DOCS_PATH.exists() else {}
    return _FILINGS


def reload_store(path: str | None = None) -> int:
    """Reload the filing store (useful in tests). Returns filing count."""
    global _FILINGS
    p = Path(path) if path else _DOCS_PATH
    _FILINGS = json.loads(p.read_text()) if p.exists() else {}
    return len(_FILINGS)


def search_filing(query: str) -> str:
    """Search filings by company, ticker, or year.

    Args:
        query: Free-text query, e.g. "ACME 2024" or "CYBD revenue".

    Returns:
        Newline-separated candidate filing_ids (up to 5), or "NO_RESULTS".
    """
    filings = _load()
    q = query.lower()
    hits = [fid for fid, f in filings.items()
            if q.replace("-", " ") in f"{f['company']} {f['ticker']} {f['year']}".lower()
            or f["ticker"].lower() in q or f["company"].lower().split()[0] in q]
    # fallback: year match
    if not hits:
        hits = [fid for fid, f in filings.items() if str(f["year"]) in q][:5]
    return "\n".join(sorted(hits)[:5]) if hits else "NO_RESULTS"


def open_section(filing_id: str, section: str) -> str:
    """Open a section of a filing.

    Args:
        filing_id: Filing id, e.g. "ACME-10K-2024".
        section: One of financial_statements, mda, notes.

    Returns:
        Section text, or an error string starting with "ERROR".
    """
    filings = _load()
    f = filings.get(filing_id)
    if f is None:
        return "ERROR: unknown filing_id"
    if section not in f["sections"]:
        return f"ERROR: unknown section. valid={sorted(f['sections'])}"
    return f["sections"][section]


def get_table(filing_id: str, table_id: str) -> str:
    """Get a structured table from a filing.

    Args:
        filing_id: Filing id, e.g. "ACME-10K-2024".
        table_id: One of income_statement, balance_sheet, cash_flow.

    Returns:
        JSON string of {metric: value}, or an error string.
    """
    filings = _load()
    f = filings.get(filing_id)
    if f is None:
        return "ERROR: unknown filing_id"
    t = f["tables"].get(table_id)
    if t is None:
        return f"ERROR: unknown table. valid={sorted(f['tables'])}"
    return json.dumps(t)


# OpenAI-style function specs, in the same order/names as above. Passed as
# `tools=` to tokenizer.apply_chat_template so the model-family-native
# tool-call syntax is rendered (Qwen3 JSON vs Qwen3.5 XML-ish). Single source
# of truth for SFT rendering, the eval harness loop, and TRL's env tools.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_filing",
        "description": "Search filings by company, ticker, or year.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Free-text query, e.g. 'ACME 2024'."}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "open_section",
        "description": "Open a section of a filing.",
        "parameters": {"type": "object",
                       "properties": {
                           "filing_id": {"type": "string", "description": "Filing id, e.g. 'ACME-10K-2024'."},
                           "section": {"type": "string", "description": "One of financial_statements, mda, notes."}},
                       "required": ["filing_id", "section"]}}},
    {"type": "function", "function": {
        "name": "get_table",
        "description": "Get a structured table from a filing.",
        "parameters": {"type": "object",
                       "properties": {
                           "filing_id": {"type": "string"},
                           "table_id": {"type": "string", "description": "One of income_statement, balance_sheet, cash_flow."}},
                       "required": ["filing_id", "table_id"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final answer. Call once, with the exact value, filing id, and verbatim evidence span.",
        "parameters": {"type": "object",
                       "properties": {
                           "value": {"type": "string", "description": "Extracted value, e.g. '$12,345 million'."},
                           "filing_id": {"type": "string"},
                           "evidence": {"type": "string", "description": "Exact evidence span from the section text."}},
                       "required": ["value", "filing_id", "evidence"]}}},
]
