"""Deterministic reward: outcome + evidence + process efficiency. Stdlib only.

Total 10 pts:
  correct filing      +2
  correct section     +1
  correct value       +3 (exact int match after normalization)
  valid evidence span +2 (evidence substring present in cited section text)
  efficiency          +2 (<=4 tool calls, no errors/repeats; partial +1 if <=6)

submit_answer() validates; reward() scores a finished episode from env state.
"""
from __future__ import annotations

import re
from . import tools as T


def normalize_value(s: str) -> int | None:
    """Parse '$12,345 million' / '12345' -> int. Returns None if unparseable."""
    if s is None:
        return None
    m = re.search(r"[\d,]+", str(s))
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def score_episode(task: dict, state: dict, submitted_value: str,
                  submitted_filing: str, evidence: str) -> dict:
    """Dispatch on task_family. L1 single-metric (below); L2 dual-metric and
    L3 YoY have their own 10-pt breakdowns but the identical return schema."""
    fam = task.get("task_family", "financial_metric_extraction")
    if fam == "financial_dual_extraction":
        return _score_dual(task, state, submitted_value, submitted_filing, evidence)
    if fam == "financial_yoy_analysis":
        return _score_yoy(task, state, submitted_value, submitted_filing, evidence)
    gt = task["ground_truth"]
    calls = state.get("tool_calls", [])
    n_calls = len(calls)
    errors = sum(1 for c in calls if str(c.get("result", "")).startswith("ERROR"))

    filing_ok = (submitted_filing == gt["filing_id"])
    value_ok = (normalize_value(submitted_value) == gt["value"])
    # evidence: submitted span must appear in the cited section text
    section_text = ""
    try:
        section_text = T.open_section(submitted_filing, state.get("section_cited", gt["section"]))
    except Exception:
        section_text = ""
    evidence_ok = bool(evidence) and evidence.strip() in section_text

    section_ok = (state.get("section_cited") == gt["section"])

    r_filing = 2 if filing_ok else 0
    r_section = 1 if section_ok else 0
    r_value = 3 if value_ok else 0
    r_evid = 2 if (evidence_ok and filing_ok) else 0
    if n_calls <= 4 and errors == 0:
        r_eff = 2
    elif n_calls <= 6 and errors <= 1:
        r_eff = 1
    else:
        r_eff = 0

    total = r_filing + r_section + r_value + r_evid + r_eff
    return {
        "reward": float(total),
        "breakdown": {"filing": r_filing, "section": r_section, "value": r_value,
                      "evidence": r_evid, "efficiency": r_eff},
        "success": bool(value_ok and filing_ok),
        "n_tool_calls": n_calls, "n_errors": errors,
    }


def _split2(s: str, sep: str) -> tuple[str, str]:
    parts = (s or "").split(sep)
    parts = [p.strip() for p in parts] + ["", ""]
    return parts[0], parts[1]


def _eff(calls: list, errors: int, limit: int, full_pts: int = 2) -> int:
    if len(calls) <= limit and errors == 0:
        return full_pts
    if len(calls) <= limit + 2 and errors <= 1:
        return max(0, full_pts - 1)
    return 0


def _score_dual(task: dict, state: dict, submitted_value: str,
                submitted_filing: str, evidence: str) -> dict:
    """L2: two values + two evidence spans, one filing. Submit value "VA | VB",
    evidence "EA || EB". 10 = filing 2 + section 1 + valueA 2 + valueB 2 +
    evidence 2 (1 per span) + efficiency 1."""
    gt = task["ground_truth"]
    calls = state.get("tool_calls", [])
    errors = sum(1 for c in calls if str(c.get("result", "")).startswith("ERROR"))
    va_s, vb_s = _split2(submitted_value, "|")
    ea_s, eb_s = _split2(evidence, "||")
    filing_ok = (submitted_filing == gt["filing_id"])
    va_ok = (normalize_value(va_s) == gt["value_a"])
    vb_ok = (normalize_value(vb_s) == gt["value_b"])
    try:
        text = T.open_section(submitted_filing, state.get("section_cited", gt["section"]))
    except Exception:
        text = ""
    ea_ok = bool(ea_s) and ea_s in text
    eb_ok = bool(eb_s) and eb_s in text
    section_ok = (state.get("section_cited") == gt["section"])
    bd = {"filing": 2 if filing_ok else 0,
          "section": 1 if section_ok else 0,
          "value_a": 2 if va_ok else 0,
          "value_b": 2 if vb_ok else 0,
          "evidence": (1 if (ea_ok and filing_ok) else 0) + (1 if (eb_ok and filing_ok) else 0),
          "efficiency": _eff(calls, errors, 4, 1)}
    return {"reward": float(sum(bd.values())), "breakdown": bd,
            "success": bool(filing_ok and va_ok and vb_ok),
            "n_tool_calls": len(calls), "n_errors": errors}


def _score_yoy(task: dict, state: dict, submitted_value: str,
               submitted_filing: str, evidence: str) -> dict:
    """L3: YoY growth + threshold verdict across two filings. Submit filing
    "ID_A,ID_B", value "PCT|yes|no", evidence "EA || EB". 10 = filingA 1 +
    filingB 1 + section 1 + yoy±0.05 → 2 + verdict 2 + evidence 2 (1 per span,
    each in its own filing) + efficiency 1."""
    gt = task["ground_truth"]
    calls = state.get("tool_calls", [])
    errors = sum(1 for c in calls if str(c.get("result", "")).startswith("ERROR"))
    fa_s, fb_s = _split2(submitted_filing, ",")
    pct_s, verdict_s = _split2(submitted_value, "|")
    ea_s, eb_s = _split2(evidence, "||")
    fa_ok = (fa_s == gt["filing_id"])
    fb_ok = (fb_s == gt["filing_id_prev"])
    m = re.search(r"-?[\d.]+", pct_s)
    yoy_ok = (m is not None) and abs(float(m.group(0)) - gt["yoy_pct"]) <= 0.05
    verdict_ok = ((verdict_s.lower() in ("yes", "y")) == bool(gt["verdict"]))
    try:
        text_a = T.open_section(gt["filing_id"], gt["section"])
    except Exception:
        text_a = ""
    try:
        text_b = T.open_section(gt["filing_id_prev"], gt["section"])
    except Exception:
        text_b = ""
    ea_ok = bool(ea_s) and ea_s in text_a
    eb_ok = bool(eb_s) and eb_s in text_b
    section_ok = (state.get("section_cited") == gt["section"])
    bd = {"filing_a": 1 if fa_ok else 0, "filing_b": 1 if fb_ok else 0,
          "section": 1 if section_ok else 0,
          "yoy": 2 if yoy_ok else 0, "verdict": 2 if verdict_ok else 0,
          "evidence": (1 if (ea_ok and fa_ok) else 0) + (1 if (eb_ok and fb_ok) else 0),
          "efficiency": _eff(calls, errors, 6, 1)}
    return {"reward": float(sum(bd.values())), "breakdown": bd,
            "success": bool(fa_ok and fb_ok and yoy_ok and verdict_ok),
            "n_tool_calls": len(calls), "n_errors": errors}
