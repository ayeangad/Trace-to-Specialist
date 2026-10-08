"""Evaluation (spec P6; P3.8 gate lives here as named by the plan).

Phase 3 gate: python -m tailguard.evaluate --phase3-gate [--runs ...]
Full P6 metrics land in this module in Phase 6.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tailguard.config import EXP
from tailguard.logging_utils import read_jsonl

TAIL_SLICES = ("T1", "T2", "T3", "T4", "T5", "T6")


def conf_risk(rec: dict) -> float:
    """Confidence-only risk: -mean value-token logprob; +inf when missing."""
    try:
        idx = rec["greedy"]["value_token_idx"]
        lp = rec["greedy"]["turns"][[t.get("tool") for t in
                                     rec["greedy"]["turns"]].index("submit_answer")]["lp"]
        vals = [lp[i] for i in idx]
        if not vals or any(v is None or (isinstance(v, float) and math.isnan(v))
                           for v in vals):
            return math.inf
        return -sum(vals) / len(vals)
    except (KeyError, ValueError, IndexError, TypeError):
        return math.inf


def phase3_gate(runs_path: str | Path) -> dict:
    recs = [r for r in read_jsonl(runs_path)
            if r.get("split") in ("fit", "cal", "test")]
    n = len(recs)
    errs = [1 - int(r["label"]["success"]) for r in recs]
    e = sum(errs) / n if n else 0.0
    n_err = sum(errs)
    tail_share = (sum(1 for r, x in zip(recs, errs)
                      if x and r.get("slice") in TAIL_SLICES) / n_err
                  if n_err else 0.0)
    s0 = [1 - int(r["label"]["success"]) for r in recs if r.get("slice") == "S0"]
    s0_err = sum(s0) / len(s0) if s0 else 0.0
    risks = [conf_risk(r) for r in recs]
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(errs, risks)) if len(set(errs)) == 2 else None
    except Exception:
        auroc = None
    go = bool(e >= 0.05 and tail_share >= 0.60 and s0_err <= 0.05)
    return {"n": n, "error_rate": e, "tail_share_of_errors": tail_share,
            "s0_error": s0_err, "auroc_conf_only": auroc, "go": go}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3-gate", action="store_true")
    ap.add_argument("--runs", default=str(EXP / "specialist_runs.jsonl"))
    a = ap.parse_args()
    if a.phase3_gate:
        print(json.dumps(phase3_gate(a.runs), indent=1))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
