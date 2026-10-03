"""Validator for the Experiment-2 real-SEC set. Stdlib only.

Checks filings.json schema + tasks.jsonl integrity WITHOUT any model:
  - required keys, id format, non-empty tables/sections
  - every values figure's "$X million"-style string occurs in financial_statements
  - task ground_truth references existing filings; evidence substrings verbatim

Usage:
    python data/real_sec/validate.py [--docs data/real_sec/filings.json]
                                     [--tasks data/real_sec/tasks.jsonl]
Exit 0 = valid (empty store warns but passes: scaffold state).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_FILING_KEYS = {"filing_id", "company", "ticker", "year", "form",
                        "source", "values", "tables", "sections"}
REQUIRED_TASK_KEYS = {"task_id", "task_family", "prompt", "company", "ticker",
                      "year", "filing_id", "section", "ground_truth"}
FAMILY_KEYS = {"financial_metric_extraction": {"metric"},
               "financial_dual_extraction": {"metric_a", "metric_b"},
               "financial_yoy_analysis": {"metric", "threshold", "year_prev",
                                          "filing_id_prev"}}
REQUIRED_GT_KEYS = {"value", "value_str", "filing_id", "section", "evidence_substr"}
FAMILY_GT_KEYS = {
    "financial_metric_extraction": {"value", "value_str", "filing_id",
                                    "section", "evidence_substr"},
    "financial_dual_extraction": {"value_a", "value_a_str", "value_b",
                                  "value_b_str", "filing_id", "section",
                                  "evidence_a", "evidence_b"},
    "financial_yoy_analysis": {"value_now", "value_now_str", "value_prev",
                               "value_prev_str", "yoy_pct", "direction",
                               "threshold", "verdict", "filing_id",
                               "filing_id_prev", "section", "evidence_now",
                               "evidence_prev"},
}


def _money_variants(v: int) -> list[str]:
    return [f"${v:,} million", f"${v:,}", str(v)]


def validate_docs(store: dict) -> list[str]:
    errs = []
    for fid, f in store.items():
        missing = REQUIRED_FILING_KEYS - set(f)
        if missing:
            errs.append(f"{fid}: missing keys {sorted(missing)}")
            continue
        if fid != f["filing_id"]:
            errs.append(f"{fid}: key != filing_id field")
        if not re.fullmatch(r"[A-Z]+-10[KQ]-\d{4}", fid):
            errs.append(f"{fid}: id not TICKER-FORM-YEAR shaped")
        if not f["tables"]:
            errs.append(f"{fid}: empty tables")
        if "financial_statements" not in f["sections"]:
            errs.append(f"{fid}: financial_statements section required")
            continue
        text = f["sections"]["financial_statements"]
        for metric, v in f["values"].items():
            if not any(var in text for var in _money_variants(v)):
                errs.append(f"{fid}: value {metric}={v} has no money string in section")
    return errs


def validate_tasks(tasks: list[dict], store: dict) -> list[str]:
    errs = []
    seen = set()
    for t in tasks:
        missing = REQUIRED_TASK_KEYS - set(t)
        missing |= FAMILY_KEYS.get(t.get("task_family", ""), set()) - set(t)
        if missing:
            errs.append(f"{t.get('task_id', '?')}: missing keys {sorted(missing)}")
            continue
        if t["task_id"] in seen:
            errs.append(f"{t['task_id']}: duplicate task_id")
        seen.add(t["task_id"])
        gt = t["ground_truth"]
        need_gt = FAMILY_GT_KEYS.get(t.get("task_family", ""), REQUIRED_GT_KEYS)
        if need_gt - set(gt):
            errs.append(f"{t['task_id']}: ground_truth missing keys")
            continue
        f = store.get(t["filing_id"])
        if f is None:
            errs.append(f"{t['task_id']}: unknown filing {t['filing_id']}")
            continue
        fam = t.get("task_family", "financial_metric_extraction")
        if fam == "financial_dual_extraction":
            for key in ("evidence_a", "evidence_b"):
                if gt.get(key, "") not in f["sections"].get(t["section"], ""):
                    errs.append(f"{t['task_id']}: {key} not verbatim in section")
            for mk, vk in (("metric_a", "value_a"), ("metric_b", "value_b")):
                if gt.get(vk) != f["values"].get(t.get(mk)):
                    errs.append(f"{t['task_id']}: gt {vk} != store value")
            continue
        if fam == "financial_yoy_analysis":
            for fid_key, ev_key in (("filing_id", "evidence_now"),
                                    ("filing_id_prev", "evidence_prev")):
                ff = store.get(gt.get(fid_key, ""))
                if ff is None:
                    errs.append(f"{t['task_id']}: unknown filing {gt.get(fid_key)}")
                elif gt.get(ev_key, "") not in ff["sections"].get(t["section"], ""):
                    errs.append(f"{t['task_id']}: {ev_key} not verbatim")
            continue
        text = f["sections"].get(t["section"], "")
        if gt["evidence_substr"] not in text:
            errs.append(f"{t['task_id']}: evidence span not verbatim in section")
        if gt["value"] != f["values"].get(t["metric"]):
            errs.append(f"{t['task_id']}: gt value != store value for {t['metric']}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "data" / "real_sec" / "filings.json"))
    ap.add_argument("--tasks", default=str(ROOT / "data" / "real_sec" / "tasks.jsonl"))
    a = ap.parse_args()
    store = json.loads(Path(a.docs).read_text()) if Path(a.docs).exists() else {}
    tasks = ([json.loads(l) for l in Path(a.tasks).read_text().splitlines()]
             if Path(a.tasks).exists() else [])
    if not store:
        print("WARN: empty filings store (scaffold state, nothing to validate)")
    errs = validate_docs(store) + validate_tasks(tasks, store)
    print(f"filings={len(store)} tasks={len(tasks)} errors={len(errs)}")
    for e in errs[:20]:
        print("  ERROR:", e)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
