"""Tail pool validator (spec P2.7). Exits 1 on any failure, prints each.

Checks:
1. filing_id exists; evidence_substr is a substring of financial_statements.
2. value == values[metric] (S0-T6), == round(values/100)*100 (D1).
3. No company in more than one split.
4. Tickers unique; first words unique; ticker.lower() not a substring of any
   template string, synonym, or other company's name.
5. search_filing("{company} {year}") includes the task's filing_id
   (HYDE_FILINGS=data/tail/filings.json).
6. Counts match N_TASKS / SLICE_SHARE as computed in P2.5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ["HYDE_FILINGS"] = str(ROOT / "data" / "tail" / "filings.json")
sys.path.insert(0, str(ROOT))

from data.generate_tasks import PROMPT_TEMPLATES  # noqa: E402
from data.tail.generate_tail import (T4_TEMPLATES, T4_SYN, T5_TEMPLATES,  # noqa: E402
                                     split_counts)
from environment.tools import reload_store, search_filing  # noqa: E402
from tailguard.companies import COMPANIES  # noqa: E402
from tailguard.config import N_DRIFT, N_TASKS, SEED, SLICES, SPLITS  # noqa: E402

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print("FAIL:", msg)


def check_tasks(path: Path, filings: dict, counts: Counter,
                company_splits: dict) -> int:
    n = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        n += 1
        gt = t["ground_truth"]
        fid = t["filing_id"]
        # 1
        if fid not in filings:
            fail(f"{t['task_id']}: unknown filing {fid}")
            continue
        f = filings[fid]
        if gt["evidence_substr"] not in f["sections"]["financial_statements"]:
            fail(f"{t['task_id']}: evidence not in statements")
        # 2
        if t["slice"] == "D1":
            want = round(f["values"][t["metric"]] / 100) * 100
        else:
            want = f["values"][t["metric"]]
        if gt["value"] != want:
            fail(f"{t['task_id']}: value {gt['value']} != {want}")
        # 3
        company_splits.setdefault(t["company"], set()).add(t["split"])
        # 6 (count later)
        counts[(t["split"], t["slice"])] += 1
        # 5
        hits = search_filing(f"{t['company']} {t['year']}").split("\n")
        if fid not in hits:
            fail(f"{t['task_id']}: search missed {fid} (hits={hits})")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tail-dir", default=str(ROOT / "data" / "tail"))
    a = ap.parse_args()

    tail = Path(a.tail_dir)
    filings = json.loads((tail / "filings.json").read_text())
    n_f = reload_store(str(tail / "filings.json"))
    print(f"store: {n_f} filings")

    counts: Counter = Counter()
    company_splits: dict[str, set] = {}
    n_tasks = check_tasks(tail / "tasks.jsonl", filings, counts, company_splits)
    n_drift = check_tasks(tail / "drift_tasks.jsonl", filings, counts,
                          company_splits)
    print(f"tasks: {n_tasks}, drift: {n_drift}")

    # 3
    for c, ss in company_splits.items():
        if len(ss) > 1:
            fail(f"company {c} in splits {sorted(ss)}")

    # 4
    tickers = [t for _, t in COMPANIES]
    if len(set(tickers)) != len(tickers):
        fail("duplicate tickers")
    firsts = [c.split()[0] for c, _ in COMPANIES]
    if len(set(firsts)) != len(firsts):
        fail("duplicate first words")
    template_strs = list(PROMPT_TEMPLATES) + T4_TEMPLATES + T5_TEMPLATES
    synonyms = [s for v in T4_SYN.values() for s in v]
    names = [c for c, _ in COMPANIES]
    for c, t in COMPANIES:
        tl = t.lower()
        for s in template_strs:
            if tl in s.lower():
                fail(f"ticker {t} substring of template {s[:60]!r}")
        for s in synonyms:
            if tl in s.lower():
                fail(f"ticker {t} substring of synonym {s!r}")
        for other in names:
            if other != c and tl in other.lower():
                fail(f"ticker {t} substring of company {other!r}")

    # 6
    for s in SPLITS:
        for sl in SLICES:
            want = split_counts(s)[sl]
            got = counts.get((s, sl), 0)
            if got != want:
                fail(f"count ({s},{sl}): got {got} want {want}")
    if counts.get(("drift", "D1"), 0) != N_DRIFT:
        fail(f"drift count {counts.get(('drift','D1'),0)} != {N_DRIFT}")
    if n_tasks != sum(N_TASKS.values()):
        fail(f"total {n_tasks} != {sum(N_TASKS.values())}")

    if FAILURES:
        print(f"{len(FAILURES)} failures")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
