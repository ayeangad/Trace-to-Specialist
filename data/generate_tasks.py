"""Procedural financial filing + task generator for Trace-to-Specialist V1.

Generates synthetic 10-K-like documents with known ground truth, then
creates metric-extraction tasks with prompt variants (simulating messy
enterprise traces). Split is by COMPANY so frozen_test contains unseen
companies — tests capability, not memorization.

Usage:
    python data/generate_tasks.py --n-train 400 --n-test 100 --seed 42

Outputs:
    data/documents/filings.json      # doc store (served by env)
    data/tasks/train.jsonl
    data/tasks/dev.jsonl
    data/tasks/frozen_test.jsonl
Stdlib only (runs on laptop, no torch needed).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root (hyde/)
DOCS_OUT = ROOT / "data" / "documents" / "filings.json"
TASKS_DIR = ROOT / "data" / "tasks"

TRAIN_COMPANIES = ["ACME Corp", "Globex Inc", "Initech LLC", "Umbrella Co",
                   "Hooli Systems", "Stark Industries", "Wayne Enterprises",
                   "Massive Dynamic"]
# Unseen at train time — frozen generalization check (+ 1 real-SEC placeholder slot)
TEST_COMPANIES = ["Cyberdyne Systems", "Tyrell Corp", "Soylent Corp"]

TICKER = {
    "ACME Corp": "ACME", "Globex Inc": "GLBX", "Initech LLC": "INIT",
    "Umbrella Co": "UMBR", "Hooli Systems": "HOOL", "Stark Industries": "STK",
    "Wayne Enterprises": "WYNE", "Massive Dynamic": "MSDV",
    "Cyberdyne Systems": "CYBD", "Tyrell Corp": "TYRL", "Soylent Corp": "SOYL",
}

METRICS = ["revenue", "net_income", "total_assets", "operating_cash_flow"]
SECTIONS = ["financial_statements", "mda", "notes"]
YEARS = [2022, 2023, 2024]

PROMPT_TEMPLATES = [
    "What was {company}'s {metric} for FY{year}? Cite the filing.",
    "Extract {metric} FY{year} for {company} ({ticker}) from the 10-K.",
    "Find {company} {year} {metric} — give value + evidence.",
    "How much {metric} did {company} report in FY{year}?",
    "{ticker} FY{year} {metric} — lookup and cite.",
]

# --- Experiment 3 difficulty levels (same env, harder jobs) ---
# L2 dual-metric: two values + two evidence spans, one filing.
# Submit protocol: value "VA | VB", evidence "EA || EB".
DUAL_TEMPLATES = [
    "What were {company}'s {metric_a} and {metric_b} for FY{year}? Give both values + evidence.",
    "Extract {metric_a} and {metric_b} FY{year} for {company} ({ticker}). Submit as \"VA | VB\" with evidence \"EA || EB\".",
    "{ticker} FY{year}: report {metric_a} and {metric_b} with cited evidence for each.",
]

# L3 YoY: two filings (Y and Y-1), compute growth, verdict vs threshold.
# Submit protocol: filing_id "ID_A,ID_B", value "PCT|yes|no", evidence "EA || EB".
YOY_TEMPLATES = [
    "Did {company}'s {metric} grow by more than {threshold}% YoY from FY{year0} to FY{year}? Show the calculation and cite both filings.",
    "{ticker} {metric} YoY FY{year0}→FY{year}: compute growth vs {threshold}% threshold. Submit value as \"PCT|yes\" or \"PCT|no\" with evidence \"EA || EB\".",
]


def _money(v: int) -> str:
    return f"${v:,} million"


def make_filing(company: str, year: int, rng: random.Random) -> dict:
    ticker = TICKER[company]
    filing_id = f"{ticker}-10K-{year}"
    base = rng.randint(8_000, 60_000)
    values = {
        "revenue": base,
        "net_income": int(base * rng.uniform(0.05, 0.22)),
        "total_assets": int(base * rng.uniform(1.5, 3.5)),
        "operating_cash_flow": int(base * rng.uniform(0.1, 0.35)),
    }
    tables = {
        "income_statement": {k: values[k] for k in ("revenue", "net_income")},
        "balance_sheet": {"total_assets": values["total_assets"]},
        "cash_flow": {"operating_cash_flow": values["operating_cash_flow"]},
    }
    sections = {
        "financial_statements": (
            f"Consolidated Statements of Operations, FY{year}. "
            f"Revenue was {_money(values['revenue'])}. "
            f"Net income was {_money(values['net_income'])}. "
            f"Total assets were {_money(values['total_assets'])}. "
            f"Operating cash flow was {_money(values['operating_cash_flow'])}."
        ),
        "mda": (
            f"Management Discussion FY{year} for {company}. "
            f"Revenue of {_money(values['revenue'])} reflects operating performance."
        ),
        "notes": f"Notes to financial statements FY{year}. See statements for audited figures.",
    }
    return {
        "filing_id": filing_id, "company": company, "ticker": ticker,
        "year": year, "form": "10-K",
        "source": "synthetic procedural filing (seeded generator, no real issuer)",
        "values": values, "tables": tables, "sections": sections,
    }


def make_task(filing: dict, metric: str, rng: random.Random, tid: str) -> dict:
    template = rng.choice(PROMPT_TEMPLATES)
    prompt = template.format(company=filing["company"], metric=metric.replace("_", " "),
                             year=filing["year"], ticker=filing["ticker"])
    value = filing["values"][metric]
    return {
        "task_id": tid,
        "task_family": "financial_metric_extraction",
        "prompt": prompt,
        "company": filing["company"],
        "ticker": filing["ticker"],
        "year": filing["year"],
        "metric": metric,
        "filing_id": filing["filing_id"],
        "section": "financial_statements",
        "ground_truth": {
            "value": value,
            "value_str": _money(value),
            "filing_id": filing["filing_id"],
            "section": "financial_statements",
            # evidence substring the validator searches for (normalized)
            "evidence_substr": _money(value),
        },
    }


def build(companies: list[str], per_company_tasks: int, seed: int, prefix: str):
    rng = random.Random(seed)
    filings: dict[str, dict] = {}
    tasks: list[dict] = []
    i = 0
    for company in companies:
        for year in YEARS:
            filings[f"{TICKER[company]}-10K-{year}"] = make_filing(company, year, rng)
    filing_list = list(filings.values())
    for _ in range(per_company_tasks * len(companies)):
        f = rng.choice(filing_list)
        m = rng.choice(METRICS)
        tasks.append(make_task(f, m, rng, f"{prefix}-{len(tasks):04d}"))
    return filings, tasks


def make_dual_task(filing: dict, metric_a: str, metric_b: str,
                   rng: random.Random, tid: str) -> dict:
    """L2: two metrics, one filing. See DUAL_TEMPLATES for submit protocol."""
    template = rng.choice(DUAL_TEMPLATES)
    prompt = template.format(
        company=filing["company"], metric_a=metric_a.replace("_", " "),
        metric_b=metric_b.replace("_", " "), year=filing["year"],
        ticker=filing["ticker"])
    va, vb = filing["values"][metric_a], filing["values"][metric_b]
    return {
        "task_id": tid,
        "task_family": "financial_dual_extraction",
        "prompt": prompt,
        "company": filing["company"],
        "ticker": filing["ticker"],
        "year": filing["year"],
        "metric_a": metric_a, "metric_b": metric_b,
        "filing_id": filing["filing_id"],
        "section": "financial_statements",
        "ground_truth": {
            "value_a": va, "value_a_str": _money(va),
            "value_b": vb, "value_b_str": _money(vb),
            "filing_id": filing["filing_id"],
            "section": "financial_statements",
            "evidence_a": _money(va), "evidence_b": _money(vb),
        },
    }


def make_yoy_task(filing_now: dict, filing_prev: dict, metric: str,
                  threshold: float, rng: random.Random, tid: str) -> dict:
    """L3: YoY growth + threshold verdict across two filings (Y and Y-1)."""
    template = rng.choice(YOY_TEMPLATES)
    prompt = template.format(
        company=filing_now["company"], metric=metric.replace("_", " "),
        threshold=threshold, year0=filing_prev["year"], year=filing_now["year"],
        ticker=filing_now["ticker"])
    a, b = filing_now["values"][metric], filing_prev["values"][metric]
    yoy = round((a - b) / b * 100, 2)
    return {
        "task_id": tid,
        "task_family": "financial_yoy_analysis",
        "prompt": prompt,
        "company": filing_now["company"],
        "ticker": filing_now["ticker"],
        "year": filing_now["year"], "year_prev": filing_prev["year"],
        "metric": metric, "threshold": threshold,
        "filing_id": filing_now["filing_id"],
        "filing_id_prev": filing_prev["filing_id"],
        "section": "financial_statements",
        "ground_truth": {
            "value_now": a, "value_now_str": _money(a),
            "value_prev": b, "value_prev_str": _money(b),
            "yoy_pct": yoy,
            "direction": "increase" if yoy >= 0 else "decrease",
            "threshold": threshold,
            "verdict": bool(yoy > threshold),
            "filing_id": filing_now["filing_id"],
            "filing_id_prev": filing_prev["filing_id"],
            "section": "financial_statements",
            "evidence_now": _money(a), "evidence_prev": _money(b),
        },
    }


def build_levels(filings: dict, companies: list[str], n_dual: int, n_yoy: int,
                 seed: int) -> tuple[list[dict], list[dict]]:
    """Build L2/L3 task sets against an EXISTING filings store (same env values
    the L1 tasks use). Year-pair draws make verdicts ~balanced noise (yearly
    bases are independent draws) — the test is compute-from-evidence, not
    economics."""
    rng = random.Random(seed)
    pool = [f for f in filings.values() if f["company"] in companies]
    by_key = {(f["company"], f["year"]): f for f in pool}
    dual, yoy = [], []
    for i in range(n_dual):
        f = rng.choice(pool)
        ma, mb = rng.sample(METRICS, 2)
        dual.append(make_dual_task(f, ma, mb, rng, f"l2-{i:04d}"))
    for i in range(n_yoy):
        company = rng.choice(companies)
        year = rng.choice([2023, 2024])  # needs year-1 >= 2022
        yoy.append(make_yoy_task(by_key[(company, year)],
                                 by_key[(company, year - 1)],
                                 rng.choice(METRICS), 10.0, rng, f"yoy-{i:04d}"))
    return dual, yoy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-dev", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-dual", type=int, default=0,
                    help="Experiment-3 L2 dual-metric tasks (train companies)")
    ap.add_argument("--n-yoy", type=int, default=0,
                    help="Experiment-3 L3 YoY tasks (train companies)")
    ap.add_argument("--n-dual-test", type=int, default=0)
    ap.add_argument("--n-yoy-test", type=int, default=0)
    args = ap.parse_args()

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)

    train_filings, train_tasks = build(TRAIN_COMPANIES, max(1, args.n_train // len(TRAIN_COMPANIES)), args.seed, "tr")
    # dev uses same companies but disjoint task sample (different seed offset)
    _, dev_tasks = build(TRAIN_COMPANIES, max(1, args.n_dev // len(TRAIN_COMPANIES)), args.seed + 1, "dev")
    test_filings, test_tasks = build(TEST_COMPANIES, max(1, args.n_test // len(TEST_COMPANIES)), args.seed + 2, "te")

    all_filings = {**train_filings, **test_filings}
    DOCS_OUT.write_text(json.dumps(all_filings, indent=1))
    for name, tasks in (("train", train_tasks), ("dev", dev_tasks), ("frozen_test", test_tasks)):
        (TASKS_DIR / f"{name}.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")

    # split manifest = freeze record
    manifest = {"seed": args.seed, "train_companies": TRAIN_COMPANIES,
                "test_companies": TEST_COMPANIES,
                "counts": {n: len(t) for n, t in (("train", train_tasks), ("dev", dev_tasks), ("frozen_test", test_tasks))}}
    (ROOT / "data" / "splits" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    print(f"wrote {DOCS_OUT} ({len(all_filings)} filings)")

    if args.n_dual or args.n_yoy or args.n_dual_test or args.n_yoy_test:
        # Experiment-3 sets reference the SAME filings store (same env, harder jobs).
        dual_tr, yoy_tr = build_levels(all_filings, TRAIN_COMPANIES,
                                       args.n_dual, args.n_yoy, args.seed + 10)
        dual_te, yoy_te = build_levels(all_filings, TEST_COMPANIES,
                                       args.n_dual_test, args.n_yoy_test, args.seed + 11)
        for name, rows in (("dual_train", dual_tr), ("yoy_train", yoy_tr),
                           ("dual_test", dual_te), ("yoy_test", yoy_te)):
            if rows:
                (TASKS_DIR / f"{name}.jsonl").write_text(
                    "\n".join(json.dumps(t) for t in rows) + "\n")
        print(f"levels: dual {len(dual_tr)}/{len(dual_te)} yoy {len(yoy_tr)}/{len(yoy_te)} "
              f"(train/test company split, same filings store)")


if __name__ == "__main__":
    main()
