"""Tests for the tail pool generator (spec P2 acceptance). Stdlib only."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TAIL = ROOT / "data" / "tail"

sys.path.insert(0, str(ROOT))
from data.tail.generate_tail import LABEL  # noqa: E402


def _load():
    filings = json.loads((TAIL / "filings.json").read_text())
    tasks = [json.loads(l) for l in (TAIL / "tasks.jsonl").read_text().splitlines()]
    drift = [json.loads(l) for l in (TAIL / "drift_tasks.jsonl").read_text().splitlines()]
    manifest = json.loads((TAIL / "manifest.json").read_text())
    return filings, tasks, drift, manifest


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_deterministic_two_runs_identical_sha():
    files = ["filings.json", "tasks.jsonl", "drift_tasks.jsonl", "manifest.json"]
    before = {f: _sha(TAIL / f) for f in files}
    subprocess.run([sys.executable, "data/tail/generate_tail.py"],
                   cwd=str(ROOT), check=True, capture_output=True)
    after = {f: _sha(TAIL / f) for f in files}
    assert before == after


def test_manifest_sha_match_files():
    _, _, _, manifest = _load()
    for f, h in manifest["sha256"].items():
        assert _sha(TAIL / f) == h, f


def _first_filing_of(filings, fam):
    for fid in sorted(filings):
        if filings[fid]["slice"] == fam:
            return filings[fid]
    raise AssertionError(fam)


def _money(v: int) -> str:
    return f"${v:,} million"


def test_slice_texts_match_templates():
    filings, tasks, drift, _ = _load()
    by_task = {}
    for t in tasks + drift:
        by_task.setdefault((t["slice"], t["filing_id"]), t)

    f = _first_filing_of(filings, "S0")
    v = f["values"]
    y = f["year"]
    assert f["sections"]["financial_statements"] == (
        f"Consolidated Statements of Operations, FY{y}. "
        f"Revenue was {_money(v['revenue'])}. Net income was {_money(v['net_income'])}. "
        f"Total assets were {_money(v['total_assets'])}. "
        f"Operating cash flow was {_money(v['operating_cash_flow'])}.")

    f = _first_filing_of(filings, "T1")
    v = f["values"]
    st = f["sections"]["financial_statements"]
    assert "(In thousands of U.S. dollars.)" in st
    assert f"${v['revenue'] * 1000:,}" in st
    t = next(t for t in tasks if t["filing_id"] == f["filing_id"])
    assert t["ground_truth"]["evidence_substr"] == f"${t['ground_truth']['value'] * 1000:,}"
    assert t["ground_truth"]["value"] == v[t["metric"]]

    f = _first_filing_of(filings, "T2")
    st = f["sections"]["financial_statements"]
    assert f"FY{f['year']}" in st and f"FY{f['year'] - 1}" in st
    t = next(t for t in tasks if t["filing_id"] == f["filing_id"])
    assert t["ground_truth"]["evidence_substr"] == _money(f["values"][t["metric"]])

    f = _first_filing_of(filings, "T3")
    st = f["sections"]["financial_statements"]
    assert st.count("non-GAAP") == 4
    t = next(t for t in tasks if t["filing_id"] == f["filing_id"])
    assert t["ground_truth"]["evidence_substr"] == _money(f["values"][t["metric"]])

    f = _first_filing_of(filings, "T6")
    st = f["sections"]["financial_statements"]
    assert "|" in st
    t = next(t for t in tasks if t["filing_id"] == f["filing_id"])
    assert t["ground_truth"]["evidence_substr"] == \
        f"{LABEL[t['metric']]} | {f['values'][t['metric']]:,}"

    f = _first_filing_of(filings, "D1")
    st = f["sections"]["financial_statements"]
    assert "billion" in st
    t = next(t for t in drift if t["filing_id"] == f["filing_id"])
    assert t["ground_truth"]["value"] == round(f["values"][t["metric"]] / 100) * 100
