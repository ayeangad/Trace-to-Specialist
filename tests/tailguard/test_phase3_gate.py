"""Local tests for P3 helpers (no GPU): seed, risk math, gate logic."""
import math

from tailguard.evaluate import conf_risk, phase3_gate
from tailguard.run_specialist import seed_for


def _rec(slice_, success, lp_vals):
    turns = [{"tool": "search_filing", "lp": [0.0], "ent": [1.0]}]
    if lp_vals is not None:
        turns.append({"tool": "submit_answer", "lp": lp_vals,
                      "ent": [1.0] * len(lp_vals)})
        idx = list(range(len(lp_vals)))
    else:
        idx = []
    return {"task_id": "t", "slice": slice_, "split": "cal",
            "greedy": {"turns": turns, "value_token_idx": idx},
            "label": {"success": success, "reward": 10.0, "had_submit": True}}


def test_seed_for_deterministic():
    assert seed_for("tg-cal-0001") == seed_for("tg-cal-0001")
    assert 0 <= seed_for("x") < 2 ** 32


def test_conf_risk_missing_is_inf():
    assert conf_risk(_rec("S0", True, None)) == math.inf
    assert conf_risk(_rec("S0", True, [])) == math.inf
    assert conf_risk({"greedy": {}, "label": {}}) == math.inf


def test_conf_risk_orders_errors(tmp_path):
    import json
    rows = []
    for i in range(20):
        err = i >= 10  # last 10 are errors with low confidence
        lp = [-0.1] * 3 if not err else [-5.0] * 3
        r = _rec("T1" if err else "S0", not err, lp)
        r["task_id"] = f"t-{i}"
        rows.append(r)
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    g = phase3_gate(p)
    assert g["n"] == 20
    assert g["error_rate"] == 0.5
    assert g["tail_share_of_errors"] == 1.0
    assert g["s0_error"] == 0.0
    assert g["auroc_conf_only"] == 1.0
    assert g["go"] is True


def test_gate_no_go_when_too_good(tmp_path):
    import json
    rows = [_rec("S0", True, [-0.1]) for _ in range(20)]
    for i, r in enumerate(rows):
        r["task_id"] = f"t-{i}"
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    g = phase3_gate(p)
    assert g["error_rate"] == 0.0
    assert g["auroc_conf_only"] is None  # single class
    assert g["go"] is False  # e < 0.05
