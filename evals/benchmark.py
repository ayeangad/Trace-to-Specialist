"""Validators + metrics + frozen benchmark runner. Stdlib only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.env import FilingEnv  # noqa: E402


def oracle_rollout(task: dict) -> dict:
    """Perfect-policy rollout: correct tools, correct submit. Used for SFT demos + ceiling."""
    fam = task.get("task_family", "financial_metric_extraction")
    if fam in ("financial_dual_extraction", "financial_yoy_analysis"):
        # The collector's oracle already executes against a real env and
        # returns the env-verified reward; reuse it (no string-copy scoring).
        from trajectories.collector import oracle_native_turns
        turns, r10 = oracle_native_turns(task)
        n_tool_turns = sum(1 for t in turns
                           if t.get("role") == "assistant" and "<tool_call>" in t["content"])
        return {"task_id": task["task_id"], "reward": r10,
                "submit_out": "oracle-native", "n_calls": n_tool_turns - 1}
    env = FilingEnv()
    env.reset(task=task)
    gt = task["ground_truth"]
    env.search_filing(f"{task['company']} {task['year']}")
    text = env.open_section(gt["filing_id"], gt["section"])
    # evidence = exact money span from section text
    evidence = gt["evidence_substr"]
    assert evidence in text, f"evidence span missing in {gt['filing_id']}"
    out = env.submit_answer(gt["value_str"], gt["filing_id"], evidence)
    return {"task_id": task["task_id"], "reward": env.get_reward() * 10, "submit_out": out,
            "n_calls": len(env.tool_calls)}


def run_benchmark(tasks_path: str, policy: str = "oracle", limit: int | None = None) -> dict:
    """Run a policy over tasks. policy=oracle only in V1-local (model policies run on Kaggle)."""
    tasks = [json.loads(l) for l in Path(tasks_path).read_text().splitlines()][:limit]
    rewards, succ, calls = [], 0, []
    for t in tasks:
        r = oracle_rollout(t)
        rewards.append(r["reward"])
        succ += int(r["reward"] >= 9)
        calls.append(r["n_calls"])
    n = max(1, len(tasks))
    return {"policy": policy, "n": len(tasks),
            "avg_reward": sum(rewards) / n,
            "success_rate": succ / n,
            "avg_tool_calls": sum(calls) / n if calls else 0}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()
    print(json.dumps(run_benchmark(a.tasks, limit=a.limit), indent=1))
