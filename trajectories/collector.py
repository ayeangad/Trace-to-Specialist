"""Trajectory collection: oracle NATIVE transcripts for SFT. Stdlib only.

Writes rows {task_id, turns, reward} where turns are semantic conversation
turns (user/tool/assistant with native <tool_call> XML). SFT text is rendered
from turns via the model's own apply_chat_template(tools=TOOL_SCHEMAS), so the
training format is byte-consistent with what TRL's tool loop feeds at GRPO
time — for both Qwen3 (JSON) and Qwen3.5 (XML-ish) syntaxes.

History:
  - v1 answer-only demos: train loss fell, frozen behavior unchanged (sft_n50 == base).
  - v2 TOOL:/FINAL: text transcripts: format compliance only (submit 1.0, success 0).
  - v3 (this): model-native tool-call transcripts. The GRPO smoke proved the
    base model never emits TRL's tool format zero-shot (0 tool calls); SFT must
    teach exactly that syntax.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from environment.env import FilingEnv  # noqa: E402


def tool_call(name: str, params: dict) -> str:
    inner = "".join(f"<parameter={k}>{v}</parameter>" for k, v in params.items())
    return f"<tool_call><function={name}>{inner}</function></tool_call>"


def oracle_native_turns(task: dict) -> tuple[list[dict], float]:
    """Perfect-policy episode as semantic turns + reward/10. Dispatches on
    task_family (L1 single, L2 dual, L3 YoY)."""
    fam = task.get("task_family", "financial_metric_extraction")
    if fam == "financial_dual_extraction":
        return _oracle_dual(task)
    if fam == "financial_yoy_analysis":
        return _oracle_yoy(task)
    return _oracle_l1(task)


def _oracle_l1(task: dict) -> tuple[list[dict], float]:
    env = FilingEnv()
    prompt = env.reset(task=task)
    gt = task["ground_truth"]
    q = f"{task['company']} {task['year']}"
    obs1 = env.search_filing(q)
    obs2 = env.open_section(gt["filing_id"], gt["section"])
    assert gt["evidence_substr"] in obs2, f"evidence span missing in {gt['filing_id']}"
    ack = env.submit_answer(gt["value_str"], gt["filing_id"], gt["evidence_substr"])
    turns = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": tool_call("search_filing", {"query": q})},
        {"role": "tool", "name": "search_filing", "content": obs1},
        {"role": "assistant", "content": tool_call(
            "open_section", {"filing_id": gt["filing_id"], "section": gt["section"]})},
        {"role": "tool", "name": "open_section", "content": obs2},
        {"role": "assistant", "content": tool_call(
            "submit_answer", {"value": gt["value_str"], "filing_id": gt["filing_id"],
                              "evidence": gt["evidence_substr"]})},
        {"role": "tool", "name": "submit_answer", "content": ack},
        {"role": "assistant", "content": (
            f"{gt['value_str']} (filing {gt['filing_id']}, "
            f"evidence: \"{gt['evidence_substr']}\")")},
    ]
    return turns, env.get_reward() * 10


def _oracle_dual(task: dict) -> tuple[list[dict], float]:
    """L2 oracle: search, open, submit combined "VA | VB" / "EA || EB"."""
    env = FilingEnv()
    prompt = env.reset(task=task)
    gt = task["ground_truth"]
    q = f"{task['company']} {task['year']}"
    obs1 = env.search_filing(q)
    obs2 = env.open_section(gt["filing_id"], gt["section"])
    assert gt["evidence_a"] in obs2 and gt["evidence_b"] in obs2
    value = f"{gt['value_a_str']} | {gt['value_b_str']}"
    evid = f"{gt['evidence_a']} || {gt['evidence_b']}"
    ack = env.submit_answer(value, gt["filing_id"], evid)
    turns = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": tool_call("search_filing", {"query": q})},
        {"role": "tool", "name": "search_filing", "content": obs1},
        {"role": "assistant", "content": tool_call(
            "open_section", {"filing_id": gt["filing_id"], "section": gt["section"]})},
        {"role": "tool", "name": "open_section", "content": obs2},
        {"role": "assistant", "content": tool_call(
            "submit_answer", {"value": value, "filing_id": gt["filing_id"],
                              "evidence": evid})},
        {"role": "tool", "name": "submit_answer", "content": ack},
        {"role": "assistant", "content": (
            f"{value} (filing {gt['filing_id']}, evidence: \"{evid}\")")},
    ]
    return turns, env.get_reward() * 10


def _oracle_yoy(task: dict) -> tuple[list[dict], float]:
    """L3 oracle: search, open year-Y, open year-(Y-1), submit combined
    "PCT|yes|no" with both evidence spans. Final text shows the arithmetic."""
    env = FilingEnv()
    prompt = env.reset(task=task)
    gt = task["ground_truth"]
    q = f"{task['company']} {task['year']}"
    obs1 = env.search_filing(q)
    obs_a = env.open_section(gt["filing_id"], gt["section"])
    obs_b = env.open_section(gt["filing_id_prev"], gt["section"])
    assert gt["evidence_now"] in obs_a and gt["evidence_prev"] in obs_b
    value = f"{gt['yoy_pct']:.2f}%|{'yes' if gt['verdict'] else 'no'}"
    evid = f"{gt['evidence_now']} || {gt['evidence_prev']}"
    ack = env.submit_answer(value, f"{gt['filing_id']},{gt['filing_id_prev']}", evid)
    calc = (f"({gt['value_now']}-{gt['value_prev']})/{gt['value_prev']}*100 "
            f"= {gt['yoy_pct']:.2f}%, {'above' if gt['verdict'] else 'below or at'} "
            f"{gt['threshold']}% threshold")
    turns = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": tool_call("search_filing", {"query": q})},
        {"role": "tool", "name": "search_filing", "content": obs1},
        {"role": "assistant", "content": tool_call(
            "open_section", {"filing_id": gt["filing_id"], "section": gt["section"]})},
        {"role": "tool", "name": "open_section", "content": obs_a},
        {"role": "assistant", "content": tool_call(
            "open_section", {"filing_id": gt["filing_id_prev"], "section": gt["section"]})},
        {"role": "tool", "name": "open_section", "content": obs_b},
        {"role": "assistant", "content": tool_call(
            "submit_answer",
            {"value": value,
             "filing_id": f"{gt['filing_id']},{gt['filing_id_prev']}",
             "evidence": evid})},
        {"role": "tool", "name": "submit_answer", "content": ack},
        {"role": "assistant", "content": f"{value} ({calc})"},
    ]
    return turns, env.get_reward() * 10


def collect(tasks_path: str, out_path: str, min_reward: float = 9.0) -> dict:
    tasks = [json.loads(l) for l in Path(tasks_path).read_text().splitlines()]
    kept, dropped = [], 0
    for t in tasks:
        try:
            turns, r = oracle_native_turns(t)
        except AssertionError:
            dropped += 1
            continue
        if r >= min_reward:
            kept.append({"task_id": t["task_id"], "turns": turns, "reward": r})
        else:
            dropped += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    return {"kept": len(kept), "dropped": dropped, "out": out_path}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(collect(a.tasks, a.out), indent=1))
