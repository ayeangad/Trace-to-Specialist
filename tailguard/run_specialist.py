"""Run the 3B specialist and capture signals (spec P3.1-P3.6). RUN ON KAGGLE.

Greedy episode + token-level confidence for the submit turn + K sampled
re-answers of the submit turn + opened texts + label. Never calls
env.submit_answer (known harness issue); scores with score_episode directly.

Usage (Kaggle, dataset qwen25-3b-sft-tailguard attached read-only):
  python -m tailguard.run_specialist --adapter /kaggle/input/qwen25-3b-sft-tailguard \\
    --base-model Qwen/Qwen2.5-3B-Instruct --tasks data/tail/tasks.jsonl \\
    --out experiments/tailguard/specialist_runs.jsonl --only-split cal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["HYDE_FILINGS"] = str(ROOT / "data" / "tail" / "filings.json")
sys.path.insert(0, str(ROOT))

from environment.env import FilingEnv  # noqa: E402
from environment.reward import normalize_value, score_episode  # noqa: E402
from environment.tools import TOOL_SCHEMAS  # noqa: E402
from evals.run_model_baseline import parse_native  # noqa: E402
from tailguard.config import (BASE_MODEL, EXP, K_SAMPLES, MAX_NEW_TOKENS,  # noqa: E402
                              MAX_TURNS, PROMPT_CHAR_WINDOW, SAMPLE_TEMPERATURE,
                              SAMPLE_TOP_P, SEED, TAIL_FILINGS)
from tailguard.logging_utils import (append_jsonl, read_done_ids,  # noqa: E402
                                     write_config)
from tailguard.signals.confidence import find_value_token_idx  # noqa: E402

ADAPTER_DEFAULT = "/kaggle/input/qwen25-3b-sft-tailguard"


def seed_for(task_id: str) -> int:
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)


def adapter_sha(adapter_dir: str) -> str:
    h = hashlib.sha256()
    with open(Path(adapter_dir) / "adapter_model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(adapter_dir: str, base_model: str):
    import torch  # noqa: F401 (Kaggle GPU)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mp = Path(adapter_dir)
    if mp.exists() and mp.is_dir():
        if not ((mp / "adapter_config.json").exists() or (mp / "config.json").exists()):
            cands = sorted(str(p.parent) for p in mp.glob("checkpoint-*/adapter_config.json"))
            hint = f" Did you mean one of: {cands}?" if cands else ""
            raise SystemExit(
                f"ERROR: {adapter_dir} has no adapter_config.json or config.json.{hint}")
    try:
        tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    is_adapter = mp.exists() and (mp / "adapter_config.json").exists()
    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def entropy_of(logits_row):
    import torch
    p = torch.softmax(logits_row.float(), -1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum())


def greedy_episode(tok, model, task: dict) -> dict:
    import torch
    env = FilingEnv()
    prompt = env.reset(task=task)
    msgs = [{"role": "user", "content": prompt}]
    turn_records, opened = [], []
    submitted, submit_ctx = None, None
    submit_rec, submit_gen_ids = None, None
    t0 = time.time()
    for turn in range(MAX_TURNS):
        rendered = tok.apply_chat_template(
            msgs, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True)
        enc = tok(rendered[-PROMPT_CHAR_WINDOW:], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, output_scores=True,
                                 return_dict_in_generate=True)
        enc_len = enc["input_ids"].shape[1]
        gen_ids = out.sequences[0, enc_len:]
        text = tok.decode(gen_ids)
        lp = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True)[0]
        ent = [entropy_of(s[0]) for s in out.scores]
        rec = {"turn": turn, "text": text, "gen_len": len(gen_ids),
               "lp": lp.float().cpu().tolist(), "ent": ent}
        name, kwargs = parse_native(text)
        rec["tool"] = name
        turn_records.append(rec)
        if not name:
            msgs.append({"role": "assistant", "content": text})
            break
        if name == "submit_answer":
            submitted = kwargs if isinstance(kwargs, dict) else {}
            submit_ctx = [dict(m) for m in msgs]
            submit_rec = rec
            submit_gen_ids = gen_ids.cpu().tolist()
            break  # never call env.submit_answer
        try:
            fn = getattr(env, name, None)
            if fn is None or not callable(fn):
                raise AttributeError(f"unknown tool {name}")
            result = fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            result = f"ERROR {e}"
        if name == "open_section":
            opened.append({"filing_id": (kwargs or {}).get("filing_id", ""),
                           "section": (kwargs or {}).get("section", ""),
                           "text": result})
        msgs += [{"role": "assistant", "content": text},
                 {"role": "tool", "name": name, "content": result}]
    value_token_idx, value_span_found = [], False
    if submitted and submit_rec is not None:
        v = submitted.get("value", "") or ""
        value_token_idx, value_span_found = find_value_token_idx(
            submit_rec["text"], submit_gen_ids, tok.decode, v)
    if submitted:
        s = score_episode(task, {"tool_calls": env.tool_calls,
                                 "section_cited": env.section_cited},
                          submitted.get("value", ""),
                          submitted.get("filing_id", ""),
                          submitted.get("evidence", ""))
        label = {"success": bool(s["success"]), "reward": float(s["reward"]),
                 "had_submit": True}
    else:
        label = {"success": False, "reward": 0.0, "had_submit": False}
    n_errors = sum(1 for c in env.tool_calls
                   if str(c.get("result", "")).startswith("ERROR"))
    greedy = {"turns": turn_records,
              "submitted": submitted,
              "value_token_idx": value_token_idx,
              "value_span_found": value_span_found,
              "opened": opened,
              "n_tool_calls": len(env.tool_calls),
              "n_tool_errors": n_errors,
              "wall_s": time.time() - t0}
    return greedy, label, submit_ctx


def sample_submit_turn(tok, model, task_id: str, submit_ctx, k: int) -> list[dict]:
    import torch
    if submit_ctx is None or k <= 0:
        return []
    torch.manual_seed(seed_for(task_id))
    rendered = tok.apply_chat_template(
        submit_ctx, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered[-PROMPT_CHAR_WINDOW:], return_tensors="pt").to(model.device)
    enc_len = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                         temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P,
                         num_return_sequences=k)
    samples = []
    for seq in out.sequences:
        text_k = tok.decode(seq[enc_len:])
        name_k, kw_k = parse_native(text_k)
        if name_k == "submit_answer" and isinstance(kw_k, dict):
            samples.append({"tool": name_k,
                            "value_norm": normalize_value(kw_k.get("value")),
                            "filing_id": kw_k.get("filing_id")})
        else:
            samples.append({"tool": name_k, "value_norm": None, "filing_id": None})
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER_DEFAULT)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k-samples", type=int, default=K_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only-split", default=None)
    ap.add_argument("--no-samples", action="store_true")
    a = ap.parse_args()

    assert TAIL_FILINGS.exists(), f"missing store {TAIL_FILINGS}"
    tok, model = load_model(a.adapter, a.base_model)
    sha = adapter_sha(a.adapter)

    tasks = [json.loads(l) for l in Path(a.tasks).read_text().splitlines() if l.strip()]
    if a.only_split:
        tasks = [t for t in tasks if t.get("split") == a.only_split]
    if a.limit:
        tasks = tasks[: a.limit]
    done = read_done_ids(a.out)
    write_config(a.out, vars(a))
    k = 0 if a.no_samples else a.k_samples
    for t in tasks:
        if t["task_id"] in done:
            continue
        greedy, label, submit_ctx = greedy_episode(tok, model, t)
        t0 = time.time()
        samples = sample_submit_turn(tok, model, t["task_id"], submit_ctx, k)
        greedy["wall_s"] += time.time() - t0
        append_jsonl(a.out, {
            "task_id": t["task_id"], "slice": t.get("slice"), "split": t.get("split"),
            "model": {"base": a.base_model, "adapter": a.adapter, "adapter_sha": sha},
            "greedy": greedy, "samples": samples, "label": label})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
