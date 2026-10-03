"""Model baseline harness, NATIVE tool loop. RUN ON KAGGLE (needs torch + HF).

Drives FilingEnv through the model family's own chat template + TOOL_SCHEMAS
(Qwen3 JSON vs Qwen3.5 XML-ish handled by the template), parses native
<tool_call> turns, feeds <tool_response> results, and scores via the env reward.
Same protocol TRL's GRPO loop uses — so harness numbers predict GRPO behavior.

Usage (Kaggle):
  python evals/run_model_baseline.py --model Qwen/Qwen3-0.6B --tasks data/tasks/frozen_test.jsonl --out experiments/baseline_qwen06.jsonl --limit 50
  python evals/run_model_baseline.py --model <lora-adapter-dir> --base-model Qwen/Qwen3.5-0.8B-Base --tasks ... --out ...

History: v1 used a custom TOOL:/FINAL: text protocol. GRPO smoke proved base
models never emit TRL's tool format zero-shot, so SFT/eval/GRPO all moved to
the native protocol (v3). Cross-protocol score comparisons are approximate.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Qwen3.5 XML-ish: <tool_call><function=f><parameter=p>v</parameter></function></tool_call>
NATIVE_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE)
PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
# Qwen3 JSON: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
JSON_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_native(text: str):
    """Return (tool_name, kwargs) or (None, None). Native formats first."""
    m = NATIVE_RE.search(text)
    if m:
        kwargs = {k: v.strip() for k, v in PARAM_RE.findall(m.group(2))}
        return m.group(1).strip(), kwargs
    m2 = JSON_CALL_RE.search(text)
    if m2:
        try:
            d = json.loads(m2.group(1))
            name = d.get("name", "")
            args = d.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            return str(name), dict(args)
        except Exception:
            return None, None
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--base-model", default=None,
                    help="Hub id for tokenizer/base weights when --model is a local LoRA adapter dir.")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from environment.env import FilingEnv
    from environment.reward import score_episode
    from environment.tools import TOOL_SCHEMAS

    # Fail loudly on model dirs without weights (2026-10-01 lesson: eval silently
    # scored base weights twice when --model pointed at a checkpoints-only dir).
    mp = Path(a.model)
    if mp.exists() and mp.is_dir():
        if not ((mp / "adapter_config.json").exists() or (mp / "config.json").exists()):
            cands = sorted(str(p.parent) for p in mp.glob("checkpoint-*/adapter_config.json"))
            hint = f" Did you mean one of: {cands}?" if cands else ""
            raise SystemExit(f"ERROR: {a.model} has no adapter_config.json or config.json.{hint}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    base_src = a.base_model or a.model
    try:
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_src, torch_dtype="auto",
                                                 device_map="auto", trust_remote_code=True)
    is_adapter_dir = mp.exists() and (mp / "adapter_config.json").exists()
    if is_adapter_dir and not a.base_model:
        raise SystemExit("ERROR: --model is a LoRA adapter dir; pass --base-model <hub-id>.")
    if is_adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.model)
    model.eval()

    tasks = [json.loads(l) for l in Path(a.tasks).read_text().splitlines()][:a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tpath = Path(a.transcripts) if a.transcripts else None
    if tpath:
        tpath.parent.mkdir(parents=True, exist_ok=True)

    with open(a.out, "w") as f:
        for t in tasks:
            env = FilingEnv()
            prompt = env.reset(task=t)
            msgs = [{"role": "user", "content": prompt}]
            submitted = None
            import torch
            for _ in range(a.max_turns):
                rendered = tok.apply_chat_template(
                    msgs, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True)
                ids = tok(rendered[-12000:], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False)
                text = tok.decode(out[0][ids["input_ids"].shape[1]:])
                name, kwargs = parse_native(text)
                if not name:
                    msgs.append({"role": "assistant", "content": text})
                    break
                try:
                    fn = getattr(env, name, None)
                    if fn is None or not callable(fn):
                        raise AttributeError(f"unknown tool {name}")
                    result = fn(**kwargs)
                except Exception as e:  # noqa: BLE001
                    result = f"ERROR {e}"
                if name == "submit_answer" and isinstance(kwargs, dict):
                    submitted = kwargs
                msgs += [{"role": "assistant", "content": text},
                         {"role": "tool", "name": name, "content": result}]
            if not submitted:
                s = {"reward": 0.0, "success": False, "n_tool_calls": len(env.tool_calls),
                     "had_submit": False}
            else:
                s = score_episode(t, {"tool_calls": env.tool_calls,
                                      "section_cited": env.section_cited},
                                  submitted.get("value", ""), submitted.get("filing_id", ""),
                                  submitted.get("evidence", ""))
                s["had_submit"] = True
            f.write(json.dumps({"task_id": t["task_id"], **s}) + "\n")
            if tpath:
                with open(tpath, "a") as tf:
                    tf.write(json.dumps({"task_id": t["task_id"], "n_turns": len(msgs),
                                         "history": rendered[-6000:] + text[-2000:]}) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
