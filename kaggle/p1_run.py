"""Phase 1 Kaggle helper: regen demos -> train SFT -> verify frozen test.

Alternative to running the runbook cells by hand (same commands, same gates).
Upload this file to the GPU session along with the repo, then run:

    pip install -q "transformers>=5.2.0" trl peft datasets accelerate huggingface_hub jinja2
    python kaggle/p1_run.py --repo /kaggle/working/hyde --hf-token <HF_TOKEN>

or set the HF_TOKEN env var / Kaggle secret and drop --hf-token.
Steps can be run separately: --steps regen | train | verify | all (default all).

Exits non-zero on any gate failure (see kaggle/tailguard_runbook.md Phase 1).
Stdlib + huggingface_hub only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def sh(cmd: str, cwd: Path) -> str:
    print(f"+ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=str(cwd),
                       capture_output=True, text=True)
    print(p.stdout, end="", flush=True)
    if p.returncode != 0:
        print(p.stderr, flush=True)
        raise SystemExit(f"FAILED ({p.returncode}): {cmd}\n{p.stderr[-2000:]}")
    return p.stdout


def step_regen(repo: Path) -> None:
    sh("python data/generate_tasks.py --n-train 400 --n-dev 60 --n-test 100", repo)
    out = sh("python trajectories/collector.py --tasks data/tasks/train.jsonl "
             "--out data/tasks/sft_demos.jsonl", repo)
    try:
        kept = json.loads(out.strip().splitlines()[-1])["kept"]
    except (json.JSONDecodeError, KeyError, IndexError):
        raise SystemExit(f"could not parse collector kept count from: {out[-500:]}")
    print(f"collector kept={kept}")
    if kept < 350:
        raise SystemExit(f"GATE: collector kept={kept}, expected ~= 400")
    first = json.loads((repo / "data/tasks/sft_demos.jsonl").read_text().splitlines()[0])
    if "turns" not in first:
        raise SystemExit("GATE: regenerated demos lack 'turns' key (old format?)")
    print("demos have 'turns' key: OK")
    changed = sh("git diff --name-only data/", repo).split()
    print("changed data files:", changed)
    if changed != ["data/tasks/sft_demos.jsonl"]:
        raise SystemExit("GATE: generator is not deterministic (files beyond "
                         f"sft_demos.jsonl changed: {changed}). STOP.")


def step_train(repo: Path, adapter_out: str) -> None:
    sh("CUDA_VISIBLE_DEVICES=0 python training/sft.py "
       "--model Qwen/Qwen2.5-3B-Instruct "
       "--demos data/tasks/sft_demos.jsonl "
       f"--out {adapter_out} --epochs 2", repo)
    for f in ("adapter_config.json", "checkpoint-100/adapter_config.json"):
        p = Path(adapter_out) / f
        print(("FOUND " if p.exists() else "MISSING ") + str(p))
        if not p.exists():
            raise SystemExit(f"GATE: {p} missing")


def summarize(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(rows)
    return {"n": n,
            "mean_reward": sum(r["reward"] for r in rows) / n,
            "success": sum(bool(r["success"]) for r in rows) / n,
            "had_submit": sum(bool(r.get("had_submit", False)) for r in rows) / n}


def step_verify(repo: Path, adapter_out: str) -> None:
    out_rel = "experiments/tailguard/p1_frozen_sft3b.jsonl"
    sh(f"python evals/run_model_baseline.py --model {adapter_out} "
       "--base-model Qwen/Qwen2.5-3B-Instruct "
       f"--tasks data/tasks/frozen_test.jsonl --out {out_rel} --limit 50", repo)
    s = summarize(repo / out_rel)
    print("SUMMARY:", json.dumps(s))
    if s["n"] != 50:
        raise SystemExit(f"GATE: expected 50 rows, got {s['n']}")
    if s["success"] < 0.90:
        raise SystemExit(f"GATE: frozen success {s['success']:.3f} < 0.90. "
                         "Confirm native demos + 2 epochs + --base-model; "
                         "re-run once in a fresh session, else STOP.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".",
                    help="repo root (default: current dir)")
    ap.add_argument("--hf-token", default=None,
                    help="HF token (default: HF_TOKEN env var)")
    ap.add_argument("--adapter-out", default="/kaggle/working/qwen25-3b-sft")
    ap.add_argument("--steps", default="all",
                    choices=["all", "regen", "train", "verify"])
    a = ap.parse_args()

    import os
    token = a.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("no HF token: pass --hf-token or set HF_TOKEN")
    from huggingface_hub import login
    login(token=token)
    import torch
    print("cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("GATE: no GPU in this session (need T4)")

    repo = Path(a.repo).resolve()
    if not (repo / "training/sft.py").exists():
        raise SystemExit(f"{repo} is not the repo root")
    print("HEAD:", sh("git rev-parse HEAD", repo).strip())
    print("status:", sh("git status --short", repo).strip() or "clean")

    steps = ["regen", "train", "verify"] if a.steps == "all" else [a.steps]
    if "regen" in steps:
        step_regen(repo)
    if "train" in steps:
        step_train(repo, a.adapter_out)
    if "verify" in steps:
        step_verify(repo, a.adapter_out)
    print("PHASE 1 COMPLETE: all gates passed.")


if __name__ == "__main__":
    main()
