# Trace-to-Specialist

A miniature implementation of the enterprise intelligence loop: observe how an
organization uses AI, mine traces into tasks, benchmark models per task, route
each task to the cheapest adequate model, and train a specialist where the
volume justifies it.

Built to answer one question: **given messy AI usage traces, can we discover the
real task distribution, find the cheapest model that handles each task, and
specialize a small model until it beats renting frontier intelligence?**

## Result

Financial metric extraction ("pull this number from that filing, prove it"),
frozen test on unseen companies, Base → SFT → GRPO per checkpoint:

| | Base | SFT | GRPO |
|---|---|---|---|
| 0.8B | 4.4 / 0.0 | 2.54 / 0.0 | 2.58 / 0.0 |
| 1.5B | 1.2 / 0.0 | 9.16 / 0.88 | 9.16 / 0.88 |
| 3B | 5.18 / 0.50 | **9.76 / 0.96** | 9.74 / 0.94 |

(avg reward /10, success rate; 90% router threshold)

Specialization pays above the capacity floor and burns money below it. The 3B
specialist clears the bar at 96%; the 0.8B never solves a single task under any
training. RL's record across four regimes (floor, ceiling, compositional middle,
fair middle): 0 for 4 — every residual wall is a capability gap RL can't sample
across at this data scale.

## Quickstart

Laptop (stdlib only, no GPU):

```bash
python data/generate_tasks.py --n-train 400 --n-dev 60 --n-test 100
python evals/benchmark.py --tasks data/tasks/frozen_test.jsonl --limit 20
python trajectories/collector.py --tasks data/tasks/train.jsonl --out /tmp/demos.jsonl
```

GPU (Kaggle free T4, see `kaggle/kaggle_V1.ipynb` for the ordered runbook):

```bash
pip install -q "transformers>=5.2.0" trl peft datasets accelerate huggingface_hub jinja2
python evals/run_model_baseline.py --model Qwen/Qwen2.5-3B-Instruct --tasks data/tasks/frozen_test.jsonl --out experiments/base.jsonl --limit 50
python training/sft.py --model Qwen/Qwen2.5-3B-Instruct --demos data/tasks/sft_native.jsonl --out /kaggle/working/sft
python training/grpo.py --model <merged-ckpt> --tasks data/tasks/train.jsonl --limit 200 --group 4 --max-steps 25
```

## Layout

- `data/generate_tasks.py` — procedural filings + company-split tasks (L1 extract, L2 dual-metric, L3 YoY)
- `environment/` — tool API, stateful `FilingEnv` (TRL-compatible), deterministic 10-pt reward
- `evals/` — oracle ceiling + native tool-loop model harness
- `trajectories/` — oracle transcripts → SFT demos
- `training/` — LoRA SFT, env-based GRPO, adapter merge
- `routing/` — cheapest-model-clearing-90% rule
- `data/real_sec/` — test-only real-filing scaffold (schema + validator, never trained on)
- `report/` — methodology, full result log (`results.md`), failure autopsy (`full_report.md` §7b), codebase deep dive, handover brief

## What I learned

- Exact-copy grounding emerges between 0.8B and 3B. Below it, neither SFT nor a
  properly engaged GRPO loop helps; above it, SFT alone solves the task.
- SFT teaches format, not grounding — and on thin data it can hurt (a 40-demo
  dual-metric run scored 0.0, its twin 5.45).
- Train, eval, and RL must speak one tool protocol (per-model native templates
  from a single `TOOL_SCHEMAS` contract) or every comparison is meaningless.
- Identical scores across different checkpoints means the weights aren't
  loading — the harness now refuses to run instead of silently scoring base.

Details, failure analyses, and the full 26-row experiment log: `report/`.
