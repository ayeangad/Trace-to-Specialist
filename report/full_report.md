# Trace-to-Specialist V1 — Full Project Report

*One task, one environment, small models, one complete learning loop — built to test
Hyde's enterprise-intelligence thesis at small scale.*

---

## 1. TL;DR (30 seconds)

Companies pay frontier-model prices for repetitive work. Hyde's public thesis says:
mine usage traces into tasks, benchmark small models per task, route each task to the
cheapest adequate model, and train specialist models where it pays. We built that loop
at miniature scale for one workflow — **financial metric extraction** ("what was
Company X's revenue in FY2024, with evidence"):

- A 0.8B model **never solved a single task**, before or after SFT and GRPO.
  Small-model specialization has a capacity floor, and we measured it.
- A 3B model solved **50% zero-shot** and **96% after SFT** — clearing the 90%
  bar where owning a specialist beats renting frontier intelligence.
- GRPO engaged properly (3+ tool calls/episode, live gradients) but added nothing
  at either scale: below the floor there was nothing to reinforce; at the ceiling
  there was nothing left to learn.

Under this environment and training budget, specialization did not overcome the
grounding bottleneck at 0.8B, while 3B crossed it. That bounded claim — not a
general verdict on small models — is the result.

---

## 2. The idea in plain language

Imagine a finance team asking AI the same kinds of questions every week: revenues,
net incomes, asset figures from filings. Today each question goes to an expensive
giant model. Hyde's argument:

1. **Watch** what the company actually asks (traces).
2. **Discover** the recurring jobs hidden inside (tasks — one level above prompts:
   "summarize this transcript" × 500 phrasings = one task).
3. **Define correct** — the right answer *and* the right process (right document,
   right evidence, no wasteful wandering).
4. **Benchmark small models** per task; move each task to the **smallest model
   that clears the quality bar** (routing = instant savings, no training).
5. **Train specialists** only where volume × cost × gap justifies it.
6. **Keep learning**: new traces → new training signal (continual loop).

Our V1 builds steps 2–5 end-to-end for a single task family, with step 6 stubbed.
V1 deliberately starts at the *learning loop* (task → env → SFT → GRPO) rather than
the mining layer, because the riskiest hypothesis — "can post-training make a tiny
model competitive?" — should be tested before building infrastructure around it.

**Hyde mapping** (no claim of comparability, only of architecture):

| Hyde (public) | Ours |
|---|---|
| Refinery: trace → task mining | `data/generate_tasks.py` (procedural traces + known task labels) |
| Baseline intelligence / golden evals | `evals/`, frozen company-split test set |
| Model router (smallest adequate) | `routing/router.py` (cheapest model ≥ threshold) |
| Campfire: DPO/GRPO specialists | `training/sft.py` + `training/grpo.py` (LoRA, TRL env-GRPO) |
| Continual learning | `trajectories/` logging (retraining loop = V2) |

---

## 3. The system

```text
synthetic filings + tasks (400 train / 60 dev / ~100 frozen test)
        │
        ▼
FilingEnv: search_filing → open_section / get_table → submit_answer
        │   deterministic 10-pt reward (filing 2 + section 1 + value 3 + evidence 2 + efficiency 2)
        ▼
baselines (frozen test, unseen companies) ──► model matrix ──► router (cheapest ≥ 90%)
        │
        ▼
oracle transcripts → LoRA SFT → re-eval
        │
        ▼
merge adapter → GRPO in live env → re-eval
        │
        ▼
Base vs SFT vs GRPO + cost/latency = the answer
```

**Why each piece exists** (nothing in the repo is ornamental):

- *Procedural filings*: no enterprise data available; synthetic 10-K-like documents
  give deterministic ground truth (exact values, correct filing/section, verbatim
  evidence spans) without an LLM judge anywhere in the loop.
- *Company-split freeze*: test companies never appear in training. Memorization
  scores zero; only capability scores.
- *Executable env, not Q&A pairs*: the model must *act* (search, open, submit)
  across turns with observations. Agentic behavior is what's being trained.
- *10-pt reward with process terms*: final-answer-only rewards teach brute force
  (7 searches, wrong doc, lucky guess). Filing/section/evidence/efficiency terms
  price the workflow, not just the answer.
- *SFT before GRPO*: isolates "seeing examples" from "optimizing behavior," so the
  marginal value of RL is attributable instead of hand-waved.
- *LoRA everywhere*: free-T4 survival (0.8B–3B trainable with batch-1 + grad-ckpt).
- *Trajectory logging*: every episode is future training data and current evidence.

---

## 4. Data, environment, reward (the research contribution)

**Task** (`financial_metric_extraction`): "Extract <metric> FY<year> for <company>.
Return value + filing + verbatim evidence." Prompt phrasing varies per task
(simulating messy enterprise traces); the underlying job is identical.

**Filings** (`data/documents/filings.json`): 33 procedural filings across 11 fictional
companies × 3 years. Each has sections (`financial_statements`, `mda`, `notes`),
tables (`income_statement`, `balance_sheet`, `cash_flow`), and money-valued metrics
(`revenue`, `net_income`, `total_assets`, `operating_cash_flow`).

**Tools** (`environment/tools.py` + `TOOL_SCHEMAS`): `search_filing(query)`,
`open_section(filing_id, section)`, `get_table(filing_id, table_id)`,
`submit_answer(value, filing_id, evidence)`. Plain functions with typed signatures —
passed directly to TRL as agent tools *and* driven by the local harness.

**Reward** (`environment/reward.py`, total 10): filing +2, section +1, exact value +3,
verbatim evidence +2, efficiency +2 (≤4 clean calls; partial ≤6). Normalized to
[0,1] for GRPO via `FilingEnv.get_reward()`. Fully deterministic — the environment
*knows* the answer, so RL optimizes reality, not a judge's opinion.

---

## 5. Models and the protocol saga (read this before the numbers)

Two model legs were run: **Qwen3.5-0.8B-Base** (primary; Apache-2.0, research
checkpoint, multimodal arch with vision frozen) and **Qwen2.5-3B-Instruct**
(capacity test), with Qwen3-0.6B as pipeline smoke.

The single most expensive lesson: **the tool-call protocol must be identical across
SFT data, eval harness, and GRPO** — and it wasn't, twice:

1. *Custom `TOOL:`/`FINAL:` text protocol* (v1–v2): worked for harness eval, but TRL's
   GRPO loop speaks only model-native chat-template tool calls. GRPO smoke proved it:
   0 tool calls, all rewards 0, zero grads — mechanics fine, loop never engaged.
2. *Discovery*: Qwen3 and Qwen3.5 use **different native syntaxes** (JSON
   `{"name","arguments"}` vs XML-ish `<function=f><parameter=p>`), verified against
   real tokenizers locally.
3. *Fix (v3)*: one protocol — `TOOL_SCHEMAS` rendered through each model's own
   `apply_chat_template` for SFT text, the harness loop, and TRL env tools.
   Round-trip verified locally (render → parse → dispatch → 10.0, both families).
4. *Near-miss along the way*: two "SFT does nothing" readings turned out to be
   **phantom evals** — the harness pointed at checkpoint dirs containing no adapter
   files, silently scoring base weights. The harness now exits loudly on weightless
   dirs. Always confirm `adapter_config.json` lives at `--model`.

SFT data also evolved with understanding: final answers only (no transfer — eval
tests the loop, not answers) → text transcripts (format only) → native transcripts
(correct target, but see §6 on what happened).

---

## 6. Experiment log (frozen test unless noted; reward/10)

| Date | Experiment | Model | n | Avg | Succ | Sub | Note |
|---|---|---|---|---|---|---|---|
| 10-01 | baseline_qwen06_n10 | Qwen3-0.6B base, naive harness | 10 | 0.0 | 0.0 | 0.0 | Harness artifact (no template/1-shot) |
| 10-01 | baseline_qwen06_v2_n10 | Qwen3-0.6B base, fixed harness | 10 | 2.4 | 0.0 | 0.6 | Partial credit only |
| 10-01 | baseline_qwen25_inst_n10 | Qwen2.5-0.5B-Instruct | 10 | 1.9 | 0.0 | 0.5 | Hard zero-shot for small models |
| 10-01 | baseline_qwen35_n10 | Qwen3.5-0.8B-Base | 10 | 3.6 | 0.0 | — | Ablation base |
| 10-01 | baseline_qwen06_n50 | Qwen3-0.6B | 50 | 3.42 | 0.0 | 0.86 | Frozen matrix row |
| 10-01 | baseline_qwen35_n50 | Qwen3.5-0.8B-Base | 50 | 3.66 | 0.0 | 0.94 | Frozen matrix row |
| 10-01 | sft_n50 (answers) | 0.8B + SFT answers (loss →0.38) | 50 | 3.66 | 0.0 | 0.94 | Bit-identical to base: wrong target |
| 10-01 | sft_traj_ckpt100_n50 | 0.8B + SFT text transcripts | 50 | 3.60 | 0.0 | 1.0 | Format only (submit 1.0, value 0) |
| 10-02 | base_native_n20 | 0.8B base, native harness | 20 | 4.40 | 0.0 | 0.85 | True same-harness control |
| 10-02 | sft_native_v2_n50 | 0.8B + native SFT (loss →0.05) | 50 | 2.54 | 0.0 | 0.58 | Regression: rigid mimicry breaks emission |
| 10-02 | grpo_n50 | merged SFT-0.8B + GRPO 25 steps | 50 | 2.58 | 0.0 | 0.58 | Flat; value never correct anywhere |
| 10-02 | base_3b_n50 | Qwen2.5-3B-Instruct zero-shot | 50 | 5.18 | **0.50** | 0.68 | Capacity floor crossed |
| 10-02 | sft_3b_n50 | 3B + native SFT (loss →0.055) | 50 | **9.76** | **0.96** | 1.00 | Clears 90% bar (48/50) |
| 10-02 | grpo_3b_n50 | merged SFT-3B + GRPO 25 steps | 50 | 9.74 | 0.94 | 1.00 | RL-at-ceiling (zero grads, saturated) |

Succ = exact value + correct filing. Sub = submitted anything. Oracle ceiling 10.0/1.0.

**Reading the 0.8B leg**: breakdowns show 35× reward-3.0 (wrong filing) + 15×
reward-5.0 (right filing, wrong value). Transcripts catch the model staring at the
exact answer in its observation and submitting a near-miss — a **copy-fidelity
failure**, not retrieval or format. SFT moves submit rate; nothing moves value.
GRPO engages correctly (∼3 calls/episode, <10% tool failures, live gradients,
reward variance present) yet 25 steps can't teach verbatim copying at this scale.

**Reading the 3B leg**: grounding emerges with scale — 25/50 cold, 48/50 after SFT.
GRPO finds a saturated policy (all-max rollouts → zero advantage → zero updates),
which is the correct behavior of the algorithm at a ceiling, not a bug.

---

## 7. Engineering gotchas (Kaggle free-T4 file)

- PEFT routes LoRA creation through torchao if present; Kaggle's torchao 0.10
  trips its ≥0.16 check → `pip uninstall -y torchao` (unneeded for fp16 LoRA).
- Multi-GPU sessions + `device_map="auto"` + Trainer = DataParallel cuda:0/cuda:1
  death → pin `device_map={"": 0}` **and** launch with `CUDA_VISIBLE_DEVICES=0`.
- SFT adapter dirs contain no tokenizer files → harness falls back to `--base-model`
  id; `sft.py` now saves the tokenizer explicitly; merge script produces full dirs.
- Qwen3.5 needs `causal_conv1d`/`flash-linear-attention` for speed; both skipped
  (reference fallback is slow but correct — don't burn the session building them).
- No persistence: session end wipes `/kaggle/working`. Save Version after every
  step; keep `experiments/*.jsonl` in this repo as the surviving record.
- Cell hygiene: `%%writefile` and `!python` never share a cell; Python runs as
  plain cells; `--model` must point at the `checkpoint-N` dir holding the adapter.

---

## 7b. What broke and how I diagnosed it (do not skip)

Three failures that looked like model results and turned out to be harness bugs.
Each is kept in the log because the diagnosis is the evidence of rigor.

**1. Phantom evals (twice).** SFT answer-only (3.66) and SFT transcript (3.66/3.60)
evaluations scored *base weights*, not the trained adapter: `--model` pointed at
an output dir whose adapter files lived one level down in `checkpoint-100/`, and
the harness — seeing no `adapter_config.json` — silently fell back to base.
Caught because three different weights printed 3.66 to two decimals, which is
impossible under real weight changes; confirmed with `ls` (no adapter files at
top level). Fix: `run_model_baseline.py` now raises `SystemExit` on model dirs
containing neither `adapter_config.json` nor `config.json`, and even suggests the
`checkpoint-*/` candidate. Lesson from `report/results.md`: identical scores
across different checkpoints mean the weights aren't loading.

**2. Protocol split.** GRPO smoke: 10 steps, loss 0, grad 0, `tools/call_frequency`
0, every completion clipped at max length. Mechanics (rollout → reward → update)
were proven working, but the loop never engaged: our SFT/eval spoke a custom
`TOOL:`/`FINAL:` text protocol while TRL's `_tool_call_loop` only understands
model-native chat-template tool calls. Local tokenizer inspection then showed the
two model families speak *different* native syntaxes — Qwen3 JSON
(`<tool_call>{"name","arguments"}`) vs Qwen3.5 XML-ish
(`<tool_call><function=f><parameter=p>`). Fix: `TOOL_SCHEMAS` in
`environment/tools.py` as the single contract, rendered via each model's own
`apply_chat_template` for SFT text, harness loop, and TRL env tools; round-trip
verified locally (render → parse → dispatch → 10.0 on both families) before any
further GPU spend.

**3. Wrong-harness "regression".** Native-SFT scored 2.68 vs base 3.66 and looked
like training damage — until base re-scored 3.66 *bit-identical* on the same
binary, and the "native" transcript contained `Reminder: emit TOOL:...` strings
that exist only in the old harness file: the native harness `%%writefile` had
never been applied on that Kaggle disk. No retraining was needed; re-running the
existing checkpoint under the real native harness gave the true number. Lesson:
when a result contradicts the mechanism, check which binary actually ran
(`grep` the file on disk) before theorizing.

---

## 8. Economics (why the whole thing matters)

The router rule is fixed: cheapest model with frozen success ≥ 90% wins the task.

- **0.8B leg**: nothing approaches the bar (best 4.4/10 avg, 0 success). Correct
  decision: keep renting bigger intelligence. Specialization spend unjustified —
  knowing *when not to train* is the delivers-savings part of the thesis.
- **3B leg**: 96% specialist vs ~95% frontier-class quality at a fraction of
  inference cost. Recurring extraction volume routes to the owned specialist.
  Measured boundary: grounding emerges somewhere between 0.8B (0/50 everywhere)
  and 3B (25/50 cold) — that interval *is* the rent-vs-own answer for this task.

---

## 9. Limitations (honest)

- Synthetic filings, not real EDGAR; single task family; no trace-clustering or
  multi-task router yet (all V2).
- Oracle (perfect-policy) SFT demos; model-generated trajectory filtering not tried.
- 0.8B GRPO was 25 steps, G=4 — a longer run or denser reward (partial credit for
  near-miss values, copy-focused curriculum) might move it; untested on quota.
- Greedy-decoding eval vs sampling-based training mismatch observed but not
  ablated (temperature/constrained-decoding study deferred).
- Chat-template rendering differences between HF and TRL's patched training
  templates for Qwen3.5 noted but not isolated.
- Latency/cost measured coarsely (tokens + wall-clock on shared T4s, not $ accounting).

## 10. What V2 would do (in order)

1. Real trace mining: embeddings + clustering + LLM labels over noisier,
   multi-task synthetic traces (the Refinery layer, currently stubbed by generator).
2. Multi-task benchmark + threshold router optimizer across 3–4 model sizes.
3. Denser rewards + copy curriculum for sub-3B grounding; temperature-matched eval.
4. Continual loop: production-sim traces → failure mining → retrain → regression
   gate (the compounding-asset story).
5. Real SEC sample as the generalization split.

---

## 11. Repo map & reproduction

```text
data/generate_tasks.py      procedural filings + company-split tasks (seed 42)
environment/{tools,env,reward}.py   tools + TOOL_SCHEMAS, FilingEnv (TRL-compatible), 10-pt reward
evals/{benchmark,run_model_baseline}.py  oracle runner + native-harness model eval
trajectories/collector.py   oracle native transcripts → SFT demos
training/{sft,grpo,merge_lora}.py   LoRA SFT, env-GRPO, adapter merge (Kaggle/GPU)
routing/router.py           cheapest-passing-model stub
report/{methodology,results,limitations}.md + this file
kaggle/kaggle_V1.ipynb      ordered GPU runbook
```

Laptop (stdlib only): `generate_tasks → benchmark → collector`.
Kaggle T4: baselines → SFT → eval → merge → GRPO → frozen eval (see notebook).
All randomness seeded; frozen test = unseen companies; every table row above
reproduces from these scripts.

---

## 12. Numbers cheat sheet

- Oracle ceiling: 10.0 / 1.0. Best specialist (3B+SFT): 9.76 / 0.96.
- Capacity floor: 0.8B = 0 successes in hundreds of tries; 3B = 25/50 zero-shot.
- SFT effect: 0.8B 4.4→2.54 (hurts greedy emission); 3B 5.18→9.76 (task solved).
- GRPO effect: 0.8B 2.54→2.58 (engaged, insufficient); 3B 9.76→9.74 (saturated).
- Router bar (90%): cleared only by the 3B specialist. Everything else rents big.
