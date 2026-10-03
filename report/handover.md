# Trace-to-Specialist — Handover Brief

*Purpose: feed this whole file to an AI (or a human) for an intuitive, complete,
accurate understanding of the project — what was built, what happened, what it
means, and how to talk about it. Every number matches `report/results.md`; every
mechanism matches the code. No aspirations are stated as results.*

---

## PART A — THE STORY IN 2 MINUTES

A company asks AI the same kinds of finance questions every week. Today each one
goes to an expensive giant model. Hyde's public thesis says: mine usage into
tasks, benchmark small models per task, route each task to the cheapest adequate
model, and train specialist models where the volume justifies it.

We built that loop miniature-style for one workflow — pulling numbers out of
filings with cited evidence — and ran the full Base → SFT → GRPO comparison on
frozen tests, at two model scales and three difficulty levels:

- **0.8B models never solved a single task** — before training, after SFT, after
  GRPO. Hundreds of attempts, zero full successes. Small-model specialization has
  a capacity floor, and we measured it from below.
- **A 3B model solved 50% cold and 96% after SFT**, clearing the 90% bar where
  owning a specialist beats renting frontier intelligence.
- **GRPO was tested at three points and added nothing anywhere**: at the floor
  (nothing to reinforce), at the ceiling (nothing left to learn), and in the
  middle (L2 dual tasks — engaged properly, 25 steps, drifted slightly down).
  Each is an honest data point about *when RL helps*, not a verdict that RL is useless.

The one-line result, stated defensibly: *under this environment and training
budget, specialization did not overcome the grounding bottleneck at 0.8B, while
3B crossed it — and the router decision writes itself from those two facts.*

---

## PART B — THE MACHINE (WHAT EACH PART DOES AND WHY)

**World + tasks — `data/generate_tasks.py`.** Procedurally generates 33 fake
10-K-like filings (11 fictional companies × 3 years; 8 train companies, 3 held-out
test companies) with exact known figures, then samples tasks against them. Three
levels, same store: **L1** single-metric extraction (400/56/99 train/dev/test),
**L2** dual-metric one filing (40/20), **L3** YoY growth + threshold verdict
across two yearly filings (40/20). Prompt phrasing varies per task — that variation
*is* the simulated "messy enterprise traces." Company-split test sets mean
memorization scores zero. Note: `dev.jsonl` is generated but no script reads it;
L2/L3 train sets are small (40 each) by construction — that thinness matters later.

**Tools — `environment/tools.py`.** Three stateless functions over the filing
store: `search_filing` (substring matching over company/ticker/year, top-5 sorted,
`NO_RESULTS` fallback), `open_section` (verbatim text or `ERROR:` strings —
errors are scored data, not crashes), `get_table` (metric dict as JSON; exposed
but never needed by the oracle path). Plus `TOOL_SCHEMAS`: the same four methods
(including `submit_answer`) as OpenAI-style function specs — the single protocol
contract rendered through each model's own chat template, so Qwen3's JSON tool
syntax and Qwen3.5's XML-ish syntax both work from one source. `HYDE_FILINGS` env
var repoints the store (the real-SEC mechanism) with zero code changes.

**Environment — `environment/env.py` (`FilingEnv`).** Wraps the tools in a
per-episode stateful object matching TRL's `environment_factory` contract:
`reset()` starts an episode (TRL calls it with no args → self-samples a task;
local code may pass a task explicitly), public methods are the agent's tools,
`get_reward()` returns the episode score ÷ 10. Tracks every call and the cited
section. The model must *act across turns with observations* — that multi-step
structure is what makes this an agent problem rather than Q&A.

**Reward — `environment/reward.py` (`score_episode`, always out of 10).**
L1: filing 2 + section 1 + exact value 3 + verbatim evidence 2 + efficiency
2 (≤4 clean calls; 1 if ≤6). L2: filing 2 + section 1 + valueA 2 + valueB 2 +
evidence 1+1 + efficiency 1. L3: filingA 1 + filingB 1 + section 1 + YoY number
within ±0.05 → 2 + verdict 2 + evidence 1+1 + efficiency 1. Multi-value protocols
ride inside the existing strings: `"VA | VB"`, `"EA || EB"`, filings `"A,B"`,
value `"PCT|yes|no"`. Success always requires the graded outcome (value/filing/
verdict), never just participation. Fully deterministic — the env *knows* the
answer, so RL optimizes reality, not a judge's opinion.

**Harness — `evals/run_model_baseline.py`.** Drives any checkpoint through the
native tool loop (up to 6 turns, 256 tokens/turn), parses native tool calls,
feeds results back, scores via the env. Handles local LoRA dirs via `--base-model`
(base weights + tokenizer from Hub, adapter layered on). Refuses to run on
weightless dirs (the phantom-eval fix). `evals/benchmark.py` runs the perfect
oracle policy (ceiling: 10.0/1.0, 2 calls L1/L2, 3 calls L3).

**Training — `training/`.** `sft.py`: LoRA r16/α32 on q/k/v/o, batch-1 × accum-8,
lr 2e-4, fp16, grad-checkpointing, single-GPU pinned, text rendered from native
turns via the model's own template. `grpo.py`: LoRA-wrapped GRPO
(`environment_factory` self-sampling envs, G=4, no vLLM on T4, `scale_rewards=False`,
explicit `max_steps` because the env owns the data). `merge_lora.py`: folds the
SFT adapter into base weights so GRPO starts from merged weights + a fresh adapter
instead of stacking adapters.

**Collector — `trajectories/collector.py`.** Runs the oracle policy into native
turn transcripts (`{task_id, turns, reward}`, reward ≥ 9 filter). Three generations
of demo format live in its docstring as history: answers-only → text-protocol →
native turns.

**Router — `routing/router.py`.** Stub encoding the commercial rule: cheapest model
with frozen success ≥ 90%, else `None` (keep frontier). Demo numbers only.

---

## PART C — THE FULL RESULT STORY (CHRONOLOGICAL)

**Phase 1 — Baselines (0.8B leg).** Raw base models: Qwen3-0.6B 0.0 (naive prompt;
harness artifact), then 2.4 with a fixed harness; Qwen2.5-0.5B-Instruct 1.9;
Qwen3.5-0.8B-Base 3.6–3.66 with 0.94 submit rate. Everyone retrieves, nobody
extracts exactly. Established: the task is hard zero-shot, partial credit flows,
full success never lands.

**Phase 2 — SFT saga (0.8B).** Answers-only SFT: loss 2.93→0.38, frozen score
bit-identical 3.66 — it taught answer style while the eval tests the tool loop.
Text-transcript SFT: submit rate 0.94→1.0, still 0 success — format without
grounding. Native-transcript SFT: 2.54 with submit *regressing* to 0.58 — rigid
mimicry that breaks emission. Each step reinterpreted by the next control;
the base-native control (4.4) proved the regression was real, not harness drift.

**Phase 3 — Protocol migration.** GRPO smoke: 10 steps, zero tool calls, zero
grads — mechanics fine, loop never engaged. Root cause, verified against real
tokenizers locally: our eval spoke a custom dialect TRL never speaks, and the two
model families speak different native syntaxes. Fix: `TOOL_SCHEMAS` + per-model
template rendering everywhere; round-trip verified (render→parse→dispatch→10.0,
both families) before further GPU spend.

**Phase 4 — GRPO 0.8B.** Merged SFT + fresh LoRA, 25 steps: loop engaged (~3.1
calls/episode, <10% failures, live grads, reward variance present), frozen result
2.58 — flat. Transcripts show the failure: staring at the exact answer in its
observation, submitting a near-miss. Copy-fidelity, not retrieval, not format.

**Phase 5 — Capacity test (3B).** Qwen2.5-3B-Instruct zero-shot: 5.18 avg, **50%
full success**. Same harness, same frozen set. The 0.8B wall was scale. SFT-3B:
loss →0.055, **9.76 avg, 96% success** — clears the router bar. GRPO-3B: zero
gradients throughout (every rollout already maxes reward) → 9.74/0.94. RL at a
ceiling correctly does nothing — that flatline is the algorithm working, not failing.

**Phase 6 — Difficulty scaling.** Same env, harder jobs. Base map (n=20):
0.8B dual 3.55 / YoY 0.3; 3B dual 2.9 (5%) / YoY 0.65 (0%). YoY floors everyone —
the loop itself (two opens + arithmetic + combined submit) is the barrier. 3B SFT
on 40-demo sets: dual went 2.9 → 5.45 in one session and 0.0-collapse in another
(**SFT on n=40 is run-unstable** — logged both, cause unknown, possibly seed/init
variance on thin data); YoY 0.65 → ~3.1–3.4 with 0 success both sessions
(retrieval moves, arithmetic never lands). GRPO-dual (merged + 25 steps): train
reward drifted 0.415 → 0.24, frozen 5.45/0.0 with submit restored to 1.0.
Experiment-4 verdict on L2: neither method fixes composition at this data scale.

**Phase 7 — The middle regime (1.5B) + the fair RL test.** Interpolating model
scale on L1 to construct what Phase 6 never yielded: SFT competence with headroom.
Qwen2.5-1.5B-Instruct zero-shot: 1.2 avg, 0 success, 0.40 submit — *below* the 0.8B
base, a confounded cross-tuning comparison (Base vs Instruct verbosity) that
meant nothing until SFT. SFT-1.5B: **9.16 avg, 88% success, submit 1.0** — from 0%
cold into the top edge of the 70–90% band. GRPO-1.5B (merged + 25 steps): **9.16 /
0.88, identical distribution** (44×10.0 + 6×3.0, same split) — all 6 failures are
filing-0 retrieval misses, and 25 steps of RL convert none. The fair test, fairly
run, still says no: at this data scale and budget, environment optimization never
beats what imitation already extracted. Every residual wall (copy-fidelity,
retrieval, arithmetic) is a *capability* gap, and RL cannot install capabilities
the policy never samples.

**Final table (frozen; avg/10, success, submit):**
L1-0.8B: 4.4 → 2.54 → 2.58 (0 succ throughout)
L1-1.5B: 1.2/0.0 → 9.16/0.88 → 9.16/0.88 (identical split)
L1-3B: 5.18/0.50 → 9.76/0.96 → 9.74/0.94
L2-3B: 2.9/0.05 → 5.45/0.0 (unstable: 0.0 twin run) → 5.45/0.0
L3-3B: 0.65/0.0 → ~3.2/0.0 → (deferred)

---

## PART D — THE FIVE FINDINGS (SAY THESE OUT LOUD)

1. **Capacity floor.** Exact-copy grounding emerges between 0.8B (0 successes in
   hundreds of tries, all methods) and 3B (25/50 cold). That interval *is* the
   rent-vs-own answer for this task class.
2. **SFT teaches format, not grounding — and can hurt.** Submit rates rise while
   value accuracy stays zero; at 0.8B greedy emission degrades; at n=40 it's
   run-unstable. Imitation memorizes shapes, not the copy discipline.
3. **RL needs the middle — and the middle still says no.** Tested at floor
   (nothing to reinforce), ceiling (nothing to learn), L2-middle (engaged,
   drifted down), and the constructed fair middle (1.5B at 88% with 12pp
   headroom and 6 failing tasks of variance — identical 44/6 split after 25
   steps). RL's record: 0 for 4. The honest reading is not "RL is useless" but
   "at this data scale and budget, environment optimization never beats what
   imitation already extracted" — every residual wall is a capability gap RL
   cannot sample its way across.
4. **Protocol unity is load-bearing.** Three separate "model failures" were really
   format mismatches between train/eval/RL harnesses. Single contract
   (`TOOL_SCHEMAS` + native templates) or nothing is comparable.
5. **Silent fallbacks lie.** Two phantom evals scored base weights while labeled
   SFT. Identical scores across different checkpoints = weights aren't loading.
   The harness now refuses weightless dirs, and every surprising number gets a
   binary check (`grep` the disk) before any theorizing.

---

## PART E — ECONOMICS (THE ROUTER READING)

The router rule never changed: cheapest model with frozen success ≥ 90% wins.
Applied per leg and level:

- **L1:** 3B-SFT (96%) is the only qualifier → recurring extraction volume routes
  to the owned specialist. Everything else (all 0.8B checkpoints, 1.5B at 88%,
  all bases) keeps renting bigger intelligence. The 1.5B miss is informative:
  near-threshold is still below-threshold; routing is binary.
- **L2/L3:** nothing clears 90% anywhere (best: 3B-SFT-dual 5.45 avg / 0 success).
  Correct decision: no training spend beyond what's done; the levels stay in
  research until demos/reward change the math.
- **Capacity interval as price signal:** grounding emerges between 0.8B (0/50
  everywhere) and 3B (25/50 cold), with 1.5B needing SFT to reach 88%. For this
  task class, specialist training is economically coherent at ≥1.5–3B and
  incoherent below — that sentence is the deliverable the router exists to produce.
- Costs were tracked as tokens/wall-clock on shared T4s, not dollar accounting
  (stated limitation). Relative claim only: a hosted 3B-LoRA inference footprint
  is an order of magnitude below frontier-per-token pricing for identical task
  success (96% vs ~95% frontier-class).

---

## PART F — Q&A BANK (30-SECOND ANSWERS)

1. *"GRPO added nothing four times — why isn't this just 'RL is useless'?"*
   Because each miss has a distinct, measured mechanism: floor = no reward signal
   to reinforce (frac_zero_std 1.0 in smoke); ceiling = saturated policy, zero
   advantage by construction; L2-middle = engaged loop that drifted down
   (0.415→0.24); fair middle = identical 44/6 split, residual wall purely
   retrieval. Four different reasons, all logged — that's evidence about *when*
   RL pays (needs headroom + variance + samplable fixes), not a verdict on RL.
2. *"Isn't this all just scale?"* Partly — and that's the finding, not a
   dismissal. The capacity interval (0.8B→3B) is measured, with 1.5B interpolated:
   base degrades non-monotonically across tuning types (1.5B-Instruct 1.2 < 0.8B-Base
   4.4), while post-SFT ordering is clean (2.54 → 9.16 → 9.76). Tuning confounds
   zero-shot; SFT reveals capacity.
3. *"96% on synthetic data — so what?"* Agreed it's bounded: same generator,
   held-out companies only. It proves the loop can specialize, not that it
   transfers — which is why Experiment 2 (real-SEC, test-only, `data/real_sec/`
   scaffolded with validator + curation rules) is the highest-value next step.
4. *"Why did native SFT hurt the 0.8B?"* Rigid mimicry: 400 perfect transcripts
   taught the emission shape so narrowly that greedy decoding broke out-of-shape
   (submit 0.85→0.58, think-block repetition). Same disease, terminal, in the
   dual 0.0-collapse twin run at n=40.
5. *"What would you do with 10× compute?"* Denser dual demos (40→150) to test
   whether instability is data scale; longer GRPO with KL anchor on L2 (the
   beta=0 drift is the prime suspect in 0.415→0.24); temperature-matched evals;
   then real-SEC curation — in that order, each gated on the previous result.
6. *"Where's the Hyde mapping, concretely?"* Refinery→`generate_tasks.py`
   (traces→tasks with labels); baseline intelligence→frozen company-split evals;
   router→`routing/router.py` threshold rule; Campfire→LoRA SFT + TRL env-GRPO;
   continual→`trajectories/` logging (loop unclosed = V2).

---

## PART G — OPEN THREADS (WHAT'S NEXT, IN ORDER)

1. **Real-SEC transfer (highest value).** Scaffold ready (`data/real_sec/`:
   schema README, `validate.py` passing on all 619 synthetic tasks, `HYDE_FILINGS`
   override proven). Needs 10–20 hand-curated filings + 50–100 tasks, zero
   training. Run all five L1 checkpoints against it.
2. **Denser L2.** Regenerate `dual_train` at n=150, single SFT-3B run: tests
   whether the 0.0-vs-5.45 instability is thin-data variance. Gated, cheap (~15 min).
3. **KL-anchored GRPO on L2.** If (2) stabilizes mid-range partials, re-run GRPO
   with non-zero beta against the drift hypothesis.
4. **1.5B YoY probe (optional).** Only if quota is spare; L3's wall looks
   arithmetic, not retrievable by scale alone.
5. **V2 mining/router.** Explicitly deferred until the specialist-learning
   question above is answered — building routing around unresolved training
   economics repeats the premature-optimization error this project was designed
   to avoid.

## File map (where everything lives)

- Loop: `data/generate_tasks.py`, `environment/{tools,env,reward}.py`
- Measure: `evals/{benchmark,run_model_baseline}.py`, `trajectories/collector.py`
- Train: `training/{sft,grpo,merge_lora}.py` (Kaggle/GPU only)
- Decide: `routing/router.py` (threshold stub)
- Prove: `report/{results.md (26-row log), methodology.md, limitations.md,
  full_report.md, codebase.md}` + this file
- Reproduce: `kaggle/kaggle_V1.ipynb` (14 cells); laptop needs stdlib only
- Pending data: `data/real_sec/` (validator + empty store + curation rules)