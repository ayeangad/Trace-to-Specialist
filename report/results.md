# Results

## Experiment series (locked vs planned)

- **Experiment 1 — Synthetic extraction (LOCKED, below).** Capacity/grounding
  study on procedural filings. Protected: rows below are never overwritten;
  later experiments get their own sections.
- **Experiment 2 — Real-SEC transfer (scaffolded, `data/real_sec/`).**
  50–100 hand-curated tasks on 10–20 real 10-K/10-Q filings, never trained on.
  Tests whether synthetic-learned behavior transfers off-generator.
- **Experiment 3 — Difficulty scaling (scaffolded + oracle-verified, no GPU yet).**
  Same env, harder jobs: L2 dual-metric (`dual_train/test`: 40/20 tasks) and L3
  YoY retrieve-retrieve-calculate-verify (`yoy_train/test`: 40/20 tasks), all
  company-split, all oracle-scoring 10.0 with adversarial negatives verified
  (tolerance ±0.05, flipped verdict → 8.0, missing filing → 8.0/6.0).
  GPU runs pending: map 0.6B/0.8B/1.5B/3B × L1/L2/L3, find SFT ≈ 70–90% cells,
  then GRPO there.
- **Experiment 4 — GRPO under non-saturation (pending Exp 3).** SFT vs SFT+GRPO
  only where headroom exists. Exp-1 GRPO runs (floor + ceiling) stay as
  boundary evidence, not as verdicts on RL.

## Experiment log (frozen_test unless noted)

| date | experiment | model | n | avg_reward/10 | success | submit_rate | notes |
|---|---|---|---|---|---|---|---|
| 2026-10-01 | baseline_qwen06_n10 | Qwen3-0.6B (Base, naive harness) | 10 | 0.0 | 0.0 | 0.0 | Harness artifact: no chat template, no 1-shot; model never emitted TOOL:/FINAL:. Kept as "why SFT" data point. |
| 2026-10-01 | baseline_qwen06_v2_n10 | Qwen3-0.6B (Base, patched harness: chat template + 1-shot) | 10 | 2.4 | 0.0 | 0.6 | Submits 60% now; partial credit (filing/section/efficiency) but value+evidence still failing. Oracle ceiling (local): 10.0 / 1.0. |
| 2026-10-01 | baseline_qwen25_inst_n10 | Qwen2.5-0.5B-Instruct (patched harness) | 10 | 1.9 | 0.0 | 0.5 | Below base-0.6B on n=10 (noise + verbose non-submits); confirms task is hard for small models zero-shot. |
| 2026-10-01 | baseline_qwen35_n10 | Qwen3.5-0.8B-Base (patched harness; ref-fallback kernels, no causal_conv1d/fla) | 10 | 3.6 | 0.0 | ? | Best zero-shot; still 0 success. This is the ablation base SFT/GRPO are measured against. |
| 2026-10-01 | baseline_qwen06_n50 | Qwen3-0.6B (patched harness) | 50 | 3.42 | 0.0 | 0.86 | Frozen matrix row. High submit rate, zero full successes. |
| 2026-10-01 | baseline_qwen35_n50 | Qwen3.5-0.8B-Base (patched harness) | 50 | 3.66 | 0.0 | 0.94 | Frozen matrix row. Consistent with n=10; leads on partial credit. |
| 2026-10-01 | sft_n50 (answers-only) | Qwen3.5-0.8B + LoRA SFT on oracle final answers (loss 2.93→0.38, tok-acc 0.89) | 50 | 3.66 | 0.0 | 0.94 | **No gain vs base.** Diagnosis: SFT taught answer style; eval tests the TOOL loop, which demos never contained. Fix: SFT on full ReAct transcripts. |
| 2026-10-01 | sft_traj_n50 (transcripts) | Qwen3.5-0.8B + LoRA SFT on oracle ReAct transcripts, 2 epochs (loss 1.52→0.11, tok-acc 0.96) | 50 | 3.66 | 0.0 | 0.94 | **Still no gain; identical to base to 2 decimals.** Suspect generations bit-identical: adapter not attached in eval OR attached but inert under greedy decoding. Diagnosing. |
| 2026-10-01 | phantom-eval lesson | — | — | — | — | — | sft_n50 + sft_traj_n50 both evaluated a dir with NO adapter files (only checkpoint-100/); harness silently fell back to base weights. Fix: harness now exits loudly on model dirs without adapter/config; always verify adapter_config.json exists at --model. |
| 2026-10-01 | sft_traj_ckpt100_n50 | Qwen3.5-0.8B + LoRA SFT transcripts, real adapter (checkpoint-100) | 50 | 3.60 | 0.0 | 1.0 | Adapter steers (submit 0.94→1.0) but no success gain. Bottleneck is extraction correctness, not format. Reading transcripts next. |
| 2026-10-01 | sft_native_n50 | Qwen3.5-0.8B + fresh LoRA on NATIVE tool-call transcripts, 2 epochs | 50 | 2.68 | 0.0 | 0.72 | REGRESSION vs text-SFT (3.60/1.0): 14 no-submits, 23×3.0, 13×5.0. Missing control: base was never scored on the NATIVE harness — level shift may be harness strictness (break-on-first-miss), not model. Get base-native before concluding. |
| 2026-10-02 | base_native_n20 | Qwen3.5-0.8B-Base, NATIVE harness (first true same-harness control) | 20 | 4.40 | 0.0 | 0.85 | Scores ABOVE old-harness base (3.66): harness is not the level shift. Native SFT genuinely hurt. |
| 2026-10-02 | sft_native_v2_n50 | Qwen3.5-0.8B + native-transcript LoRA (retrained clean session, loss →0.05), NATIVE harness | 50 | 2.54 | 0.0 | 0.58 | Confirmed regression vs base (4.4→2.54, submit 0.85→0.58). SFT drives rigid mimicry that breaks tool emission; grounding/copy-fidelity untouched. GRPO-real is now the decisive test. |
| 2026-10-02 | grpo_n50 | Merged SFT + fresh LoRA, GRPO 25 steps G=4 (train: call_freq ~3.1, fail <0.1, reward 0.33–0.38, grads live) | 50 | 2.58 | 0.0 | 0.58 | Flat vs SFT. Breakdown: 21×5.0 (right filing, wrong value), 21× no-submit, 8×3.0. Value NEVER correct in any checkpoint. Train-time sampling explores (call 3+/ep); greedy eval collapses (repetitive think blocks). |
| 2026-10-02 | base_3b_n50 | Qwen2.5-3B-Instruct, NATIVE harness, zero-shot | 50 | 5.18 | 0.50 | 0.68 | **Capacity hypothesis CONFIRMED.** 25/50 full successes with no training. The 0.8B failure was scale, not method — exact-copy grounding emerges by 3B. Specialist thesis alive: SFT/GRPO-3B now chase 0.50 → threshold. |
| 2026-10-02 | sft_3b_n50 | Qwen2.5-3B-Instruct + LoRA on native transcripts, 2 epochs (loss →0.055) | 50 | 9.76 | 0.96 | 1.00 | **Specialist crosses the bar.** 48/50 full successes, clears the 90% router threshold. Chain complete: Base 5.18/0.50 → SFT 9.76/0.96. |
| 2026-10-02 | grpo_3b_n50 | Merged SFT-3B + fresh LoRA, GRPO 25 steps (loss/grad ≈ 0 throughout) | 50 | 9.74 | 0.94 | 1.00 | Flat vs SFT (48/50 vs 47/50, noise). RL-at-ceiling confirmed: policy already saturates reward, no advantage signal left. Final chains — 0.8B: 4.4→2.54→2.58 (all 0 success). 3B: 5.18/0.50→9.76/0.96→9.74/0.94. |
| 2026-10-03 | base08_dual_n20 | Qwen3.5-0.8B-Base, L2 dual-metric | 20 | 3.55 | 0.0 | — | Partial credit only, same grounding wall. |
| 2026-10-03 | base08_yoy_n20 | Qwen3.5-0.8B-Base, L3 YoY | 20 | 0.30 | 0.0 | — | Near-zero: loop rarely completes (2 opens + combined submit). |
| 2026-10-03 | base3b_dual_n20 | Qwen2.5-3B-Instruct, L2 dual-metric | 20 | 2.90 | 0.05 | — | Below its L1 (5.18): combined protocol is genuinely harder zero-shot. |
| 2026-10-03 | base3b_yoy_n20 | Qwen2.5-3B-Instruct, L3 YoY | 20 | 0.65 | 0.0 | — | Both models floor on L3 at base. SFT is the next probe. |
| 2026-10-03 | sft3b_dual_n20 | 3B + LoRA on 40 dual transcripts, 3 epochs (loss →1.28, thin data) | 20 | 0.00 | 0.0 | — | Collapse: below base (2.9). Suspect zero submits (greedy emission failure). Transcript autopsy pending. |
| 2026-10-03 | sft3b_yoy_n20 | 3B + LoRA on 40 yoy transcripts, 3 epochs | 20 | 3.40 | 0.0 | — | Partial-credit lift vs base (0.65) but 0 success. Retrieval moves, arithmetic/grounding does not. GRPO candidate only for retrieval signal. |
| 2026-10-03 | sft3b_dual_n20 (rerun) | 3B + LoRA on 40 dual transcripts, 3 epochs, fresh session | 20 | 5.45 | 0.0 | 0.95 | Different from last session's collapse (0.0): 12×5.0 + 7×7.0 + 1 no-submit. SFT on n=40 is run-unstable. Pattern: value_a lands, value_b never; evidence always exactly 1 span — model emits single-value submits, "|" protocol not robustly learned. GRPO candidate (variance + headroom). |
| 2026-10-03 | sft3b_yoy_n20 (rerun) | 3B + LoRA on 40 yoy transcripts, 3 epochs, fresh session | 20 | 3.10 | 0.0 | 0.75 | Consistent with last session (3.4). Mixed partials, verdict occasionally right, yoy number never within tolerance. |
| 2026-10-03 | grpo_dual_n20 | Merged dual-SFT + fresh LoRA, GRPO 25 steps (train 0.415→0.24, drift) | 20 | 5.45 | 0.0 | 1.00 | Flat vs SFT average (13×5.0, 6×7.0, 1×2.0); submit rate restored to 1.0, value_b still never lands. Exp-4 verdict: neither SFT nor GRPO fixes L2 composition at this data scale. |
| 2026-10-04 | base15b_n50 | Qwen2.5-1.5B-Instruct, NATIVE harness, zero-shot | 50 | 1.20 | 0.0 | 0.40 | BELOW 0.8B-base (4.4/0.85): instruct verbosity suspected (0.40 submit). Cross-family base comparison is confounded (Base vs Instruct tuning); SFT number decides. |
| 2026-10-04 | sft15b_n50 | Qwen2.5-1.5B-Instruct + LoRA native transcripts, 2 epochs | 50 | 9.16 | 0.88 | 1.00 | **Middle regime found.** 44/50: SFT climbs from 0% cold into the top edge of the 70–90% band. GRPO-1.5B is the fair expansion test (12pp headroom, 6 failing tasks of variance). |
| 2026-10-04 | grpo15b_n50 | Merged SFT-1.5B + fresh LoRA, GRPO 25 steps | 50 | 9.16 | 0.88 | 1.00 | **Identical to SFT** (44×10.0 + 6×3.0, same split). All 6 failures are filing-0 retrieval misses; 25 steps of RL convert none. Exp-4 verdict: even in the fair middle regime, GRPO adds nothing at this budget — residual wall is retrieval, not refinement. |

## V1 answer (0.8B leg — negative result, complete chain)

Base 4.4 → SFT 2.54 → GRPO 2.58 (frozen n50/n20, reward/10, success = exact value + filing).
Specialization at this scale does not close the grounding gap; SFT hurts greedy
tool emission, GRPO engages the loop (call_freq 3+, failures <10%) but 25 steps
cannot teach verbatim copy-fidelity. No checkpoint clears a 90% router threshold —
economics: nothing to route yet. This is a negative result with a complete causal
chain, not an inconclusive one. See limitations.md for the three leading hypotheses
(reward sparsity on value/evidence, greedy/eval vs sampling/train mismatch, 0.8B
copy-fidelity capacity).

## V1 answer (3B leg — thesis confirmed)

Base 5.18/0.50 → SFT 9.76/0.96 (frozen n50). A 3B LoRA specialist clears the 90%
router threshold (48/50) on unseen companies. Economics: the router now has
somewhere to route — recurring extraction volume moves off frontier pricing onto
a cheap, owned specialist. Capacity floor measured: grounding emerges between
0.8B (0/50, all checkpoints) and 3B (25/50 zero-shot). GRPO-3B optional: at 96%
the headroom is ~4pp and reward variance is thin (expect high frac_reward_zero_std),
so it tests RL-at-ceiling, not the core claim — which SFT already settled.

## Failure analysis (sft_traj_ckpt100_n50)

Breakdown: 35× reward 3.0 (wrong filing, right section) + 15× reward 5.0 (right filing, wrong value). Value never correct, evidence never correct. Paired vs base: 9/50 tasks changed — SFT steers without gain.
Transcript te-0000 (Cyberdyne 2022 revenue): model searches, opens the CORRECT section, OBS contains the exact answer — then submits a near-miss filing id/value instead of copying verbatim. Diagnosis: **grounding/copy-fidelity failure**, not retrieval or format. SFT (imitation) can't fix what it can't see fail; this is the case for GRPO (direct optimization of value+evidence reward).

## Protocol migration v3 (2026-10-01): native tool calls everywhere

GRPO smoke mechanics pass, but base models never emit TRL's tool format zero-shot
(tools/call_frequency ≈ 0, all rewards 0, completions clipped). Root causes found
via local tokenizer inspection:
  - Qwen3 uses `<tool_call>{"name","arguments"}</tool_call>` (JSON);
    Qwen3.5 uses `<tool_call><function=f><parameter=p>v</parameter></function></tool_call>` (XML-ish).
  - All prior SFT/eval used a custom TOOL:/FINAL: text protocol TRL never speaks.
Fix: single native protocol — TOOL_SCHEMAS (environment/tools.py) rendered via each
model's own apply_chat_template for SFT text, the eval harness loop, and TRL's env
tools. Round-trip verified locally with real tokenizers (render→parse→dispatch→10.0
for both families). Old-protocol scores stay logged but are not comparable.

## Summary table (n=50 frozen runs; fill as completed)

| checkpoint | success | avg reward | avg calls | evidence ok | tokens | latency | cost |
|---|---|---|---|---|---|---|---|
| Base 0.8B | ? | ? | ? | ? | ? | ? | ? |
| SFT | ? | ? | ? | ? | ? | ? | ? |
| SFT+GRPO | ? | ? | ? | ? | ? | ? | ? |

Oracle ceiling (local): 1.0 success, 10.0 reward, 2.0 calls (n=20 frozen_test sample).
