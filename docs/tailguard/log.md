# Tail Guard build log

## 2026-10-08 — BEFORE ANY CODE (pre-phase summary, in my own words)

### Goal
Build a per-request safety check that runs after the small specialist
(Qwen2.5-3B-Instruct + LoRA) answers a filing-extraction request. The check
collects cheap signals (confidence, self-consistency, rarity, verifier, optional
OpenAI judge), combines them with logistic regression into a risk score, and
picks a keep/escalate threshold with Learn-then-Test so that the error rate on
kept requests is ≤ ε with probability ≥ 1−δ. Low risk → keep the cheap answer;
high risk → escalate to an OpenAI frontier model. Add-ons: a rare-cluster drift
alarm, an adapter fingerprint, and a judge canary. The headline to earn (Phase 6)
is: at ε = 2 %, δ = 0.1, Tail Guard keeps clearly more traffic on the small
model than confidence alone, the guarantee holds on held-out companies, and a
per-slice table shows which tails get caught.

### The 12 operating rules (spec §0), paraphrased
1. Read the whole plan first; work one phase at a time in order; never start the
   next phase until the current one's checks all pass and are logged.
2. Never invent APIs — only use functions/params/flags from the plan, the repo,
   or verified on the installed package with inspect/help, pasted into this log.
3. Never invent results — every number comes from a file under
   experiments/tailguard/; write MISSING when absent.
4. Never weaken a threshold, test, or gate to make it pass; on gate failure, log
   it and follow that phase's "If it fails" section (stop if it says stop).
5. Never edit protected files: environment/*.py, data/generate_tasks.py,
   data/tasks/*, data/documents/filings.json, data/real_sec/*,
   evals/run_model_baseline.py, training/*.py, trajectories/collector.py,
   routing/router.py (only exception: Phase 1 regenerates sft_demos.jsonl via
   the existing collector runbook step).
6. New code only in: tailguard/, data/tail/, tests/tailguard/, docs/tailguard/,
   kaggle/tailguard_runbook.md, requirements-tailguard.txt,
   experiments/tailguard/ outputs. Nothing else.
7. Labels are never features — label/ground_truth/success/reward fields are read
   only by labelling/calibration/evaluation code (enforced by a Phase 5 test).
8. Only synthetic or public data goes to OpenAI (free tokens share traffic).
9. Log everything here (date, phase, step, command, outcome, errors); on doubt,
   write to docs/tailguard/questions.md and stop that step.
10. Git: branch `tailguard`; commit after each passing numbered step as
    `tailguard: P<phase>.<step> <what>`; only add rule-6 files; never commit
    experiments/tailguard/cache/, .env, weights, reports/, research_notes/,
    website/.
11. Reproducibility: every script takes --seed (default 20261008) and writes its
    config (args + git hash) as first line or sidecar *.config.json.
12. Resumability: long scripts append one JSON line per item with flush, and skip
    ids already present on restart.

### Constants (spec §3, in tailguard/config.py to come)
- SEED 20261008; BASE_MODEL Qwen/Qwen2.5-3B-Instruct; Kaggle adapter dirs
  /kaggle/working/qwen25-3b-sft (+ /checkpoint-100 stale).
- TAIL_DIR data/tail (filings.json, tasks.jsonl, drift_tasks.jsonl, manifest.json);
  EXP experiments/tailguard.
- SLICES S0,T1–T6 (D1 drift-only); SPLITS fit/cal/test; N_TASKS 500/700/800;
  SLICE_SHARE S0 0.64, each tail 0.06; N_DRIFT 200.
- MAX_TURNS 6; MAX_NEW_TOKENS 256; PROMPT_CHAR_WINDOW 12000; K_SAMPLES 5;
  SAMPLE_TEMPERATURE 0.7; SAMPLE_TOP_P 0.95.
- EPSILONS [0.01, 0.02, 0.05]; DELTA 0.10; PRIMARY_EPS 0.02; LTT_MIN_COVERAGE 0.10;
  LTT_GRID_STEP 0.01; N_RESPLITS 200.
- EMBED_MODEL sentence-transformers/all-MiniLM-L6-v2; KNN_K 10.
- OpenAI: frontier gpt-5.4-mini (small pool), judge gpt-4.1-mini,
  strong gpt-5.4 (large pool, 100-item check only); POOL_SMALL/POOL_LARGE sets;
  DAILY_CAP small 2.3M / large 230k tokens (8% margin under free limits).
  API params come ONLY from docs/tailguard/openai_caps.json (Phase 0 probe).

### Phase about to do (P0 — laptop, setup/guardrails/caps probe)
Branch + folders, laptop install from requirements-tailguard.txt (+ freeze lock),
.env key + data-sharing confirmation, logging_utils.py, caps-aware budget-safe
openai_client.py (cache/ledger/budget/retries/build_params), the capability
probe writing openai_caps.json, and a one-episode frontier pilot for token
calibration — then the P0 acceptance checks (imports, caps file, client tests,
ledger < 10k tokens). Waiting for "go".
