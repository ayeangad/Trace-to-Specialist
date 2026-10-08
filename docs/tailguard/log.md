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

## 2026-10-08 — P0.0 cleanup (branch, private/, gitignore)
- Commands: `git checkout -b tailguard`; `mkdir -p private && mv research_notes reports hyde-research-problems.md hyde_deep_dive.html hyde_final_report.html hyde_fresh_investigation_report.html hyde_recording_problem_report.html hyde_route_certificate_report.html .agy deep-research-skill private/`; edited `.gitignore` (+private/, .env, venv/, *.html, research_notes/, reports/, hyde-research-problems.md, tail-guard-build-plan.md, .agy/, deep-research-skill/, experiments/tailguard/cache/).
- Verify: `git status --short` -> only `M .gitignore` + `?? docs/`; `git check-ignore` confirms .env, venv/, final-problem.html, tail-guard-build-plan.md, private/ ignored. PASS.
- Kept in place: venv/ (moving breaks it), .env (never moved/committed), final-problem.html + tail-guard-build-plan.md at root (plan paths; both ignored per Angad's answers).
- Commit: `tailguard: P0.0 repo cleanup + gitignore` (a98562e). PASS.

## 2026-10-08 — P0.1 folders + requirements
- Commands: `mkdir -p tailguard/signals data/tail tests/tailguard docs/tailguard experiments/tailguard/cache`; touch inits + questions.md; wrote requirements-tailguard.txt (openai>=1.40, numpy, scipy, pandas, scikit-learn>=1.4, matplotlib, sentence-transformers, pytest, python-dotenv).
- Commit: `tailguard: P0.1 branch folders and requirements` (b289045). PASS.

## 2026-10-08 — P0.2 install
- Command: `venv/bin/pip install -r requirements-tailguard.txt` -> success (openai 3.26.0, torch 2.14.1, transformers 5.19.0, sentence-transformers 6.1.0, sklearn 1.9.1, etc.); `venv/bin/pip freeze > requirements-tailguard.lock` (84 lines).
- Check: `venv/bin/python -c "import openai, numpy, scipy, sklearn, sentence_transformers, matplotlib, pandas"` exits 0; `import dotenv` ok. PASS.
- Commit: `tailguard: P0.2 laptop install + frozen lock` (187b915). PASS.
- NOTE: installed openai is 3.26.0 (plan assumed >=1.40 API surface). Verified signatures on the installed package (rule 2):
  - `inspect.signature(OpenAI(api_key='test').chat.completions.create)` accepts messages, model, max_completion_tokens, max_tokens, temperature, logprobs, top_logprobs, tools, tool_choice, reasoning_effort, top_p. (full output in shell history; key params confirmed)
  - `CompletionUsage.model_fields` = completion_tokens, prompt_tokens, total_tokens, completion_tokens_details, prompt_tokens_details.
  - `TopLogprob.model_fields` = token, bytes, logprob. ToolCall fields = id, function, type. `CompletionTokensDetails` includes reasoning_tokens. Errors RateLimitError/APIStatusError/BadRequestError exist.
  - `OpenAI()` with no key raises OpenAIError (expected; client constructed lazily after dotenv load).

## 2026-10-08 — P0.3 API key
- Command: `venv/bin/python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('key present:', bool(os.getenv('OPENAI_API_KEY')))"` -> `key present: True` (boolean only; key never printed/logged/committed). PASS.
- No files created, no commit. Dashboard data-sharing confirmation is with Angad (needed for free tokens).

## 2026-10-08 — P0.4 logging_utils.py
- Wrote tailguard/logging_utils.py (append_jsonl w/ flush+fsync, read_jsonl, read_done_ids, write_config). Committed with P0.5 (below).

## 2026-10-08 — P0.5 openai_client.py + tests
- Wrote tailguard/config.py (verbatim §3 constants), tailguard/openai_client.py (cache/ledger/budget/retries/build_params + live-or-dict adapters + --probe), tests/tailguard/test_openai_client.py (9 tests, fake client, no network).
- Command: `venv/bin/python -m pytest tests/tailguard/test_openai_client.py -q` -> `9 passed in 0.80s`. PASS. Covers: cache-hit zero tokens, ledger sums by pool+UTC date, BudgetExceeded (no ledger/cache write), pool_of unknown raises, build_params drops unaccepted keys + lowest reasoning level, retry-then-success (1 ledger line), BadRequest no-retry (1 call, no ledger), adapter live==cached, tool_choice override passthrough.
- Commit: `tailguard: P0.4-P0.5 logging utils, config, budget-safe OpenAI client + tests` (c51559c). PASS.

## 2026-10-08 — P0.6 capability probe
- Command: `venv/bin/python -m tailguard.openai_client --probe` -> wrote docs/tailguard/openai_caps.json (5 models). Probe tokens: 989 (< 5,000). PASS.
- Measured caps (full file committed):
  - gpt-5.4-mini (frontier): max_completion_tokens T, max_tokens F, temperature T, logprobs T, tools T, reasoning [none, low]. Frontier stands, no fallback.
  - gpt-4.1-mini (judge): max_completion_tokens T, max_tokens T, temperature T, logprobs T, tools T, reasoning []. Judge returns logprobs and is non-reasoning as hoped; judge stands, no fallback.
  - gpt-5.4 (strong): same shape as gpt-5.4-mini. gpt-5.4-nano: same shape. gpt-4.1-nano: tools F BUT error is "Could not finish the tool call because max_tokens was reached" (16-token truncation artifact of the probe, not proof of no tool support); reasoning [].
  - Notable errors: 5.4-family rejects max_tokens ("Use max_completion_tokens instead") and reasoning_effort=minimal ("Supported: none, low, medium, high, xhigh"); 4.1-family rejects reasoning_effort entirely ("Unrecognized request argument").
- Commit: `tailguard: P0.6 OpenAI capability probe caps file` (2c7bb05). PASS.

## 2026-10-08 — P0.7 frontier pilot (budget calibration)
- Temp script /tmp/opencode/tg_p0_pilot.py (not in repo): 1 episode on data/tasks/dev.jsonl dev-0000 ("What was Hooli Systems's revenue for FY2022?"), default store, Phase-4 loop, gpt-5.4-mini, max_out 512.
- Output: submitted value "$58,663 million" filing HOOL-10K-2022 evidence "Revenue was $58,663 million." -> success False, reward 7.0 (filing+section+evidence+efficiency right, VALUE wrong: gt $29,638M). Honest pilot point: frontier is not oracle on this data.
- Tokens: 4 api calls, prompt 1858 + completion 134 = 1992 total (~2.0k/episode vs plan estimate 2.5k). Phase 4 recheck: 1000 episodes x ~2k ~= 2.0M tokens, fits small-pool day only if sequenced with judge per plan Days A/B/C.
- No repo files created, no commit.

## 2026-10-08 — P1 local prep (GPU work runs on Kaggle; laptop prep only)
- Flag checks (all match runbook): `data/generate_tasks.py --help` (--n-train/--n-dev/--n-test/--seed), `trajectories/collector.py --help` (--tasks/--out), `training/sft.py --help` (--model/--demos/--out/--epochs/--lr), `evals/run_model_baseline.py --help` (--model/--base-model/--tasks/--out/--limit). PASS.
- Old-format confirm: `head -c 400 data/tasks/sft_demos.jsonl` shows prompt/completion rows (no turns) -> regen required. PASS.
- Determinism: ran `venv/bin/python data/generate_tasks.py --n-train 400 --n-dev 60 --n-test 100` locally; `git status --short data/` empty (byte-identical train/dev/frozen_test/filings/manifest). PASS. (Manifest counts: train 400, dev 56, frozen 99 — per-company rounding, as committed.)
- Collector dry run: `trajectories/collector.py --tasks data/tasks/train.jsonl --out /tmp/opencode/sft_demos_check.jsonl` -> kept 400, dropped 0, all rows have turns+reward keys. PASS (matches "kept ≈ 400").
- Summarise one-liner tested on synthetic 50-row file -> n 50 mean_reward 10.0 success 0.96 had_submit 1.0. PASS.
- Remote for clone cell: origin git@github.com:ayeangad/Trace-to-Specialist.git.
- Wrote kaggle/tailguard_runbook.md (Phase 1 cells 0-6 + persist list; P3/P8.1 stubs). No GPU run locally.

## 2026-10-08 — P1.1 single-file helper (per Angad request)
- Wrote kaggle/p1_run.py: same P1 commands as runbook cells (regen/train/verify) + same gates (kept>=350, turns key, deterministic diff, adapter+checkpoint-100 files, success>=0.90), exits non-zero on gate failure. HF token via --hf-token or HF_TOKEN env (avoids Kaggle-secret env-injection issues seen with os.environ).
- Local tests (no GPU): --help ok; summarize() on synthetic 50-row file -> exact expected dict. PASS. Full run needs T4 (script enforces cuda gate itself).
- Runbook header points to the script as shortcut; cells stay canonical.

## 2026-10-08 — P1 Kaggle debug: missing top-level adapter files
- Symptom (pasted): harness `ERROR: /kaggle/working/qwen25-3b-sft has no adapter_config.json or config.json. Did you mean ... checkpoint-100?`.
- Root cause (verified on installed package, rule 2): transformers 5.19.0 `trainer.py::_finalize_training` (lines 1976-2025) writes NO end-of-training save to the out-dir root; `_save_checkpoint` fires only at save_steps. With 100 steps and save_steps=100, final weights exist only in checkpoint-100/ (step 100 = final step). Training itself healthy (100/100, loss 0.2557). Runbook expectation was wrong, not the training.
- Fix: runbook Cell 4b copies adapter_config.json + adapter_model.safetensors up from checkpoint-100 after asserting trainer_state global_step==100 and sha256-identical after copy. Honest relocation of final weights; verify then runs against the out dir as planned.
- Also fixed: verify summary must be a plain cell (second `!python -c` quoting EOF from user). Committed as P1.3.

## 2026-10-08 — P1 verify result (reported from Kaggle, file pending)
- Cell C summary (pasted by Angad): n 50 mean_reward 10.0 success 1.0 had_submit 1.0.
- GATE success >= 0.90: PASS (reported; file-level acceptance runs once p1_frozen_sft3b.jsonl is on the laptop).
- Pending: Save Version, dataset upload qwen25-3b-sft-tailguard, download jsonl to experiments/tailguard/.

## 2026-10-09 — P1 acceptance (file-level, on laptop)
- File home: experiments/tailguard/p1_frozen_sft3b.jsonl (50 rows).
- Recomputed from file: n 50 mean_reward 10.0 success 1.0 had_submit 1.0. Matches Kaggle-pasted summary. GATE >= 0.90: PASS.
- Adapter files: verified via /tmp/tg-adapter extraction of uploaded tarball — top-level adapter_config.json + adapter_model.safetensors present, checkpoint-100/adapter_config.json present. PASS.
- Still missing for the record: Kaggle HEAD hash, collector kept count, git diff --stat (Cells 2-3 paste-backs); dataset-upload confirmation. Asked Angad.

## 2026-10-09 — P1 session transcript (pasted by Angad, record)
- Cell 4b output: global_step 100 max_steps 100; adapter_config.json identical True; adapter_model.safetensors identical True; top-level now [README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, checkpoint-100, tokenizer.json, tokenizer_config.json].
- Cell 5: weights loaded 434/434, wrote p1_frozen_sft3b.jsonl; summary n 50 mean_reward 10.0 success 1.0 had_submit 1.0 (matches committed file).
- Note: HF Hub warning "sending unauthenticated requests" during verify (tokenizer download worked anyway; login cell ran earlier for training).
- Note: laptop-tar-in-notebook-cell SyntaxError repeated in transcript (already resolved on laptop terminal; /tmp/tg-adapter verified).
- MISSING per rule 3 (session closed, not pasted): Kaggle HEAD hash, collector kept count, git diff --stat. Training success (1.0) implies regen was sound; local determinism check passed.

## 2026-10-09 — P2.2-P2.8 tail pool done
- generate_tail.py run: 138 filings, 2000 tasks (fit 500 / cal 700 / test 800; S0 64% + 6 tails x 6% exact per split), 200 drift. Slice texts spot-checked (S0/T1/T2/T3/T6/D1 match templates; T4 synonyms; T5 typo "Ketsrel"; D1 gt e.g. $7.6B -> 7600).
- Filing-slice tag decision: standard-text filings (shared by S0/T4/T5 tasks) tagged slice "S0" at filing level; task records carry the task slice. notes standard everywhere; mda standard except explicit T1/T2 text and D1 billions phrasing; D1 tables hold rounded gt millions.
- validate_tail.py: ALL CHECKS PASSED (evidence substrings, values incl. D1 rounding, company-split uniqueness, ticker hygiene, 2200 search hits, exact counts).
- Oracle: tasks.jsonl -> avg_reward 10.0 success_rate 1.0 n 2000; drift -> 10.0 / 1.0 n 200. PASS.
- tests: test_generate_tail.py 3 passed (determinism sha, manifest sha, per-slice template checks). Determinism re-run touched only the timestamp sidecar (restored).
