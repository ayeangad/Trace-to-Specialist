# Trace-to-Specialist — Codebase Deep Dive

*Every statement below is grounded in the source files in this repo (verified
2026-10-02; all 11 Python files compile). Nothing here describes aspirations,
only what the code does. Where the code is a stub, demo, or dead, it says so.*

---

## 0. What this project is (one paragraph)

A miniature, end-to-end implementation of the enterprise-intelligence loop:
procedural financial filings stand in for company documents; a task generator
turns them into metric-extraction jobs with known answers; a stateful tool
environment lets models act (search → open → submit) instead of just answering;
a deterministic 10-point reward scores outcome *and* process; frozen test tasks
use unseen companies; and three Kaggle stages — multi-model baselines, LoRA SFT,
environment-based GRPO — measure Base → SFT → GRPO on identical tasks. A router
stub encodes the commercial rule (cheapest model clearing 90%). Two model legs
were actually run: Qwen3.5-0.8B (fails everywhere: 4.4 → 2.54 → 2.58, 0 successes)
and Qwen2.5-3B-Instruct (5.18/0.50 → **9.76/0.96** → 9.74/0.94).

---

## 1. Repository layout (actual files on disk)

```text
hyde/
├── data/
│   ├── generate_tasks.py        # the only data creator; stdlib only
│   ├── documents/filings.json   # generated: 33 filings (company × year store)
│   ├── tasks/{train,dev,frozen_test}.jsonl  # generated: 400 / 56 / 99 tasks
│   └── splits/manifest.json     # generated: seed + company lists + counts
├── environment/
│   ├── tools.py                 # 3 stateless tool functions + TOOL_SCHEMAS
│   ├── env.py                   # FilingEnv: stateful episode wrapper (TRL-compatible)
│   └── reward.py                # normalize_value + score_episode (the 10-pt judge)
├── evals/
│   ├── benchmark.py             # oracle_rollout (perfect policy) + run_benchmark
│   └── run_model_baseline.py    # native-tool-loop model eval (the measuring tape)
├── trajectories/
│   └── collector.py             # oracle_native_turns + collect (SFT demo maker)
├── training/
│   ├── sft.py                   # LoRA SFT (Kaggle/GPU)
│   ├── grpo.py                  # LoRA GRPO with environment_factory (Kaggle/GPU)
│   └── merge_lora.py            # merge adapter → full weights (SFT→GRPO bridge)
├── routing/router.py            # route(): cheapest model ≥ threshold (stub + demo)
├── report/{methodology,results,limitations,full_report}.md
├── kaggle/kaggle_V1.ipynb       # 14-cell ordered GPU runbook
├── requirements.txt             # stdlib locally; lists GPU deps for Kaggle
├── README.md
└── experiments/                 # empty locally; filled on Kaggle (*.jsonl results)
```

Split of compute, enforced by imports: everything except `training/` and the model
half of `run_model_baseline.py` is **stdlib-only** (no torch/transformers imports)
and runs on a laptop. GPU code paths import torch/transformers/peft/trl/datasets
inside functions or scripts marked RUN ON KAGGLE.

---

## 2. `data/generate_tasks.py` — the world builder (167 lines, stdlib)

**What it does:** creates the entire synthetic world — documents *and* tasks *and*
answers — deterministically from `--seed` (default 42).

**How, step by step:**
1. Constants: 8 `TRAIN_COMPANIES` + 3 `TEST_COMPANIES` (disjoint by construction);
   a `TICKER` map; 4 `METRICS` (revenue, net_income, total_assets,
   operating_cash_flow); 3 `SECTIONS`; years 2022–2024; 5 `PROMPT_TEMPLATES`.
2. `make_filing(company, year, rng)`: draws `base = randint(8_000, 60_000)`, derives
   the other three metrics as fixed fractions of it (net_income 5–22%,
   total_assets 1.5–3.5×, cash flow 10–35%), and renders three text sections plus
   three tables (`income_statement`, `balance_sheet`, `cash_flow`). Filing id is
   `{TICKER}-10K-{year}`. Money always formats as `$12,345 million` (`_money`).
3. `make_task(filing, metric, rng, tid)`: picks a random prompt template (this
   phrasing variation *is* the "messy traces" simulation), and records
   `ground_truth = {value (int), value_str, filing_id,
   section ("financial_statements" always), evidence_substr (= value_str)}`.
   Task ids are prefixed `tr-`/`dev-`/`te-`.
4. `build(companies, per_company_tasks, seed, prefix)`: builds every company×year
   filing, then samples `per_company_tasks × len(companies)` random
   (filing, metric) pairs. Note the arithmetic: `--n-train 400` over 8 companies =
   50/company = 400; `--n-dev 60` → `60//8 = 7` → **56, not 60**; `--n-test 100`
   → `100//3 = 33` → **99, not 100**. Exact committed counts: **400 / 56 / 99**.
   Dev uses train companies with `seed+1`; test uses test companies with `seed+2`,
   so all three sets differ.
5. Writes `filings.json` (train + test filings merged: 24 + 9 = **33**),
   the three task files, and `manifest.json` (seed, company lists, counts).

**Why company-split:** `frozen_test` companies (Cyberdyne, Tyrell, Soylent) never
appear in train, so a memorizing model scores zero — the eval measures capability.

**What it does NOT do:** no real SEC/EDGAR fetching (despite the project narrative
mentioning EDGAR as motivation); no trace clustering (V2); no difficulty grading.
`SECTIONS` lists `mda`/`notes` but every task's ground-truth section is
`financial_statements`, and section text for mda even repeats the revenue figure —
retrieval is deliberately easy so the test isolates *extraction fidelity*.

**Dead-but-present:** `dev.jsonl` is generated and counted in the manifest but
**no script in the repo ever reads it** (verified by grep). `build()` has an unused
`i = 0`.

---

## 3. `environment/tools.py` — the three tools + the schema contract (126 lines, stdlib)

**What it does:** implements the world API as three plain functions over a
lazily-loaded, module-cached filing store (`_FILINGS`, loaded once from
`filings.json`; `reload_store()` exists for tests).

1. `search_filing(query)`: lowercases the query; matches if the query (dashes→spaces)
   is a substring of `"company ticker year"`, or the ticker/first company word
   appears in it; fallback = year-substring match, capped at 5, sorted. Returns
   newline-joined filing ids or `"NO_RESULTS"`. Note: matching is substring-based,
   so vague queries return multiple candidates — retrieval is non-trivial by design.
2. `open_section(filing_id, section)`: returns section text, or `"ERROR: ..."`
   strings for unknown filing/section (errors are *data*: the reward counts them).
3. `get_table(filing_id, table_id)`: returns the metric dict as JSON, or `"ERROR"`.

**`TOOL_SCHEMAS`** (lines 94–126): OpenAI-style `{"type":"function",...}` specs for
all four env methods (the three above + `submit_answer`, which lives on the env,
not here). Same names, same argument names. This list is the **single protocol
contract**: it is passed as `tools=` to `tokenizer.apply_chat_template` in SFT
rendering (`training/sft.py:62-63`) and the eval harness (`run_model_baseline.py`),
so each model family renders its *native* tool-call syntax (Qwen3 JSON vs Qwen3.5
XML-ish) from one source. Docstrings are Google-style with `Args:`/`Returns:`
because TRL exposes raw functions as tools from exactly that metadata.

**What it does NOT do:** no network, no state, no auth, no pagination. `submit_answer`
is specified here but implemented in `env.py` (it needs episode state).

---

## 4. `environment/env.py` — `FilingEnv`, the episode wrapper (137 lines, stdlib)

**What it does:** turns stateless tools into a stateful per-episode environment
matching TRL's `environment_factory` contract (stated in the module docstring).

**State per instance:** `task`, `tool_calls` (list of `{tool, result}`), 
...[truncated 13588 chars]