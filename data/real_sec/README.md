# Experiment 2 — Real-SEC transfer set

**Status: scaffolded, empty.** This directory will hold 10–20 real 10-K/10-Q
filings plus 50–100 hand-curated tasks. Nothing here is trained on — ever.
It is a test-only generalization probe: *synthetic training → real documents*.

## Rules (read before adding anything)

1. **Test-only.** No training script may take `--tasks data/real_sec/*`.
   The filenames are deliberately distinct from `data/tasks/` so a tired
   copy-paste can't contaminate training. The day a real filing enters a
   training file, Experiment 2 is void.
2. **Same schema as `data/documents/filings.json`.** The env, tools, reward,
   harness, and GRPO all work unchanged — point them with:
   `HYDE_FILINGS=data/real_sec/filings.json`.
3. **Evidence spans must be verbatim substrings** of the stored section text
   (validator enforces this). Real filings are long: store *only the excerpt
   you cite* (a few paragraphs around each figure), not the whole 10-K.
   Record the source (company, form, fiscal year, accession number) in the
   entry so anyone can re-verify.
4. **Values are curator-verified.** Two pairs of eyes per task: the figure in
   `ground_truth.value` must equal the figure in the cited excerpt, read
   independently of whoever wrote the excerpt.

## Filing entry schema (one key per filing id)

```json
{
  "AAPL-10K-2024": {
    "filing_id": "AAPL-10K-2024",
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "year": 2024,
    "form": "10-K",
    "source": "SEC EDGAR accession 0000320193-0000320193 (verify before use)",
    "values": {"revenue": 391035, "net_income": 93736},
    "tables": {"income_statement": {"revenue": 391035, "net_income": 93736}},
    "sections": {
      "financial_statements": "<verbatim excerpt containing the figures>"
    }
  }
}
```

Constraints the validator checks: required keys; `tables`/`sections` non-empty;
every `values` figure's money string appears in `financial_statements`;
`filing_id` matches `{TICKER}-{FORM}-{YEAR}`; no duplicate ids.

## Task schema (one JSON object per line in `tasks.jsonl`)

Identical fields to synthetic tasks: `task_id` (`sec-0000`…), `task_family`
(`financial_metric_extraction`), `prompt`, `company`, `ticker`, `year`,
`metric`, `filing_id`, `section`, `ground_truth` (`value`, `value_str`,
`filing_id`, `section`, `evidence_substr`). Prompts should be phrased the way
the original request was made, not normalized — realism is the point.

## Curation procedure

1. Pick company + form + year. Download from EDGAR, note the accession number.
2. Extract the excerpt around each target figure (keep units! "$ millions" vs
   "$ billions" has burned every financial benchmark ever).
3. Write the filing entry; run `python data/real_sec/validate.py`.
4. Write 3–8 tasks per filing with varied phrasing; re-run the validator.
5. Second person verifies 100% of values against the live EDGAR source.
6. Run, never train:
   `HYDE_FILINGS=data/real_sec/filings.json python evals/run_model_baseline.py
   --model <ckpt> --tasks data/real_sec/tasks.jsonl --out experiments/sec_<model>.jsonl --limit 100`

## Target size

10–20 filings, 50–100 tasks. Small on purpose: each task is hand-verified, and
the statistical claim is transfer/no-transfer, not a leaderboard.
