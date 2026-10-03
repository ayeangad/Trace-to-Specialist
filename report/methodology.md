# Methodology (V1)

Every component answers one question:

- **Task generator**: can we get deterministic ground truth without enterprise data? (procedural filings, company-split freeze)
- **Task-level eval**: what is correct outcome + process? (value match + filing/section/evidence)
- **FilingEnv**: can the model learn multi-step execution, not answer imitation? (search→open→get_table→submit, TRL `environment_factory` + `get_reward`)
- **Reward (10pt)**: does process quality matter beyond final answer? (filing 2 + section 1 + value 3 + evidence 2 + efficiency 2)
- **SFT**: how much gain is just seeing examples? (LoRA on reward>=9 trajectories)
- **GRPO**: does env-optimization add value beyond SFT? (Base vs SFT vs SFT+GRPO)
- **Router stub**: what is cheapest model clearing ≥90%? (full optimizer in V2)
- **Economics**: is 2pts quality worth 10x cost? (tokens/latency/$ on frozen test)
- **Difficulty levels (Exp 3)**: same env, harder jobs, family-dispatched reward.
  L1 single extraction (filing 2 + section 1 + value 3 + evidence 2 + eff 2).
  L2 dual-metric, one filing (filing 2 + section 1 + valueA 2 + valueB 2 +
  evidence 1+1 + eff 1; submit "VA | VB" / "EA || EB"). L3 YoY across two
  filings (filingA 1 + filingB 1 + section 1 + yoy±0.05 → 2 + verdict 2 +
  evidence 1+1 + eff 1; submit "PCT|yes|no", filings "A,B"). Oracle-verified
  10.0 on all levels; negatives (wrong value, flipped verdict, tolerance edge,
  missing filing) verified point-by-point. Goal: the non-saturated regime where
  GRPO gets a fair test, and a model size × difficulty map.

Non-goals V1: trace clustering, multi-task matrix, dashboard, continual learning.
