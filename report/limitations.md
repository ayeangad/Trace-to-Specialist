# Limitations (updated post-V1)

Supersedes the planning-era notes below the line. Standing limitations:

- Synthetic filings, not real EDGAR; single task family; no trace clustering or
  multi-task router optimizer yet (V2).
- Oracle (perfect-policy) SFT demos only; filtering model-generated trajectories
  was never tried — the highest-leverage untested SFT variant.
- 0.8B GRPO: 25 steps, G=4, sparse value/evidence reward. Longer runs, denser
  shaping (near-miss value credit), or a copy curriculum might move it.
- Train (sampling) vs eval (greedy) decoding mismatch observed, never ablated.
- Possible HF-vs-TRL chat-template rendering differences for Qwen3.5 tool calls
  noted, not isolated (loop engaged regardless: ~3 calls/episode).
- Cost measured as tokens/wall-clock on shared T4s, not dollar accounting.
- 3B leg is n=50 frozen; base_native control on 0.8B leg is n=20 (noise caveat
  on the exact 4.4 figure, not on the 0-success conclusion).
- Qwen3.5 runs used slow reference kernels (no causal_conv1d/flash-attention);
  affects speed only, verified correct.

---
Planning-era notes (kept for history):
- Qwen3.5-0.8B is multimodal arch; text-path only, vision frozen.
