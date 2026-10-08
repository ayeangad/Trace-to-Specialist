# Tail Guard Kaggle runbook (free T4)

GPU phases (1, 3, 8.1) run here. Execute cells in order, top to bottom, and
paste back every `OUTPUT:` block plus the listed download files.

Repo: `git@github.com:ayeangad/Trace-to-Specialist.git`, branch `tailguard`.

---

## Phase 1 — Rebuild and verify the 3B specialist (~0.5 day)

### Cell 0 (settings, no code)
1. Settings → Accelerator **GPU T4**, Internet **on**.
2. Add secret `HF_TOKEN` (Settings → Secrets).
3. Expected at end: adapter at `/kaggle/working/qwen25-3b-sft`
   (with `adapter_config.json` + `checkpoint-100/adapter_config.json`) and
   `experiments/tailguard/p1_frozen_sft3b.jsonl` (50 tasks, success ≥ 0.90).

### Cell 1 (code) — install and login
```python
!pip install -q "transformers>=5.2.0" trl peft datasets accelerate huggingface_hub jinja2
!pip uninstall -y torchao
from huggingface_hub import login
import os
login(token=os.environ["HF_TOKEN"])
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
EXPECTED: `cuda: True` + a Tesla T4 name.

### Cell 2 (code) — get the repo
```python
!git clone --branch tailguard git@github.com:ayeangad/Trace-to-Specialist.git /kaggle/working/hyde
%cd /kaggle/working/hyde
!git rev-parse HEAD
!git status --short
```
EXPECTED: status clean. Paste back the HEAD hash (recorded as the run's config).

### Cell 3 (code) — regenerate native SFT demos
```python
!python data/generate_tasks.py --n-train 400 --n-dev 60 --n-test 100
!python trajectories/collector.py --tasks data/tasks/train.jsonl --out data/tasks/sft_demos.jsonl
!head -c 300 data/tasks/sft_demos.jsonl
```
EXPECTED: collector prints `kept` ≈ 400 (local prep: 400 kept, 0 dropped);
rows contain a `"turns"` key.
```python
!git diff --stat data/tasks/
```
EXPECTED: only `data/tasks/sft_demos.jsonl` changed
(`generate_tasks.py` is deterministic with seed 42, so
`train/dev/frozen_test.jsonl` and `filings.json` are byte-identical).
If anything else differs: STOP, paste back the diff.

### Cell 4 (code) — train (~1.5 GPU-h)
```python
!CUDA_VISIBLE_DEVICES=0 python training/sft.py --model Qwen/Qwen2.5-3B-Instruct --demos data/tasks/sft_demos.jsonl --out /kaggle/working/qwen25-3b-sft --epochs 2
!ls /kaggle/working/qwen25-3b-sft
!ls /kaggle/working/qwen25-3b-sft/checkpoint-100
```
EXPECTED: out dir contains `adapter_config.json`; `checkpoint-100/adapter_config.json` exists.

### Cell 5 (code) — verify on frozen test
```python
!python evals/run_model_baseline.py --model /kaggle/working/qwen25-3b-sft --base-model Qwen/Qwen2.5-3B-Instruct --tasks data/tasks/frozen_test.jsonl --out experiments/tailguard/p1_frozen_sft3b.jsonl --limit 50
!python -c "
import json
rows=[json.loads(l) for l in open('experiments/tailguard/p1_frozen_sft3b.jsonl')]
n=len(rows)
print('n',n,'mean_reward',sum(r['reward'] for r in rows)/n,'success',sum(r['success'] for r in rows)/n,'had_submit',sum(r.get('had_submit',False) for r in rows)/n)
"
```
EXPECTED: n 50, success ≥ 0.90 (historical 0.96). GATE: if < 0.90, confirm
native demos (`turns` key), 2 epochs, `--base-model` passed; re-run once in a
fresh session; if still < 0.90, STOP (do not lower the gate).

### Cell 6 (after run) — persist, then bring back to laptop
1. Kaggle "Save Version" with GPU outputs.
2. Upload `/kaggle/working/qwen25-3b-sft` as a private Kaggle Dataset named
   `qwen25-3b-sft-tailguard` (later sessions attach it read-only).
3. Download and place at repo root: `experiments/tailguard/p1_frozen_sft3b.jsonl`.
4. Paste back: HEAD hash (Cell 2), `kept` count (Cell 3), `git diff --stat`
   output (Cell 3), `ls` outputs (Cell 4), summary line (Cell 5).

---

## Phase 3 — (to be written when Phase 2 passes)

## Phase 8.1 — (to be written when Phase 7 passes)
