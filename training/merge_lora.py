"""Merge a LoRA adapter into base weights and save full model. RUN ON KAGGLE.

Used between SFT and GRPO-real: GRPO starts from merged weights + a fresh LoRA
adapter (rather than stacking adapters).

Usage:
  !CUDA_VISIBLE_DEVICES=0 python training/merge_lora.py \
      --base Qwen/Qwen3.5-0.8B-Base \
      --adapter /kaggle/working/qwen35-08b-sft-traj/checkpoint-100 \
      --out /kaggle/working/qwen35-08b-sft-merged
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, a.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"merged -> {a.out}")


if __name__ == "__main__":
    main()
