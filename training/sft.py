"""LoRA SFT for Qwen3.5-0.8B on V1 demos. RUN ON KAGGLE (GPU), not laptop.

Kaggle setup (free T4):
  !pip install -q transformers>=5.2.0 trl peft datasets accelerate
  Attach this repo as input, demos jsonl from trajectories/collector.py

Why SFT before GRPO: isolates supervised-specialization gain so the
Base->SFT->GRPO ablation can attribute marginal RL value.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--demos", required=True, help="jsonl with prompt/completion")
    ap.add_argument("--out", default="/kaggle/working/qwen35-08b-sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Single-device map: with device_map="auto" on multi-GPU Kaggle sessions the
    # Trainer wraps the already-dispatched model in DataParallel -> cuda:0/cuda:1
    # mismatch. The 0.8B fits on one T4, so pin it. (Also launch with
    # CUDA_VISIBLE_DEVICES=0 as belt-and-braces.)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype="auto", trust_remote_code=True,
        device_map={"": 0})
    model.config.use_cache = False  # required with gradient checkpointing
    # Qwen3.5 vision tower frozen / unused for text-only task
    for n, p in model.named_parameters():
        if "visual" in n.lower() or "vision" in n.lower():
            p.requires_grad = False

    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))

    ds = load_dataset("json", data_files=a.demos, split="train")

    # Train text must mirror the eval/GRPO prompt format, else SFT learns a style
    # the tool loop never sees (proven twice: answers-only and TOOL:-protocol).
    # Native rows (turns) render via the model's own chat template + TOOL_SCHEMAS.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from environment.tools import TOOL_SCHEMAS

    def to_text(r):
        if r.get("turns"):
            return {"text": tok.apply_chat_template(
                r["turns"], tools=TOOL_SCHEMAS, tokenize=False)}
        if r.get("text"):
            return {"text": r["text"]}
        raise KeyError("demo row has neither native 'turns' nor 'text'; "
                       "regenerate with trajectories/collector.py")

    ds = ds.map(to_text)

    args = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs,
                     per_device_train_batch_size=1, gradient_accumulation_steps=8,
                     learning_rate=a.lr, fp16=True, gradient_checkpointing=True,
                     logging_steps=10, save_steps=100)
    SFTTrainer(model=model, train_dataset=ds, args=args,
               processing_class=tok).train()
    tok.save_pretrained(a.out)  # adapter ckpt has no tokenizer files otherwise
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
