"""GRPO with stateful env. RUN ON KAGGLE (GPU). Smoke first, then real.

Smoke (pipeline debug): Qwen/Qwen3-0.6B, 50 tasks, G=4, 512 tokens.
Real: SFT checkpoint from training/sft.py, same env, short run.

Requires transformers>=5.2.0 (environment_factory) + trl[vllm]-optional.
On free T4: keep use_vllm=False initially; enable continuous-batching only
if OOM-free. Goal of smoke = "does reward move?", not SOTA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_tasks(path: str, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="smoke: Qwen/Qwen3-0.6B | real: /kaggle/working/qwen35-08b-sft")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--group", type=int, default=4, help="rollouts per prompt (G)")
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--no-lora", action="store_true",
                    help="full fine-tune instead of LoRA (likely OOM on free T4)")
    ap.add_argument("--max-steps", type=int, default=20,
                    help="optimizer steps (required: env owns the data, no dataset length)")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from environment.env import FilingEnv
    from trl import GRPOTrainer, GRPOConfig

    tasks = load_tasks(a.tasks, a.limit)

    if a.no_lora:
        model = a.model
    else:
        # LoRA-wrap here (rather than relying on TRL peft_config support):
        # full GRPO fine-tune OOMs a T4 during generation.
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        m = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype="auto", trust_remote_code=True)
        model = get_peft_model(m, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM"))
        model.print_trainable_parameters()

    def make_env():
        # TRL instantiates one env per rollout, then calls reset() with no
        # args -> env self-samples a task from the pool. Do NOT pre-reset here.
        return FilingEnv(tasks=tasks)

    args = GRPOConfig(
        output_dir="/kaggle/working/qwen-grpo",
        max_steps=a.max_steps,  # required when env owns data (no train_dataset)
        num_generations=a.group,
        max_completion_length=a.max_completion,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        use_vllm=False,  # T4-safe default; flip after memory measurement
        logging_steps=5,
        scale_rewards=False,  # avoid question-difficulty bias per TRL tip
    )
    trainer = GRPOTrainer(model=model, args=args,
                          environment_factory=make_env)
    trainer.train()
    print("GRPO done")


if __name__ == "__main__":
    main()
