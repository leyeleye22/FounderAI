"""
Merge LoRA adapter with base Qwen3 model.
Run this after training to create a standalone model.

Usage:
    python merge_lora.py [--base_model PATH] [--lora_adapter PATH] [--output PATH]
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base_model", type=str, default=str(_repo_root() / "base_model_fp32"))
    parser.add_argument("--lora_adapter", type=str, default=str(_repo_root() / "lora_adapter"))
    parser.add_argument("--output", type=str, default=str(_repo_root() / "teranga-qwen3-merged"))
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {args.lora_adapter}...")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter)

    print("Merging adapter weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output}...")
    model.save_pretrained(args.output, safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    print(f"Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
