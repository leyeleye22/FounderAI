"""Evaluate a saved LoRA adapter on the explicit validation/test splits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

try:
    from .finetune_utils import (
        add_perplexity,
        dataset_stats,
        load_jsonl_records,
        prepare_split_dataset,
        split_records,
    )
except ImportError:
    from finetune_utils import (
        add_perplexity,
        dataset_stats,
        load_jsonl_records,
        prepare_split_dataset,
        split_records,
    )


@dataclass
class EvaluationConfig:
    base_model_path: str = field(default=r"C:\Users\Mr LEYE\Downloads\FounderAI\base_model_fp32")
    adapter_path: str = field(default=r"C:\Users\Mr LEYE\Downloads\FounderAI\lora_adapter")
    data_path: str = field(default=r"C:\Users\Mr LEYE\Downloads\FounderAI\training_data\teranga_merged.jsonl")
    output_path: str = field(default=r"C:\Users\Mr LEYE\Downloads\FounderAI\lora_adapter\evaluation_metrics.json")
    max_seq_length: int = field(default=768)


def write_json(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config = EvaluationConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(model, config.adapter_path)
    model.config.use_cache = False

    records = load_jsonl_records(config.data_path)
    split_map = split_records(records)
    validation_dataset = prepare_split_dataset(split_map["validation"], tokenizer, config.max_seq_length)
    test_dataset = prepare_split_dataset(split_map["test"], tokenizer, config.max_seq_length)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=config.adapter_path,
            do_eval=True,
            report_to="none",
            per_device_eval_batch_size=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            use_cpu=True,
        ),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    metrics = add_perplexity(
        {
            "dataset": dataset_stats(records),
            **trainer.evaluate(eval_dataset=validation_dataset, metric_key_prefix="validation"),
            **trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test"),
        }
    )
    write_json(config.output_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
