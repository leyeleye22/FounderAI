"""Qwen3 LoRA fine-tuning script with stable split-aware evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

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
class TrainingConfig:
    base_model_path: str = field(
        default=r"C:\Users\Mr LEYE\Downloads\FounderAI\base_model_fp32",
        metadata={"help": "Path to the base Qwen3 model"}
    )
    data_path: str = field(
        default=r"C:\Users\Mr LEYE\Downloads\FounderAI\training_data\teranga_merged.jsonl",
        metadata={"help": "Path to the merged JSONL file"}
    )
    output_dir: str = field(
        default=r"C:\Users\Mr LEYE\Downloads\FounderAI\lora_adapter",
        metadata={"help": "Directory to save the LoRA adapter"}
    )
    metrics_path: str = field(
        default=r"C:\Users\Mr LEYE\Downloads\FounderAI\lora_adapter\training_metrics.json",
        metadata={"help": "Path to write training metrics"}
    )

    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    num_train_epochs: float = field(default=1.0)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    warmup_ratio: float = field(default=0.03)
    max_seq_length: int = field(default=768)
    logging_steps: int = field(default=5)
    save_strategy: str = field(default="epoch")
    eval_strategy: str = field(default="epoch")
    save_total_limit: int = field(default=2)

    use_4bit: bool = field(default=False)
    use_8bit: bool = field(default=False)

    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    seed: int = field(default=42)


def create_lora_config(config: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def create_quantization_config(config: TrainingConfig) -> Optional[BitsAndBytesConfig]:
    if config.use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if config.use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model_and_tokenizer(config: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    quant_config = create_quantization_config(config)
    use_cuda = torch.cuda.is_available()

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if use_cuda else torch.float32,
    }
    if use_cuda and quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.base_model_path, **model_kwargs)
    model.config.use_cache = False

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, create_lora_config(config))
    return model, tokenizer


def build_datasets(config: TrainingConfig, tokenizer):
    records = load_jsonl_records(config.data_path)
    split_map = split_records(records)

    missing = [name for name in ("train", "validation", "test") if not split_map[name]]
    if missing:
        raise ValueError(f"Missing dataset splits in {config.data_path}: {', '.join(missing)}")

    train_dataset = prepare_split_dataset(split_map["train"], tokenizer, config.max_seq_length)
    validation_dataset = prepare_split_dataset(split_map["validation"], tokenizer, config.max_seq_length)
    test_dataset = prepare_split_dataset(split_map["test"], tokenizer, config.max_seq_length)
    return split_map, train_dataset, validation_dataset, test_dataset


def write_metrics(path: str, metrics: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config = TrainingConfig()
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if not torch.cuda.is_available():
        print("CPU-only environment detected. Training is configured conservatively to stay feasible.")

    model, tokenizer = load_model_and_tokenizer(config)
    model.print_trainable_parameters()

    split_map, train_dataset, validation_dataset, test_dataset = build_datasets(config, tokenizer)
    print(json.dumps(dataset_stats(sum(split_map.values(), [])), ensure_ascii=False, indent=2))
    print(
        f"Prepared datasets -> train: {len(train_dataset)}, "
        f"validation: {len(validation_dataset)}, test: {len(test_dataset)}"
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        bf16=config.bf16,
        seed=config.seed,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        weight_decay=0.01,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=config.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    train_output = trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    validation_metrics = trainer.evaluate(eval_dataset=validation_dataset, metric_key_prefix="validation")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    metrics = add_perplexity(
        {
            "config": config.__dict__,
            "train_runtime_seconds": train_output.metrics.get("train_runtime"),
            "train_steps_per_second": train_output.metrics.get("train_steps_per_second"),
            "train_samples_per_second": train_output.metrics.get("train_samples_per_second"),
            "train_loss": train_output.metrics.get("train_loss"),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            **validation_metrics,
            **test_metrics,
        }
    )
    write_metrics(config.metrics_path, metrics)

    print("Training complete.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
