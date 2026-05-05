"""Incremental relay-style LoRA training for CPU-constrained machines.

Run this script repeatedly. Each run trains only one tiny shard, saves the
adapter, updates the relay state, and exits cleanly so progress accumulates
over time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

try:
    from .build_relay_curriculum import CurriculumConfig, build_curriculum
    from .finetune_utils import add_perplexity, load_jsonl_records, prepare_split_dataset
except ImportError:
    from build_relay_curriculum import CurriculumConfig, build_curriculum
    from finetune_utils import add_perplexity, load_jsonl_records, prepare_split_dataset


@dataclass
class RelayConfig:
    base_model_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\base_model_fp32")
    data_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\training_data\teranga_merged.jsonl")
    curriculum_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\training_data\relay_curriculum.json")
    curriculum_analysis_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\training_data\relay_dataset_analysis.json")
    output_dir: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\lora_adapter_relay")
    state_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\lora_adapter_relay\relay_state.json")
    history_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\lora_adapter_relay\relay_history.json")
    full_eval_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\lora_adapter_relay\full_eval_metrics.json")
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    logging_steps: int = 1
    seed: int = 42
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    quick_eval_size: int = 4
    dry_run: bool = field(default_factory=lambda: os.getenv("FOUNDER_AI_RELAY_DRY_RUN", "").lower() == "true")
    max_steps_override: int | None = field(default_factory=lambda: _read_optional_int_env("FOUNDER_AI_RELAY_MAX_STEPS"))
    max_seq_length_override: int | None = field(default_factory=lambda: _read_optional_int_env("FOUNDER_AI_RELAY_MAX_SEQ_LENGTH"))
    quick_eval_size_override: int | None = field(default_factory=lambda: _read_optional_int_env("FOUNDER_AI_RELAY_QUICK_EVAL_SIZE"))


def _read_optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def ensure_curriculum(config: RelayConfig) -> dict:
    if not config.curriculum_path.exists() or not config.curriculum_analysis_path.exists():
        analysis, curriculum = build_curriculum(
            CurriculumConfig(
                data_path=config.data_path,
                analysis_path=config.curriculum_analysis_path,
                curriculum_path=config.curriculum_path,
            )
        )
        config.curriculum_analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        config.curriculum_path.write_text(json.dumps(curriculum, ensure_ascii=False, indent=2), encoding="utf-8")
        return curriculum
    return json.loads(config.curriculum_path.read_text(encoding="utf-8"))


def load_state(config: RelayConfig, curriculum: dict) -> dict:
    if config.state_path.exists():
        return json.loads(config.state_path.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "cycle_index": 0,
        "stage_index": 0,
        "shard_index": 0,
        "session_index": 0,
        "validation_cursor": 0,
        "completed": False,
        "full_eval_every_n_sessions": curriculum.get("full_eval_every_n_sessions", 4),
        "max_cycles": curriculum.get("max_cycles", 2),
        "history": [],
    }


def save_state(config: RelayConfig, state: dict) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    config.history_path.write_text(json.dumps(state["history"], ensure_ascii=False, indent=2), encoding="utf-8")


def load_records_by_id(config: RelayConfig) -> dict[str, dict]:
    records = load_jsonl_records(config.data_path)
    record_map: dict[str, dict] = {}
    for index, record in enumerate(records, start=1):
        record_map[str(record.get("id") or f"record_{index:04d}")] = record
    return record_map


def current_stage_payload(curriculum: dict, state: dict) -> dict:
    stage = dict(curriculum["stages"][state["stage_index"]])
    return stage


def load_model_and_tokenizer(config: RelayConfig):
    tokenizer = AutoTokenizer.from_pretrained(str(config.base_model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        str(config.base_model_path),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    adapter_config = config.output_dir / "adapter_config.json"
    if adapter_config.exists():
        model = PeftModel.from_pretrained(base_model, str(config.output_dir), is_trainable=True)
    else:
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    return model, tokenizer


def pick_quick_eval_records(curriculum: dict, state: dict, record_map: dict[str, dict], quick_eval_size: int) -> list[dict]:
    validation_ids = curriculum["validation_record_ids"]
    if not validation_ids:
        return []
    start = state["validation_cursor"]
    selected = []
    for offset in range(min(quick_eval_size, len(validation_ids))):
        selected.append(record_map[validation_ids[(start + offset) % len(validation_ids)]])
    state["validation_cursor"] = (start + len(selected)) % len(validation_ids)
    return selected


def evaluate_subset(model, tokenizer, records: list[dict], max_seq_length: int, output_dir: Path, metric_key_prefix: str) -> dict:
    if not records:
        return {}
    dataset = prepare_split_dataset(records, tokenizer, max_seq_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
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
    return add_perplexity(trainer.evaluate(eval_dataset=dataset, metric_key_prefix=metric_key_prefix))


def advance_state(state: dict, curriculum: dict) -> None:
    state["session_index"] += 1
    stage = curriculum["stages"][state["stage_index"]]
    state["shard_index"] += 1
    if state["shard_index"] < stage["num_shards"]:
        return

    state["shard_index"] = 0
    state["stage_index"] += 1
    if state["stage_index"] < len(curriculum["stages"]):
        return

    state["stage_index"] = 0
    state["cycle_index"] += 1
    if state["cycle_index"] >= state["max_cycles"]:
        state["completed"] = True


def maybe_run_full_eval(config: RelayConfig, curriculum: dict, state: dict, record_map: dict[str, dict], model, tokenizer) -> dict | None:
    if state["session_index"] == 0:
        return None
    if state["session_index"] % state["full_eval_every_n_sessions"] != 0 and not state["completed"]:
        return None

    validation_records = [record_map[record_id] for record_id in curriculum["validation_record_ids"]]
    test_records = [record_map[record_id] for record_id in curriculum["test_record_ids"]]
    metrics = {
        "session_index": state["session_index"],
        "cycle_index": state["cycle_index"],
        **evaluate_subset(model, tokenizer, validation_records, 512, config.output_dir, "full_validation"),
        **evaluate_subset(model, tokenizer, test_records, 512, config.output_dir, "full_test"),
    }
    config.full_eval_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    config = RelayConfig()
    curriculum = ensure_curriculum(config)
    state = load_state(config, curriculum)
    if state["completed"]:
        print("Relay training is already complete according to relay_state.json.")
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    record_map = load_records_by_id(config)
    stage = current_stage_payload(curriculum, state)
    if config.max_steps_override is not None:
        stage["max_steps"] = max(1, config.max_steps_override)
    if config.max_seq_length_override is not None:
        stage["max_seq_length"] = max(64, config.max_seq_length_override)
    if config.quick_eval_size_override is not None:
        config.quick_eval_size = max(0, config.quick_eval_size_override)
    shard_record_ids = stage["shards"][state["shard_index"]]
    shard_records = [record_map[record_id] for record_id in shard_record_ids]

    if config.dry_run:
        preview = {
            "message": "Relay dry-run preview.",
            "current_stage": stage["name"],
            "cycle_index": state["cycle_index"],
            "session_index": state["session_index"],
            "shard_index": state["shard_index"],
            "max_seq_length": stage["max_seq_length"],
            "max_steps": stage["max_steps"],
            "record_count": len(shard_record_ids),
            "record_ids": shard_record_ids,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    model, tokenizer = load_model_and_tokenizer(config)
    model.print_trainable_parameters()

    train_dataset = prepare_split_dataset(shard_records, tokenizer, stage["max_seq_length"])
    quick_eval_records = pick_quick_eval_records(curriculum, state, record_map, config.quick_eval_size)
    quick_eval_dataset = prepare_split_dataset(quick_eval_records, tokenizer, stage["max_seq_length"]) if quick_eval_records else None

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        do_train=True,
        do_eval=False,
        save_strategy="no",
        eval_strategy="no",
        max_steps=stage["max_steps"],
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=stage["gradient_accumulation_steps"],
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=True,
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    train_result = trainer.train()
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    quick_metrics = (
        add_perplexity(trainer.evaluate(eval_dataset=quick_eval_dataset, metric_key_prefix="quick_validation"))
        if quick_eval_dataset is not None
        else {}
    )

    history_entry = {
        "session_index": state["session_index"],
        "cycle_index": state["cycle_index"],
        "stage_name": stage["name"],
        "stage_index": state["stage_index"],
        "shard_index": state["shard_index"],
        "max_seq_length": stage["max_seq_length"],
        "shard_record_ids": shard_record_ids,
        "train_metrics": add_perplexity(dict(train_result.metrics)),
        "quick_validation_metrics": quick_metrics,
    }
    state["history"].append(history_entry)

    advance_state(state, curriculum)
    full_eval_metrics = maybe_run_full_eval(config, curriculum, state, record_map, model, tokenizer)
    if full_eval_metrics is not None:
        state["history"][-1]["full_eval_metrics"] = full_eval_metrics

    save_state(config, state)

    summary = {
        "message": "Relay session complete.",
        "current_session": history_entry,
        "next_state": {
            "completed": state["completed"],
            "cycle_index": state["cycle_index"],
            "stage_index": state["stage_index"],
            "shard_index": state["shard_index"],
            "session_index": state["session_index"],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
