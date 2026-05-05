"""Colab-friendly QLoRA fine-tuning entrypoint for FounderAI.

This script is tuned for Google Colab free/limited GPU sessions:
- portable paths (no Windows-only defaults)
- 4-bit QLoRA by default
- shorter context windows to fit smaller GPUs
- resumable checkpoints
- Colab-local output paths by default
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    from .finetune_utils import add_perplexity, dataset_stats, load_jsonl_records, prepare_split_dataset, split_records
except ImportError:
    from finetune_utils import add_perplexity, dataset_stats, load_jsonl_records, prepare_split_dataset, split_records

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional in some environments
    plt = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _preferred_cuda_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32

    major, _minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


@dataclass
class ColabTrainingConfig:
    base_model_id: str = field(default_factory=lambda: os.getenv("FOUNDER_AI_COLAB_BASE_MODEL", "Qwen/Qwen3-4B"))
    data_path: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_DATA_PATH", _repo_root() / "training_data" / "teranga_merged.jsonl"))
    output_dir: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_OUTPUT_DIR", Path("/content/founderai-colab-v1/lora_adapter")))
    metrics_path: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_METRICS_PATH", Path("/content/founderai-colab-v1/lora_adapter/training_metrics.json")))
    history_path: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_HISTORY_PATH", Path("/content/founderai-colab-v1/lora_adapter/training_history.json")))
    report_path: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_REPORT_PATH", Path("/content/founderai-colab-v1/lora_adapter/training_report.md")))
    plot_path: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_PLOT_PATH", Path("/content/founderai-colab-v1/lora_adapter/loss_curve.png")))
    sample_limit: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_SAMPLE_LIMIT", 0))
    reset_output_dir: bool = field(default_factory=lambda: os.getenv("FOUNDER_AI_COLAB_RESET_OUTPUT", "false").lower() == "true")

    lora_r: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_LORA_R", 8))
    lora_alpha: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_LORA_ALPHA", 16))
    lora_dropout: float = field(default_factory=lambda: _env_float("FOUNDER_AI_COLAB_LORA_DROPOUT", 0.05))
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    num_train_epochs: float = field(default_factory=lambda: _env_float("FOUNDER_AI_COLAB_EPOCHS", 1.0))
    per_device_train_batch_size: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_TRAIN_BATCH", 1))
    per_device_eval_batch_size: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_EVAL_BATCH", 1))
    gradient_accumulation_steps: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_GRAD_ACCUM", 8))
    learning_rate: float = field(default_factory=lambda: _env_float("FOUNDER_AI_COLAB_LR", 2e-4))
    warmup_ratio: float = field(default_factory=lambda: _env_float("FOUNDER_AI_COLAB_WARMUP_RATIO", 0.03))
    max_seq_length: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_MAX_SEQ_LENGTH", 512))
    logging_steps: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_LOGGING_STEPS", 5))
    save_steps: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_SAVE_STEPS", 25))
    eval_steps: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_EVAL_STEPS", 25))
    save_total_limit: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_SAVE_TOTAL_LIMIT", 2))

    use_4bit: bool = field(default_factory=lambda: os.getenv("FOUNDER_AI_COLAB_USE_4BIT", "true").lower() == "true")
    use_8bit: bool = field(default_factory=lambda: os.getenv("FOUNDER_AI_COLAB_USE_8BIT", "false").lower() == "true")
    gradient_checkpointing: bool = field(default_factory=lambda: os.getenv("FOUNDER_AI_COLAB_GRADIENT_CHECKPOINTING", "true").lower() == "true")
    seed: int = field(default_factory=lambda: _env_int("FOUNDER_AI_COLAB_SEED", 42))


def create_lora_config(config: ColabTrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def create_quantization_config(config: ColabTrainingConfig) -> Optional[BitsAndBytesConfig]:
    if config.use_4bit:
        compute_dtype = _preferred_cuda_dtype()
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    if config.use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model_and_tokenizer(config: ColabTrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    quant_config = create_quantization_config(config)
    use_cuda = torch.cuda.is_available()
    default_dtype = _preferred_cuda_dtype()

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": default_dtype,
        "low_cpu_mem_usage": True,
    }
    if quant_config is not None and use_cuda:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.base_model_id, **model_kwargs)
    model.config.use_cache = False

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, create_lora_config(config))
    return model, tokenizer


def build_datasets(config: ColabTrainingConfig, tokenizer):
    records = load_jsonl_records(config.data_path)
    split_map = split_records(records)

    if config.sample_limit > 0:
        split_map["train"] = split_map["train"][: config.sample_limit]

    missing = [name for name in ("train", "validation", "test") if not split_map[name]]
    if missing:
        raise ValueError(f"Missing dataset splits in {config.data_path}: {', '.join(missing)}")

    train_dataset = prepare_split_dataset(split_map["train"], tokenizer, config.max_seq_length)
    validation_dataset = prepare_split_dataset(split_map["validation"], tokenizer, config.max_seq_length)
    test_dataset = prepare_split_dataset(split_map["test"], tokenizer, config.max_seq_length)
    return split_map, train_dataset, validation_dataset, test_dataset


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_history(log_history: list[dict]) -> dict[str, list[dict[str, float]]]:
    train_points: list[dict[str, float]] = []
    eval_points: list[dict[str, float]] = []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_points.append(
                {
                    "step": float(entry.get("step", len(train_points) + 1)),
                    "loss": float(entry["loss"]),
                    "learning_rate": float(entry.get("learning_rate", 0.0)),
                    "epoch": float(entry.get("epoch", 0.0)),
                }
            )
        if "eval_loss" in entry:
            eval_points.append(
                {
                    "step": float(entry.get("step", len(eval_points) + 1)),
                    "eval_loss": float(entry["eval_loss"]),
                    "epoch": float(entry.get("epoch", 0.0)),
                }
            )

    return {"train": train_points, "eval": eval_points}


def summarize_overfit(*, history: dict[str, list[dict[str, float]]], train_loss: float | None, validation_loss: float | None, test_loss: float | None) -> dict:
    eval_points = history.get("eval", [])
    train_points = history.get("train", [])
    reasons: list[str] = []

    best_eval_loss = None
    best_eval_step = None
    final_eval_loss = validation_loss
    overfit_risk = "low"

    def bump_risk(level: str) -> None:
        nonlocal overfit_risk
        order = {"low": 0, "medium": 1, "high": 2}
        if order[level] > order[overfit_risk]:
            overfit_risk = level

    if eval_points:
        best_point = min(eval_points, key=lambda item: item["eval_loss"])
        best_eval_loss = best_point["eval_loss"]
        best_eval_step = best_point["step"]
        if final_eval_loss is None:
            final_eval_loss = eval_points[-1]["eval_loss"]

    if best_eval_loss is not None and final_eval_loss is not None:
        drift = final_eval_loss - best_eval_loss
        drift_ratio = drift / max(best_eval_loss, 1e-8)
        if drift_ratio > 0.10:
            bump_risk("high")
            reasons.append("final validation loss is more than 10% worse than the best validation loss")
        elif drift_ratio > 0.04:
            bump_risk("medium")
            reasons.append("final validation loss is meaningfully above the best validation loss")

    if len(eval_points) >= 3:
        last_eval_losses = [point["eval_loss"] for point in eval_points[-3:]]
        if last_eval_losses[0] < last_eval_losses[1] < last_eval_losses[2]:
            bump_risk("high" if overfit_risk == "medium" else "medium")
            reasons.append("validation loss increased for the last three evaluation points")

    if train_loss is not None and validation_loss is not None:
        gap = validation_loss - train_loss
        if gap > 0.75:
            bump_risk("high")
            reasons.append("generalization gap between train and validation loss is large")
        elif gap > 0.35:
            bump_risk("medium")
            reasons.append("validation loss is noticeably above train loss")

    if test_loss is not None and validation_loss is not None:
        test_gap = test_loss - validation_loss
        if test_gap > 0.25:
            bump_risk("medium")
            reasons.append("test loss is worse than validation loss")

    if not reasons:
        reasons.append("no strong overfitting signal detected from the available loss curves")

    return {
        "risk_level": overfit_risk,
        "best_validation_loss": best_eval_loss,
        "best_validation_step": best_eval_step,
        "final_validation_loss": final_eval_loss,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "reasons": reasons,
        "train_points": len(train_points),
        "eval_points": len(eval_points),
    }


def write_report(path: Path, *, metrics: dict, overfit_summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FounderAI Colab Training Report",
        "",
        "## Summary",
        "",
        f"- Base model: `{metrics['config']['base_model_id']}`",
        f"- Train loss: `{metrics.get('train_loss')}`",
        f"- Validation loss: `{metrics.get('validation_loss')}`",
        f"- Test loss: `{metrics.get('test_loss')}`",
        f"- Validation perplexity: `{metrics.get('validation_perplexity')}`",
        f"- Test perplexity: `{metrics.get('test_perplexity')}`",
        f"- Overfit risk: `{overfit_summary['risk_level']}`",
        "",
        "## Overfit analysis",
        "",
    ]
    for reason in overfit_summary["reasons"]:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## Curve checkpoints",
            "",
            f"- Best validation loss: `{overfit_summary.get('best_validation_loss')}`",
            f"- Best validation step: `{overfit_summary.get('best_validation_step')}`",
            f"- Final validation loss: `{overfit_summary.get('final_validation_loss')}`",
            f"- Logged train points: `{overfit_summary.get('train_points')}`",
            f"- Logged eval points: `{overfit_summary.get('eval_points')}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_history(path: Path, history: dict[str, list[dict[str, float]]]) -> str | None:
    if plt is None:
        return None

    train_points = history.get("train", [])
    eval_points = history.get("eval", [])
    if not train_points and not eval_points:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if train_points:
        plt.plot([point["step"] for point in train_points], [point["loss"] for point in train_points], label="train_loss")
    if eval_points:
        plt.plot([point["step"] for point in eval_points], [point["eval_loss"] for point in eval_points], marker="o", label="validation_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("FounderAI Colab loss curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def validate_saved_adapter(output_dir: Path) -> dict:
    expected_any = [
        output_dir / "adapter_model.safetensors",
        output_dir / "adapter_model.bin",
    ]
    adapter_config = output_dir / "adapter_config.json"
    tokenizer_config = output_dir / "tokenizer_config.json"
    generated_files = sorted(path.name for path in output_dir.iterdir()) if output_dir.exists() else []

    if not adapter_config.exists():
        raise RuntimeError(
            f"Training finished but adapter_config.json is missing in {output_dir}. Files found: {generated_files}"
        )

    if not any(path.exists() for path in expected_any):
        raise RuntimeError(
            f"Training finished but no adapter weights were saved in {output_dir}. Files found: {generated_files}"
        )

    if not tokenizer_config.exists():
        raise RuntimeError(
            f"Training finished but tokenizer_config.json is missing in {output_dir}. Files found: {generated_files}"
        )

    return {
        "output_dir": str(output_dir),
        "files": generated_files,
    }


def gpu_summary() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "cuda": True,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "compute_capability": f"{major}.{minor}",
        "preferred_dtype": str(_preferred_cuda_dtype()).replace("torch.", ""),
    }


def main() -> None:
    config = ColabTrainingConfig()
    if config.reset_output_dir and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. In Colab, switch to Runtime > Change runtime type > T4 GPU before launching training."
        )

    print(
        json.dumps(
            {
                "gpu": gpu_summary(),
                "config": {
                    **asdict(config),
                    "data_path": str(config.data_path),
                    "output_dir": str(config.output_dir),
                    "metrics_path": str(config.metrics_path),
                    "history_path": str(config.history_path),
                    "report_path": str(config.report_path),
                    "plot_path": str(config.plot_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    model, tokenizer = load_model_and_tokenizer(config)
    model.print_trainable_parameters()

    split_map, train_dataset, validation_dataset, test_dataset = build_datasets(config, tokenizer)
    merged_records = split_map["train"] + split_map["validation"] + split_map["test"]
    print(json.dumps(dataset_stats(merged_records), ensure_ascii=False, indent=2))
    print(
        f"Prepared datasets -> train: {len(train_dataset)}, "
        f"validation: {len(validation_dataset)}, test: {len(test_dataset)}"
    )

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
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
        fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        bf16=bool(torch.cuda.is_available() and _preferred_cuda_dtype() == torch.bfloat16),
        seed=config.seed,
        report_to="none",
        optim="paged_adamw_8bit" if (config.use_4bit or config.use_8bit) else "adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        weight_decay=0.01,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    last_checkpoint = None if config.reset_output_dir else get_last_checkpoint(str(config.output_dir))
    train_output = trainer.train(resume_from_checkpoint=last_checkpoint) if last_checkpoint else trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    saved_adapter_manifest = validate_saved_adapter(config.output_dir)

    validation_metrics = trainer.evaluate(eval_dataset=validation_dataset, metric_key_prefix="validation")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    history = extract_history(trainer.state.log_history)
    write_metrics(config.history_path, history)
    overfit_summary = summarize_overfit(
        history=history,
        train_loss=train_output.metrics.get("train_loss"),
        validation_loss=validation_metrics.get("validation_loss"),
        test_loss=test_metrics.get("test_loss"),
    )
    plot_file = plot_history(config.plot_path, history)

    metrics = add_perplexity(
        {
            "config": {
                **asdict(config),
                "data_path": str(config.data_path),
                "output_dir": str(config.output_dir),
                "metrics_path": str(config.metrics_path),
                "history_path": str(config.history_path),
                "report_path": str(config.report_path),
                "plot_path": str(config.plot_path),
            },
            "gpu": gpu_summary(),
            "train_runtime_seconds": train_output.metrics.get("train_runtime"),
            "train_steps_per_second": train_output.metrics.get("train_steps_per_second"),
            "train_samples_per_second": train_output.metrics.get("train_samples_per_second"),
            "train_loss": train_output.metrics.get("train_loss"),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "resumed_from_checkpoint": last_checkpoint,
            "saved_adapter_manifest": saved_adapter_manifest,
            "training_history": history,
            "overfit_analysis": overfit_summary,
            "loss_curve_path": plot_file,
            **validation_metrics,
            **test_metrics,
        }
    )
    write_metrics(config.metrics_path, metrics)
    write_report(config.report_path, metrics=metrics, overfit_summary=overfit_summary)

    print("Colab training complete.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
