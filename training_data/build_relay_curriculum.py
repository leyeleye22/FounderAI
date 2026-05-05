"""Build a CPU-friendly relay curriculum from the merged dataset.

The goal is not to finish training in one marathon session. Instead, we split
the work into many small, resumable relay sessions with a gradual curriculum:
short/easy samples first, then medium analytical samples, then long/integrated
examples.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .finetune_utils import build_record_profiles, dataset_stats, load_jsonl_records
except ImportError:
    from finetune_utils import build_record_profiles, dataset_stats, load_jsonl_records


TRAIN_TASKS_STAGE_1 = {
    "problem_statement_generation",
    "problem_statement_rewrite",
    "interview_script",
    "icp_creation",
}

TRAIN_TASKS_STAGE_2 = {
    "problem_validation_plan",
    "interview_debrief",
    "icp_critique",
    "market_sizing",
    "roi_modeling",
}


@dataclass
class CurriculumConfig:
    data_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\training_data\teranga_merged.jsonl")
    analysis_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\training_data\relay_dataset_analysis.json")
    curriculum_path: Path = Path(r"C:\Users\Mr LEYE\Downloads\FounderAI\training_data\relay_curriculum.json")
    stage_1_shard_size: int = 8
    stage_2_shard_size: int = 6
    stage_3_shard_size: int = 4
    warmup_records_target: int = 24


def chunk_ids(record_ids: list[str], shard_size: int) -> list[list[str]]:
    return [record_ids[idx : idx + shard_size] for idx in range(0, len(record_ids), shard_size)]


def stage_name_for_profile(profile) -> str:
    if profile.task_type in TRAIN_TASKS_STAGE_2 and profile.estimated_tokens <= 560:
        return "build-up"
    return "integration"


def build_curriculum(config: CurriculumConfig) -> tuple[dict, dict]:
    records = load_jsonl_records(config.data_path)
    profiles = build_record_profiles(records)

    train_profiles = [profile for profile in profiles if profile.split == "train"]
    validation_profiles = [profile for profile in profiles if profile.split == "validation"]
    test_profiles = [profile for profile in profiles if profile.split == "test"]

    grouped: dict[str, list] = defaultdict(list)
    sorted_train_profiles = sorted(
        train_profiles,
        key=lambda item: (item.estimated_tokens, item.difficulty_score, item.record_id),
    )
    warmup_ids = {profile.record_id for profile in sorted_train_profiles[: config.warmup_records_target]}

    for profile in train_profiles:
        if profile.record_id in warmup_ids:
            grouped["warmup"].append(profile)
        else:
            grouped[stage_name_for_profile(profile)].append(profile)

    for stage_profiles in grouped.values():
        stage_profiles.sort(key=lambda item: (item.difficulty_score, item.estimated_tokens, item.record_id))

    stages = [
        {
            "name": "warmup",
            "description": "Short, structured coaching tasks to warm up the LoRA adapter cheaply on CPU.",
            "max_seq_length": 384,
            "max_steps": 2,
            "gradient_accumulation_steps": 1,
            "shard_size": config.stage_1_shard_size,
            "record_ids": [profile.record_id for profile in grouped["warmup"]],
        },
        {
            "name": "build-up",
            "description": "Medium-length analytical tasks that deepen evidence-based reasoning.",
            "max_seq_length": 448,
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
            "shard_size": config.stage_2_shard_size,
            "record_ids": [profile.record_id for profile in grouped["build-up"]],
        },
        {
            "name": "integration",
            "description": "Longest and most integrative examples, including native Teranga samples.",
            "max_seq_length": 576,
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
            "shard_size": config.stage_3_shard_size,
            "record_ids": [profile.record_id for profile in grouped["integration"]],
        },
    ]

    for stage in stages:
        stage["shards"] = chunk_ids(stage["record_ids"], stage["shard_size"])
        stage["num_records"] = len(stage["record_ids"])
        stage["num_shards"] = len(stage["shards"])
        del stage["record_ids"]

    analysis = {
        "dataset": dataset_stats(records),
        "train_task_mix": dict(sorted(Counter(profile.task_type for profile in train_profiles).items())),
        "validation_task_mix": dict(sorted(Counter(profile.task_type for profile in validation_profiles).items())),
        "test_task_mix": dict(sorted(Counter(profile.task_type for profile in test_profiles).items())),
        "train_stage_mix": {
            "warmup": len(grouped["warmup"]),
            "build-up": len(grouped["build-up"]),
            "integration": len(grouped["integration"]),
        },
        "recommendation": {
            "why_relay": [
                "CPU training is too slow for one full run on 202 train records.",
                "Starting with shorter records reduces early-step cost and lets the adapter learn the house style first.",
                "Manual stop/resume across many short sessions is more realistic on this machine than marathon training.",
            ],
            "expected_usage": [
                "Run one relay session when the machine is free.",
                "Let the trainer save LoRA weights and state after each shard.",
                "Resume later without losing progress.",
            ],
        },
    }

    curriculum = {
        "version": 1,
        "data_path": str(config.data_path),
        "analysis_path": str(config.analysis_path),
        "stages": stages,
        "validation_record_ids": [profile.record_id for profile in validation_profiles],
        "test_record_ids": [profile.record_id for profile in test_profiles],
        "full_eval_every_n_sessions": 4,
        "max_cycles": 2,
    }
    return analysis, curriculum


def main() -> None:
    config = CurriculumConfig()
    analysis, curriculum = build_curriculum(config)
    config.analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    config.curriculum_path.write_text(json.dumps(curriculum, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote analysis to {config.analysis_path}")
    print(f"Wrote curriculum to {config.curriculum_path}")
    print(json.dumps(analysis["train_stage_mix"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
