"""Utility helpers shared by training and evaluation scripts."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset


VALID_SPLITS = {"train", "validation", "test"}

DEFAULT_CHAT_CHARS_PER_TOKEN = 4.0


def load_jsonl_records(path: str | Path) -> list[dict]:
    file_path = Path(path)
    records: list[dict] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_split(split: str | None) -> str:
    candidate = (split or "").strip().lower()
    return candidate if candidate in VALID_SPLITS else "train"


def split_records(records: list[dict]) -> dict[str, list[dict]]:
    buckets = {"train": [], "validation": [], "test": []}
    for record in records:
        buckets[normalize_split(record.get("split"))].append(record)
    return buckets


def records_to_messages_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_dict({"messages": [record["messages"] for record in records]})


def render_messages_as_text(messages: list[dict]) -> str:
    return "\n".join(message["content"] for message in messages)


def apply_chat_template(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def tokenize_example(example: dict, tokenizer, max_seq_length: int) -> dict:
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    labels = tokenized["input_ids"].copy()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        labels = [token if token != pad_token_id else -100 for token in labels]
    tokenized["labels"] = labels
    return tokenized


def prepare_split_dataset(records: list[dict], tokenizer, max_seq_length: int) -> Dataset:
    dataset = records_to_messages_dataset(records)
    dataset = dataset.map(
        lambda row: apply_chat_template(row, tokenizer),
        remove_columns=["messages"],
    )
    dataset = dataset.map(
        lambda row: tokenize_example(row, tokenizer, max_seq_length),
        remove_columns=["text"],
    )
    return dataset


@dataclass
class RecordProfile:
    record_id: str
    split: str
    language: str
    task_type: str
    module: str
    source_dataset: str
    char_length: int
    estimated_tokens: int
    difficulty_score: int


def estimate_token_count_from_text(text: str, chars_per_token: float = DEFAULT_CHAT_CHARS_PER_TOKEN) -> int:
    return max(1, math.ceil(len(text) / chars_per_token))


def score_record_difficulty(record: dict, estimated_tokens: int) -> int:
    task_type = record.get("task_type", "teranga_native")
    task_weight = {
        "problem_statement_generation": 1,
        "problem_statement_rewrite": 1,
        "interview_script": 1,
        "icp_creation": 1,
        "problem_validation_plan": 2,
        "interview_debrief": 2,
        "icp_critique": 2,
        "market_sizing": 2,
        "roi_modeling": 2,
        "bmc_draft": 3,
        "assumption_mapping": 3,
        "user_journey_map": 3,
        "teranga_native": 4,
    }
    length_weight = 1
    if estimated_tokens >= 550:
        length_weight = 4
    elif estimated_tokens >= 420:
        length_weight = 3
    elif estimated_tokens >= 300:
        length_weight = 2
    return task_weight.get(task_type, 3) * 10 + length_weight


def build_record_profiles(records: list[dict]) -> list[RecordProfile]:
    profiles: list[RecordProfile] = []
    for index, record in enumerate(records, start=1):
        text = render_messages_as_text(record["messages"])
        estimated_tokens = estimate_token_count_from_text(text)
        profiles.append(
            RecordProfile(
                record_id=str(record.get("id") or f"record_{index:04d}"),
                split=normalize_split(record.get("split")),
                language=record.get("language", "mixed"),
                task_type=record.get("task_type", "teranga_native"),
                module=record.get("teranga_module", "unknown"),
                source_dataset=record.get("source_dataset", "unknown"),
                char_length=len(text),
                estimated_tokens=estimated_tokens,
                difficulty_score=score_record_difficulty(record, estimated_tokens),
            )
        )
    return profiles


def dataset_stats(records: list[dict]) -> dict:
    by_split = Counter(normalize_split(record.get("split")) for record in records)
    by_language = Counter(record.get("language", "mixed") for record in records)
    by_source = Counter(record.get("source_dataset", "unknown") for record in records)
    profiles = build_record_profiles(records)
    estimated_tokens = sorted(profile.estimated_tokens for profile in profiles)

    def percentile(p: float) -> int:
        if not estimated_tokens:
            return 0
        idx = min(len(estimated_tokens) - 1, max(0, round((len(estimated_tokens) - 1) * p)))
        return estimated_tokens[idx]

    return {
        "total_records": len(records),
        "by_split": dict(sorted(by_split.items())),
        "by_language": dict(sorted(by_language.items())),
        "by_source_dataset": dict(sorted(by_source.items())),
        "estimated_token_summary": {
            "min": estimated_tokens[0] if estimated_tokens else 0,
            "p50": percentile(0.50),
            "p75": percentile(0.75),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": estimated_tokens[-1] if estimated_tokens else 0,
        },
    }


def add_perplexity(metrics: dict) -> dict:
    enriched = dict(metrics)
    for key in ("eval_loss", "validation_loss", "test_loss", "train_loss"):
        if key in enriched:
            try:
                enriched[key.replace("loss", "perplexity")] = math.exp(float(enriched[key]))
            except OverflowError:
                enriched[key.replace("loss", "perplexity")] = float("inf")
    return enriched
