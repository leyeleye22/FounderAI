"""Merge the external Venture Discovery dataset with the Teranga native set.

This script keeps explicit train/validation/test splits so fine-tuning and
evaluation stay stable across runs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
EXTERNAL_DATA_ROOT = Path(r"C:\Users\Mr LEYE\Downloads\DataSet Project help")
VENTURE_JSONL_PATH = EXTERNAL_DATA_ROOT / "venture_discovery_training_dataset.jsonl"
TERANGA_PATH = ROOT / "teranga_finetune_advanced.jsonl"
BEHAVIOR_REPAIR_PATH = ROOT / "behavior_repair_dataset.jsonl"

MERGED_PATH = ROOT / "teranga_merged.jsonl"
TRAIN_PATH = ROOT / "teranga_train.jsonl"
VALIDATION_PATH = ROOT / "teranga_validation.jsonl"
TEST_PATH = ROOT / "teranga_test.jsonl"
STATS_PATH = ROOT / "teranga_merged_stats.json"

VALID_SPLITS = {"train", "validation", "test"}

TERANGA_SYSTEM_PROMPTS = {
    "fr": (
        "Tu es le copilot de Teranga Power, un assistant IA pour fondateurs africains. "
        "Tu connais chaque module de la plateforme et leurs liens: problem-statement, "
        "problem-validation, research, icp, business-model-canvas, go-to-market, "
        "market-sizing, competitive-landscape, roi, user-journey, workshop, sprints, gamma. "
        "Tu restes toujours dans le module courant. Tu ne donnes jamais de reponses vagues "
        "de consultant. Tu es court d'abord, puis detaille si demande. Tu n'inventes jamais "
        "de chiffres. Tu proposes toujours des actions concretes. Tu fais des liens entre "
        "les modules quand c'est pertinent."
    ),
    "en": (
        "You are the Teranga Power copilot, an AI assistant for African founders. "
        "You know every module of the platform and their connections: problem-statement, "
        "problem-validation, research, icp, business-model-canvas, go-to-market, "
        "market-sizing, competitive-landscape, roi, user-journey, workshop, sprints, gamma. "
        "You always stay in the current module. You never give vague consultant answers. "
        "You are brief first, then detailed if asked. You never invent numbers. "
        "You always propose concrete actions. You make cross-module links when relevant."
    ),
}

MODULE_MAP = {
    "problem_statement_generation": "problem-statement",
    "problem_statement_rewrite": "problem-statement",
    "problem_validation_plan": "problem-validation",
    "interview_script": "research",
    "interview_debrief": "research",
    "icp_creation": "icp",
    "icp_critique": "icp",
    "bmc_draft": "business-model-canvas",
    "assumption_mapping": "business-model-canvas",
    "market_sizing": "market-sizing",
    "roi_modeling": "roi",
    "user_journey_map": "user-journey",
}

MODULE_LABELS_FR = {
    "problem-statement": "Enonce du probleme",
    "problem-validation": "Validation du probleme",
    "research": "Recherche terrain",
    "icp": "Ideal Customer Profile",
    "business-model-canvas": "Business Model Canvas",
    "go-to-market": "Go-to-Market",
    "market-sizing": "Taille du marche",
    "competitive-landscape": "Paysage concurrentiel",
    "roi": "ROI",
    "user-journey": "User Journey",
    "workshop": "Workshop",
    "sprints": "Sprints",
    "gamma": "Gamma Pitch",
}

MODULE_LABELS_EN = {
    "problem-statement": "Problem Statement",
    "problem-validation": "Problem Validation",
    "research": "Field Research",
    "icp": "Ideal Customer Profile",
    "business-model-canvas": "Business Model Canvas",
    "go-to-market": "Go-to-Market",
    "market-sizing": "Market Sizing",
    "competitive-landscape": "Competitive Landscape",
    "roi": "ROI",
    "user-journey": "User Journey",
    "workshop": "Workshop",
    "sprints": "Sprints",
    "gamma": "Gamma Pitch",
}


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_split(split: str | None) -> str:
    candidate = (split or "").strip().lower()
    return candidate if candidate in VALID_SPLITS else "train"


def split_for_native_record(index: int) -> str:
    remainder = index % 10
    if remainder == 0:
        return "test"
    if remainder == 1:
        return "validation"
    return "train"


def adapt_venture_record(record: dict) -> dict:
    task_type = record["task_type"]
    language = record.get("language", "fr")
    module = MODULE_MAP.get(task_type, "problem-statement")
    module_label = (
        MODULE_LABELS_FR.get(module, module)
        if language == "fr"
        else MODULE_LABELS_EN.get(module, module)
    )
    original_user = record["messages"][1]["content"]
    scenario = record.get("scenario", "Projet")

    if language == "fr":
        module_prefix = f"Page actuelle: {module_label} (Module: {module})\nProjet: {scenario}\n\n"
    else:
        module_prefix = f"Current page: {module_label} (Module: {module})\nProject: {scenario}\n\n"

    adapted = dict(record)
    adapted["messages"] = [dict(message) for message in record["messages"]]
    adapted["messages"][0]["content"] = TERANGA_SYSTEM_PROMPTS[language]
    adapted["messages"][1]["content"] = module_prefix + original_user
    adapted["teranga_module"] = module
    adapted["split"] = normalize_split(record.get("split"))
    adapted["source_dataset"] = "venture_discovery"
    return adapted


def adapt_native_record(record: dict, index: int) -> dict:
    adapted = dict(record)
    adapted["teranga_module"] = adapted.get("teranga_module", "mixed")
    adapted["split"] = normalize_split(adapted.get("split")) if adapted.get("split") else split_for_native_record(index)
    adapted["source_dataset"] = adapted.get("source_dataset", "teranga_native")
    return adapted


def deduplicate(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique_records: list[dict] = []
    for record in records:
        signature = hashlib.sha256(
            json.dumps(record["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if signature in seen:
            continue
        seen.add(signature)
        unique_records.append(record)
    return unique_records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_stats(records: list[dict]) -> dict:
    by_split = Counter(record.get("split", "train") for record in records)
    by_language = Counter(record.get("language", "mixed") for record in records)
    by_module = Counter(record.get("teranga_module", "unknown") for record in records)
    by_source = Counter(record.get("source_dataset", "unknown") for record in records)
    by_task = Counter(record.get("task_type", "teranga_native") for record in records)
    return {
        "total_records": len(records),
        "by_split": dict(sorted(by_split.items())),
        "by_language": dict(sorted(by_language.items())),
        "by_module": dict(sorted(by_module.items())),
        "by_source_dataset": dict(sorted(by_source.items())),
        "by_task_type": dict(sorted(by_task.items())),
    }


def main() -> None:
    if not VENTURE_JSONL_PATH.exists():
        raise FileNotFoundError(f"Missing external dataset: {VENTURE_JSONL_PATH}")
    if not TERANGA_PATH.exists():
        raise FileNotFoundError(f"Missing native dataset: {TERANGA_PATH}")

    venture_records = [adapt_venture_record(record) for record in load_jsonl(VENTURE_JSONL_PATH)]
    native_records = [
        adapt_native_record(record, index)
        for index, record in enumerate(load_jsonl(TERANGA_PATH), start=1)
    ]
    repair_records: list[dict] = []
    if BEHAVIOR_REPAIR_PATH.exists():
        repair_records = [
            adapt_native_record(record, index)
            for index, record in enumerate(load_jsonl(BEHAVIOR_REPAIR_PATH), start=1)
        ]

    merged_records = deduplicate(venture_records + native_records + repair_records)
    train_records = [record for record in merged_records if record["split"] == "train"]
    validation_records = [record for record in merged_records if record["split"] == "validation"]
    test_records = [record for record in merged_records if record["split"] == "test"]

    write_jsonl(MERGED_PATH, merged_records)
    write_jsonl(TRAIN_PATH, train_records)
    write_jsonl(VALIDATION_PATH, validation_records)
    write_jsonl(TEST_PATH, test_records)

    stats = build_stats(merged_records)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Merged dataset written to {MERGED_PATH}")
    print(f"Train: {len(train_records)} | Validation: {len(validation_records)} | Test: {len(test_records)}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
