"""Publish a local or Colab-generated FounderAI LoRA adapter to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED_FILES = {
    "adapter_config.json",
    "tokenizer_config.json",
}

WEIGHT_FILES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a FounderAI LoRA adapter to the Hugging Face Hub.")
    parser.add_argument(
        "source",
        type=Path,
        help="Path to a local adapter directory or a downloaded Colab zip file.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face model repo id, for example leyeleye22/founderai-qwen3-lora-v1.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private if it does not exist yet.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B",
        help="Base model id to mention in the model card.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=".",
        help="Optional target folder inside the remote repo.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload FounderAI LoRA adapter",
        help="Commit message to use on the Hub.",
    )
    return parser.parse_args()


def validate_adapter_dir(folder: Path) -> dict | None:
    files = {path.name for path in folder.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - files)
    if missing:
        raise RuntimeError(f"Adapter directory is missing required files: {', '.join(missing)}")
    if not any(weight in files for weight in WEIGHT_FILES):
        raise RuntimeError("Adapter directory is missing adapter weights.")

    metrics_path = folder / "training_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return None


def extract_zip_to_temp(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            entry_path = Path(entry.filename)
            if len(entry_path.parts) != 1:
                continue

            output_path = target_dir / entry_path.name
            with archive.open(entry) as source, output_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def build_model_card(base_model: str, repo_id: str, metrics: dict | None) -> str:
    lines = [
        "---",
        "library_name: peft",
        "tags:",
        "- lora",
        "- founderai",
        "- venture-discovery",
        f"base_model: {base_model}",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        "LoRA adapter for FounderAI / Teranga Power style venture-discovery assistance.",
        "",
        "## Intended use",
        "",
        "- sharpen problem statements",
        "- support validation, ICP, BMC, ROI, and GTM coaching",
        "- plug into FounderAI as a lightweight domain adapter",
        "",
        "## Loading example",
        "",
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'base_model = "{base_model}"',
        f'adapter_repo = "{repo_id}"',
        "",
        "tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)",
        "model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)",
        "model = PeftModel.from_pretrained(model, adapter_repo)",
        "```",
    ]

    if metrics:
        lines.extend(
            [
                "",
                "## Training snapshot",
                "",
            ]
        )
        for key in ("train_loss", "validation_loss", "test_loss", "validation_perplexity", "test_perplexity"):
            if key in metrics:
                lines.append(f"- `{key}`: `{metrics[key]}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- produced from occasional Colab fine-tuning runs",
            "- optimized for FounderAI chat structure and startup discovery coaching",
        ]
    )
    return "\n".join(lines) + "\n"


def prepare_upload_dir(source: Path, base_model: str, repo_id: str) -> tuple[Path, dict | None]:
    if source.is_dir():
        metrics = validate_adapter_dir(source)
        return source, metrics

    if source.suffix.lower() != ".zip":
        raise ValueError("Source must be either an adapter directory or a .zip file from Colab.")

    temp_dir = Path(tempfile.mkdtemp(prefix="founderai-lora-hub-"))
    extract_zip_to_temp(source, temp_dir)
    metrics = validate_adapter_dir(temp_dir)
    readme = temp_dir / "README.md"
    if not readme.exists():
        readme.write_text(build_model_card(base_model=base_model, repo_id=repo_id, metrics=metrics), encoding="utf-8")
    return temp_dir, metrics


def main() -> None:
    args = parse_args()
    token = None
    api = HfApi()

    source = args.source.expanduser().resolve()
    upload_dir, _metrics = prepare_upload_dir(source=source, base_model=args.base_model, repo_id=args.repo_id)

    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(upload_dir),
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
    )
    print(f"Published adapter to https://huggingface.co/{args.repo_id}")

    if source.is_file():
        shutil.rmtree(upload_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
