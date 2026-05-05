"""Install a Colab-generated LoRA adapter zip into the local FounderAI model folder."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


REQUIRED_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir() -> Path:
    return repo_root() / "models" / "founderai" / "current" / "lora_adapter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install a FounderAI LoRA adapter zip produced by the Colab notebook."
    )
    parser.add_argument("zip_path", type=Path, help="Path to the downloaded founderai_lora_adapter zip file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Target directory for the extracted adapter files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the target directory if it already exists.",
    )
    return parser.parse_args()


def validate_zip(zip_path: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    if zip_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip file, got: {zip_path}")


def prepare_target_dir(target_dir: Path, force: bool) -> None:
    if target_dir.exists():
        if not force:
            raise FileExistsError(
                f"Target directory already exists: {target_dir}. Re-run with --force to replace it."
            )
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def extract_root_files(zip_path: Path, target_dir: Path) -> list[str]:
    extracted: list[str] = []
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
            extracted.append(entry_path.name)
    return sorted(extracted)


def validate_extraction(target_dir: Path) -> None:
    found = {path.name for path in target_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - found)
    if missing:
        raise RuntimeError(f"Adapter installation incomplete. Missing files: {', '.join(missing)}")


def read_metrics(target_dir: Path) -> dict | None:
    metrics_path = target_dir / "training_metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def print_summary(target_dir: Path, extracted: list[str], metrics: dict | None) -> None:
    print("Installed adapter files:")
    for name in extracted:
        print(f"- {name}")

    print("")
    print(f"Adapter directory ready: {target_dir}")
    print("")
    print("Recommended .env values:")
    print("USE_FINETUNED_MODEL=true")
    print("FINETUNED_MODEL_PATH=Qwen/Qwen3-4B")
    print(f"LORA_ADAPTER_PATH={target_dir}")
    print("HF_TOKEN=")

    if metrics:
        print("")
        print("Training metrics snapshot:")
        for key in ("train_loss", "validation_loss", "test_loss", "validation_perplexity", "test_perplexity"):
            if key in metrics:
                print(f"- {key}: {metrics[key]}")


def main() -> None:
    args = parse_args()
    zip_path = args.zip_path.expanduser().resolve()
    target_dir = args.output_dir.expanduser().resolve()

    validate_zip(zip_path)
    prepare_target_dir(target_dir, args.force)
    extracted = extract_root_files(zip_path, target_dir)
    validate_extraction(target_dir)
    metrics = read_metrics(target_dir)
    print_summary(target_dir, extracted, metrics)


if __name__ == "__main__":
    main()
