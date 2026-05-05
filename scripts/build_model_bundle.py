from __future__ import annotations

import argparse
import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_INCLUDE_PATHS = [
    "training_data",
    "lora_adapter",
    "lora_adapter_relay",
    "lora_adapter_cpu_pilot_bench",
    "config.json",
    "generation_config.json",
]


def resolve_version(explicit_version: str | None) -> str:
    if explicit_version:
        return explicit_version

    sha = os.getenv("GITHUB_SHA", "local")[:7]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{sha}"


def find_existing_paths(root: Path, include_paths: list[str]) -> list[Path]:
    existing = []
    for raw_path in include_paths:
        candidate = root / raw_path
        if candidate.exists():
            existing.append(candidate)
    return existing


def build_manifest(bundle_version: str, included_paths: list[Path]) -> dict:
    return {
        "schema_version": 1,
        "bundle_version": bundle_version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "included_paths": [path.as_posix() for path in included_paths],
        "notes": (
            "This bundle is intended to ship trained adapters, training metadata, and "
            "runtime config separately from the FounderAI API image."
        ),
    }


def add_to_tar(tar: tarfile.TarFile, root: Path, path: Path) -> None:
    tar.add(path, arcname=path.relative_to(root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a versioned FounderAI model bundle.")
    parser.add_argument("--output-dir", default="dist/model-bundles", help="Output directory for bundle files.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional relative paths to include in the bundle. Can be passed multiple times.",
    )
    parser.add_argument("--version", default=None, help="Explicit bundle version. Defaults to timestamp-sha.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_paths = DEFAULT_INCLUDE_PATHS + args.include
    existing_paths = find_existing_paths(root, include_paths)
    if not existing_paths:
        raise SystemExit("No model or training artifact paths were found to bundle.")

    bundle_version = resolve_version(args.version)
    manifest = build_manifest(bundle_version, existing_paths)

    manifest_path = output_dir / f"founderai-model-bundle-{bundle_version}.manifest.json"
    archive_path = output_dir / f"founderai-model-bundle-{bundle_version}.tar.gz"

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with tarfile.open(archive_path, "w:gz") as tar:
        add_to_tar(tar, root, manifest_path)
        for path in existing_paths:
            add_to_tar(tar, root, path)

    print(f"Bundle manifest: {manifest_path}")
    print(f"Bundle archive:  {archive_path}")


if __name__ == "__main__":
    main()
