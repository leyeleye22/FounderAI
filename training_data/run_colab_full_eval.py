from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


@dataclass
class ColabEvalConfig:
    repo_root: Path = field(default_factory=_repo_root)
    output_dir: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_OUTPUT_DIR", Path("/content/founderai-colab-train-eval/colab_outputs/lora_adapter")))
    summary_json_path: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_EVAL_SUMMARY_JSON", Path("/content/founderai-colab-train-eval/colab_outputs/lora_adapter/behavioral_eval_summary.json")))
    summary_md_path: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_EVAL_SUMMARY_MD", Path("/content/founderai-colab-train-eval/colab_outputs/lora_adapter/behavioral_eval_summary.md")))
    training_metrics_path: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_METRICS_PATH", Path("/content/founderai-colab-train-eval/colab_outputs/lora_adapter/training_metrics.json")))
    base_model_id: str = field(default_factory=lambda: os.getenv("FOUNDER_AI_COLAB_BASE_MODEL", "Qwen/Qwen3-4B"))
    adapter_path: Path = field(default_factory=lambda: _env_path("FOUNDER_AI_COLAB_OUTPUT_DIR", Path("/content/founderai-colab-train-eval/colab_outputs/lora_adapter")))


def _run_script(script_path: Path, *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# FounderAI Colab Behavioral Eval",
        "",
        "## Overview",
        "",
        f"- Adapter path: `{payload['adapter_path']}`",
        f"- Base model: `{payload['base_model_id']}`",
    ]

    training = payload.get("training_metrics")
    if training:
        lines.extend(
            [
                f"- Train loss: `{training.get('train_loss')}`",
                f"- Validation loss: `{training.get('validation_loss')}`",
                f"- Test loss: `{training.get('test_loss')}`",
                f"- Validation perplexity: `{training.get('validation_perplexity')}`",
                f"- Test perplexity: `{training.get('test_perplexity')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Behavioral summaries",
            "",
            f"- Problem-statement eval: `{payload['problem_statement_eval']['passed']}/{payload['problem_statement_eval']['total']}` passed",
            f"- Multi-module eval: `{payload['conversational_eval']['passed']}/{payload['conversational_eval']['total']}` passed",
            "",
            "## By module",
            "",
            "| Module | Total | Passed | Failed |",
            "| --- | --- | --- | --- |",
        ]
    )

    for module_key, stats in sorted(payload["conversational_eval"]["by_module"].items()):
        lines.append(f"| {module_key} | {stats['total']} | {stats['passed']} | {stats['failed']} |")

    failures = payload.get("top_failures", [])
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("- No behavioral failures detected in the scripted eval set.")
    else:
        for item in failures:
            lines.extend(
                [
                    f"### {item['case_id']}",
                    "",
                    f"- Module: `{item['module_key']}`",
                    f"- Missing expected: `{', '.join(item.get('missing_expected', [])) or '-'}`",
                    f"- Forbidden hits: `{', '.join(item.get('forbidden_hits', [])) or '-'}`",
                    f"- Reply preview: `{item['reply'][:280].replace(chr(10), ' ')}`",
                    "",
                ]
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = ColabEvalConfig()
    config.summary_json_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["USE_FINETUNED_MODEL"] = "true"
    env["FINETUNED_MODEL_PATH"] = config.base_model_id
    env["LORA_ADAPTER_PATH"] = str(config.adapter_path)
    env["FOUNDER_AI_FORCE_IN_MEMORY_RETRIEVAL"] = env.get("FOUNDER_AI_FORCE_IN_MEMORY_RETRIEVAL", "true")
    env["LLM_MAX_TOKENS"] = env.get("LLM_MAX_TOKENS", "350")
    env["LLM_CHAT_MAX_TOKENS"] = env.get("LLM_CHAT_MAX_TOKENS", "350")
    env["LLM_REASONING_MAX_TOKENS"] = env.get("LLM_REASONING_MAX_TOKENS", "500")

    scripts = [
        config.repo_root / "scripts" / "run_problem_statement_eval.py",
        config.repo_root / "scripts" / "run_conversational_module_eval.py",
    ]

    logs: dict[str, str] = {}
    for script_path in scripts:
        result = _run_script(script_path, env=env, cwd=config.repo_root)
        logs[script_path.name] = result.stdout
        if result.returncode != 0:
            raise RuntimeError(f"{script_path.name} failed with exit code {result.returncode}\n\n{result.stdout}")

    problem_eval = _load_json(config.repo_root / "evaluation" / "latest_problem_statement_eval.json")
    conversational_eval = _load_json(config.repo_root / "evaluation" / "latest_conversational_module_eval.json")
    training_metrics = _load_json(config.training_metrics_path) if config.training_metrics_path.exists() else None

    problem_summary = {
        "total": len(problem_eval),
        "passed": sum(1 for item in problem_eval if item.get("passed")),
        "failed": sum(1 for item in problem_eval if not item.get("passed")),
    }
    conv_summary = conversational_eval["summary"]
    conv_results = conversational_eval["results"]
    top_failures = [item for item in conv_results if not item.get("passed")][:5]

    payload = {
        "adapter_path": str(config.adapter_path),
        "base_model_id": config.base_model_id,
        "training_metrics": training_metrics,
        "problem_statement_eval": problem_summary,
        "conversational_eval": {
            "total": conv_summary["total"],
            "passed": conv_summary["passed"],
            "failed": conv_summary["failed"],
            "by_module": conv_summary["by_module"],
        },
        "top_failures": top_failures,
        "script_logs": logs,
    }

    config.summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_markdown(config.summary_md_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

