from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.conversational import ConversationalAgent
from app.domain.project_context import ProjectSnapshot
from app.schemas.chat import ChatRequest, FieldStatus, ModuleContext
from app.services.retrieval.in_memory import InMemoryRetriever

CASES_PATH = ROOT / "evaluation" / "conversational_module_cases.json"
OUTPUT_JSON = ROOT / "evaluation" / "latest_conversational_module_eval.json"
OUTPUT_MD = ROOT / "evaluation" / "latest_conversational_module_eval.md"


class FakeSnapshotTool:
    def run(self, *, workspace_id: str | None, project_id: str | None) -> ProjectSnapshot:
        return ProjectSnapshot(workspace_id=workspace_id, project_id=project_id, project_name="Local eval project", modules=[])


@dataclass
class EvalResult:
    case_id: str
    module_key: str
    locale: str
    message: str
    reply: str
    action_types: list[str]
    supporting_context_count: int
    expected_hits: list[str]
    missing_expected: list[str]
    forbidden_hits: list[str]
    passed: bool


def build_agent() -> ConversationalAgent:
    return ConversationalAgent(
        project_snapshot_tool=FakeSnapshotTool(),
        retriever=InMemoryRetriever(),
    )


def run_case(agent: ConversationalAgent, case: dict[str, object]) -> EvalResult:
    request = ChatRequest(
        module=ModuleContext(
            module_key=str(case["module_key"]),
            label=str(case.get("label", case["module_key"])),
            filled_fields=[FieldStatus(**field) for field in case.get("filled_fields", [])],
            empty_fields=[str(item) for item in case.get("empty_fields", [])],
        ),
        message=str(case["message"]),
        locale=str(case.get("locale", "fr")),
        conversation_history=[],
    )

    response = agent.run(request)
    reply_lower = response.reply.lower()
    expected = [item.lower() for item in case.get("expected_signals", [])]
    forbidden = [item.lower() for item in case.get("forbidden_signals", [])]

    expected_hits = [item for item in expected if item in reply_lower]
    missing_expected = [item for item in expected if item not in reply_lower]
    forbidden_hits = [item for item in forbidden if item in reply_lower]

    return EvalResult(
        case_id=str(case["id"]),
        module_key=request.module.module_key,
        locale=request.locale,
        message=request.message,
        reply=response.reply,
        action_types=[action.type for action in response.actions],
        supporting_context_count=len(response.supporting_context),
        expected_hits=expected_hits,
        missing_expected=missing_expected,
        forbidden_hits=forbidden_hits,
        passed=(not missing_expected and not forbidden_hits),
    )


def build_module_summary(results: list[EvalResult]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = summary.setdefault(result.module_key, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if result.passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    return summary


def write_markdown(results: list[EvalResult]) -> None:
    module_summary = build_module_summary(results)
    passed_count = sum(1 for item in results if item.passed)
    lines = [
        "# Conversational Module Eval",
        "",
        f"- Total cases: {len(results)}",
        f"- Passed: {passed_count}",
        f"- Failed: {len(results) - passed_count}",
        "",
        "## By Module",
        "",
        "| Module | Total | Passed | Failed |",
        "| --- | --- | --- | --- |",
    ]

    for module_key, stats in sorted(module_summary.items()):
        lines.append(f"| {module_key} | {stats['total']} | {stats['passed']} | {stats['failed']} |")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Module | Pass | Actions | Missing expected | Forbidden hits |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for result in results:
        actions = ", ".join(result.action_types) if result.action_types else "-"
        missing = ", ".join(result.missing_expected) if result.missing_expected else "-"
        forbidden = ", ".join(result.forbidden_hits) if result.forbidden_hits else "-"
        lines.append(
            f"| {result.case_id} | {result.module_key} | {'yes' if result.passed else 'no'} | {actions} | {missing} | {forbidden} |"
        )

    for result in results:
        lines.extend(
            [
                "",
                f"## {result.case_id}",
                "",
                f"**Module**: {result.module_key}",
                "",
                f"**Message**: {result.message}",
                "",
                f"**Reply**: {result.reply}",
                "",
                f"**Actions**: {', '.join(result.action_types) if result.action_types else '-'}",
                "",
                f"**Expected hits**: {', '.join(result.expected_hits) if result.expected_hits else '-'}",
                "",
                f"**Missing expected**: {', '.join(result.missing_expected) if result.missing_expected else '-'}",
                "",
                f"**Forbidden hits**: {', '.join(result.forbidden_hits) if result.forbidden_hits else '-'}",
            ]
        )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    agent = build_agent()
    results = [run_case(agent, case) for case in cases]
    summary = build_module_summary(results)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "summary": {
                    "total": len(results),
                    "passed": sum(1 for item in results if item.passed),
                    "failed": sum(1 for item in results if not item.passed),
                    "by_module": summary,
                },
                "results": [asdict(item) for item in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(results)
    print(
        json.dumps(
            {
                "total": len(results),
                "passed": sum(1 for item in results if item.passed),
                "failed": sum(1 for item in results if not item.passed),
                "by_module": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
