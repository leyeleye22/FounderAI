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
from app.schemas.chat import ChatRequest, ModuleContext
from app.services.retrieval.in_memory import InMemoryRetriever

CASES_PATH = ROOT / "evaluation" / "problem_statement_cases.json"
OUTPUT_JSON = ROOT / "evaluation" / "latest_problem_statement_eval.json"
OUTPUT_MD = ROOT / "evaluation" / "latest_problem_statement_eval.md"


class FakeSnapshotTool:
    def run(self, *, workspace_id: str | None, project_id: str | None) -> ProjectSnapshot:
        return ProjectSnapshot(workspace_id=workspace_id, project_id=project_id, project_name="Local eval project", modules=[])


@dataclass
class EvalResult:
    case_id: str
    locale: str
    message: str
    reply: str
    action_types: list[str]
    field_proposals: list[dict[str, str]]
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
            module_key="problem-statement",
            label="Problem Statement",
            filled_fields=[],
            empty_fields=["problemStatement", "who", "when", "howOften", "currentWorkaround", "cost"],
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

    field_proposals: list[dict[str, str]] = []
    for action in response.actions:
        for proposal in action.field_proposals:
            field_proposals.append(
                {
                    "field_name": proposal.field_name,
                    "label": proposal.label,
                    "value": proposal.value,
                }
            )

    return EvalResult(
        case_id=str(case["id"]),
        locale=request.locale,
        message=request.message,
        reply=response.reply,
        action_types=[action.type for action in response.actions],
        field_proposals=field_proposals,
        supporting_context_count=len(response.supporting_context),
        expected_hits=expected_hits,
        missing_expected=missing_expected,
        forbidden_hits=forbidden_hits,
        passed=(not missing_expected and not forbidden_hits),
    )


def write_markdown(results: list[EvalResult]) -> None:
    passed_count = sum(1 for item in results if item.passed)
    lines = [
        "# Problem Statement Eval",
        "",
        f"- Total cases: {len(results)}",
        f"- Passed: {passed_count}",
        f"- Failed: {len(results) - passed_count}",
        "",
        "| Case | Pass | Actions | Missing expected | Forbidden hits |",
        "| --- | --- | --- | --- | --- |",
    ]

    for result in results:
        actions = ", ".join(result.action_types) if result.action_types else "-"
        missing = ", ".join(result.missing_expected) if result.missing_expected else "-"
        forbidden = ", ".join(result.forbidden_hits) if result.forbidden_hits else "-"
        lines.append(f"| {result.case_id} | {'yes' if result.passed else 'no'} | {actions} | {missing} | {forbidden} |")

    for result in results:
        lines.extend(
            [
                "",
                f"## {result.case_id}",
                "",
                f"**Message**: {result.message}",
                "",
                f"**Reply**: {result.reply}",
                "",
                f"**Actions**: {', '.join(result.action_types) if result.action_types else '-'}",
                "",
                f"**Field proposals**: {json.dumps(result.field_proposals, ensure_ascii=False)}",
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
    OUTPUT_JSON.write_text(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(results)
    print(json.dumps({"total": len(results), "passed": sum(1 for item in results if item.passed), "failed": sum(1 for item in results if not item.passed)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
