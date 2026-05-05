from app.agents.base import BaseAgent
from app.domain.project_context import ProjectSnapshot
from app.schemas.common import SourceChunk
from app.schemas.sprint import SprintPlanRequest, SprintPlanResponse, SprintTaskProposal
from app.tools.project_snapshot import ProjectSnapshotTool
from app.tools.sprint_rules import SprintRulesTool


class SprintPlannerAgent(BaseAgent):
    agent_name = "sprint_planner"

    def __init__(
        self,
        *,
        project_snapshot_tool: ProjectSnapshotTool,
        sprint_rules_tool: SprintRulesTool,
    ) -> None:
        self.project_snapshot_tool = project_snapshot_tool
        self.sprint_rules_tool = sprint_rules_tool

    def run(self, payload: SprintPlanRequest) -> SprintPlanResponse:
        snapshot = self.project_snapshot_tool.run(
            workspace_id=payload.project.workspace_id,
            project_id=payload.project.project_id,
        )
        query = self._build_query(snapshot)
        rag_sources = self.sprint_rules_tool.run(query=query)

        problem_module = snapshot.module("problem_statement")
        interview_module = snapshot.module("interview")
        problem_summary = problem_module.summary if problem_module and problem_module.summary else "probleme encore peu defini"
        interview_summary = (
            interview_module.summary if interview_module and interview_module.summary else "pas encore d'interviews completes"
        )

        tasks = self._build_tasks(problem_summary, interview_summary)
        goal = self._build_goal(problem_summary)
        reasoning = [
            "Le sprint part du probleme actuel pour eviter de construire trop large.",
            "Les interviews orientent les taches vers la preuve terrain et les signaux d'achat.",
            "Le plan reste volontairement court pour maximiser la valeur dans le prochain cycle.",
        ]

        return SprintPlanResponse(
            name=payload.sprint_name or "Sprint 1",
            goal=goal,
            duration=payload.duration,
            status="planned",
            tasks=tasks,
            review="Verifier ce qui a ete appris sur le probleme, les objections et le signal d'achat.",
            retrospective="Identifier ce qui a bloque l'equipe et ce qu'il faut simplifier au prochain sprint.",
            reasoning=reasoning,
            supporting_context=self._project_sources(snapshot) + rag_sources,
        )

    def _build_query(self, snapshot: ProjectSnapshot) -> str:
        parts = []
        problem_module = snapshot.module("problem_statement")
        interview_module = snapshot.module("interview")
        if problem_module and problem_module.summary:
            parts.append(problem_module.summary)
        if interview_module and interview_module.summary:
            parts.append(interview_module.summary)
        return " ".join(parts) or "build a founder sprint from problem validation and interviews"

    def _build_goal(self, problem_summary: str) -> str:
        return f"Clarifier et valider une avancee concrete autour de: {problem_summary}"

    def _build_tasks(self, problem_summary: str, interview_summary: str) -> list[SprintTaskProposal]:
        return [
            SprintTaskProposal(
                title="Reserrer la formulation du probleme",
                reason=f"Le sprint doit partir d'un probleme clair: {problem_summary}",
            ),
            SprintTaskProposal(
                title="Extraire 3 preuves terrain utilisables",
                reason="Sans preuves, le reste du parcours repose encore sur des suppositions.",
            ),
            SprintTaskProposal(
                title="Analyser les objections recurrentes des interviews",
                reason=f"Le contexte interview actuel est: {interview_summary}",
            ),
            SprintTaskProposal(
                title="Mettre a jour BMC ou GTM avec les nouveaux apprentissages",
                reason="Les apprentissages terrain doivent etre reinjectes rapidement dans le projet.",
            ),
        ]

    def _project_sources(self, snapshot: ProjectSnapshot) -> list[SourceChunk]:
        items: list[SourceChunk] = []
        for module in snapshot.modules[:3]:
            if module.summary:
                items.append(
                    SourceChunk(
                        title=module.title,
                        excerpt=module.summary,
                        source_type="project_context",
                    )
                )
        return items
