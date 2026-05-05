from app.agents.base import BaseAgent
from app.domain.project_context import ProjectSnapshot
from app.schemas.common import SourceChunk
from app.schemas.problem import ProblemChallengeRequest, ProblemChallengeResponse
from app.tools.problem_rules import ProblemRulesTool
from app.tools.project_snapshot import ProjectSnapshotTool


class ProblemChallengerAgent(BaseAgent):
    agent_name = "problem_challenger"

    def __init__(
        self,
        *,
        project_snapshot_tool: ProjectSnapshotTool,
        problem_rules_tool: ProblemRulesTool,
    ) -> None:
        self.project_snapshot_tool = project_snapshot_tool
        self.problem_rules_tool = problem_rules_tool

    def run(self, payload: ProblemChallengeRequest) -> ProblemChallengeResponse:
        statement = payload.statement.strip()
        snapshot = self.project_snapshot_tool.run(
            workspace_id=payload.project.workspace_id,
            project_id=payload.project.project_id,
        )
        sources = self.problem_rules_tool.run(statement=statement)

        strengths: list[str] = []
        weaknesses: list[str] = []
        questions: list[str] = []

        score_clarity = 0.45
        score_specificity = 0.40

        problem_module = snapshot.module("problem_statement")
        interview_module = snapshot.module("interview")

        if len(statement.split()) >= 8:
            score_clarity += 0.15
        else:
            weaknesses.append("La phrase est trop courte pour decrire clairement la situation.")
            questions.append("Qui vit ce probleme exactement ?")

        lowered = statement.lower()

        if any(token in lowered for token in ["application", "app", "plateforme", "site", "outil"]):
            weaknesses.append("La formulation parle deja de solution au lieu de rester centree sur le probleme.")
            questions.append("Comment les gens decrivent-ils leur douleur sans parler de ta solution ?")
        else:
            strengths.append("La phrase n'est pas immediatement centree sur une solution.")
            score_clarity += 0.1

        if any(token in lowered for token in ["client", "clients", "femmes", "meres", "restaurateurs", "commercants", "entrepreneurs"]):
            strengths.append("Une cible commence a apparaitre dans la phrase.")
            score_specificity += 0.2
        else:
            weaknesses.append("La cible reste floue ou absente.")
            questions.append("Quel groupe precis souffre le plus de ce probleme ?")

        if any(token in lowered for token in ["car", "parce", "parce que", "faute de", "sans", "manque"]):
            strengths.append("La phrase donne deja une cause ou un contexte utile.")
            score_specificity += 0.15
        else:
            weaknesses.append("La cause ou le contexte du probleme ne sont pas encore clairs.")
            questions.append("Dans quel contexte exact ce probleme apparait-il ?")

        if problem_module and problem_module.summary:
            strengths.append("Le projet contient deja un bloc probleme enregistre que l'assistant peut comparer.")
            score_clarity += 0.05

        if interview_module and interview_module.raw.get("items"):
            strengths.append("Des interviews existent deja pour appuyer ou contredire cette formulation.")
            score_specificity += 0.05
            if "preuve" not in lowered and "citation" not in lowered:
                questions.append("Quelle citation ou preuve issue des interviews appuie ce probleme ?")

        if not questions:
            questions.append("Quelle preuve terrain montres-tu pour confirmer ce probleme ?")

        rewrite = None
        if payload.include_rewrite:
            rewrite = self._rewrite_statement(statement)

        next_action = (
            "Valide cette formulation avec trois personnes du terrain avant de passer a l'etape suivante."
            if weaknesses
            else "Ajoute une preuve concrete issue du terrain, puis continue vers la validation."
        )

        return ProblemChallengeResponse(
            score_clarity=min(score_clarity, 1.0),
            score_specificity=min(score_specificity, 1.0),
            strengths=strengths,
            weaknesses=weaknesses,
            challenge_questions=questions[:3],
            rewrite_suggestion=rewrite,
            next_action=next_action,
            supporting_context=sources + self._project_sources(snapshot),
        )

    def _rewrite_statement(self, statement: str) -> str:
        if statement.endswith("."):
            statement = statement[:-1]
        return (
            "Version plus claire: "
            f"{statement}. Precise maintenant qui souffre le plus, dans quel contexte, et avec quelle consequence concrete."
        )

    def _project_sources(self, snapshot: ProjectSnapshot) -> list[SourceChunk]:
        items: list[SourceChunk] = []
        for module in snapshot.modules[:2]:
            if module.summary:
                items.append(
                    SourceChunk(
                        title=module.title,
                        excerpt=module.summary,
                        source_type="project_context",
                    )
                )
        return items
