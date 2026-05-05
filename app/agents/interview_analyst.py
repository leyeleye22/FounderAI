from app.agents.base import BaseAgent
from app.domain.project_context import InterviewSnapshot, ProjectSnapshot
from app.schemas.common import SourceChunk
from app.schemas.interview import InterviewAnalysisRequest, InterviewAnalysisResponse
from app.tools.interview_reader import InterviewReaderTool
from app.tools.interview_rules import InterviewRulesTool
from app.tools.project_snapshot import ProjectSnapshotTool


class InterviewAnalystAgent(BaseAgent):
    agent_name = "interview_analyst"

    def __init__(
        self,
        *,
        project_snapshot_tool: ProjectSnapshotTool,
        interview_reader_tool: InterviewReaderTool,
        interview_rules_tool: InterviewRulesTool,
    ) -> None:
        self.project_snapshot_tool = project_snapshot_tool
        self.interview_reader_tool = interview_reader_tool
        self.interview_rules_tool = interview_rules_tool

    def run(self, payload: InterviewAnalysisRequest) -> InterviewAnalysisResponse:
        snapshot = self.project_snapshot_tool.run(
            workspace_id=payload.project.workspace_id,
            project_id=payload.project.project_id,
        )
        interview = self.interview_reader_tool.run(
            workspace_id=snapshot.workspace_id or payload.project.workspace_id,
            project_id=snapshot.project_id or payload.project.project_id,
            interview_id=payload.interview_id,
        )
        query = self._build_query(snapshot, interview)
        rag_sources = self.interview_rules_tool.run(query=query)

        if interview is None:
            return InterviewAnalysisResponse(
                interview_id=payload.interview_id or "unknown",
                summary="Aucune interview exploitable n'a ete trouvee pour cette demande.",
                top_insights=[],
                objections=[],
                pains=[],
                evidence=[],
                buying_signals=[],
                risks=["L'assistant ne peut pas analyser une interview absente ou vide."],
                recommended_updates=["Verifie l'identifiant d'interview ou enregistre plus de contenu."],
                next_actions=["Ajoute une interview completee avant de lancer l'analyse."],
                supporting_context=self._project_sources(snapshot) + rag_sources,
            )

        top_insights = self._extract_insights(interview)
        objections = self._extract_objections(interview)
        pains = self._extract_pains(interview)
        evidence = self._extract_evidence(interview)
        buying_signals = self._extract_buying_signals(interview)
        risks = self._extract_risks(interview)
        recommended_updates = self._recommended_updates(snapshot, interview, pains, buying_signals)
        next_actions = self._next_actions(interview, pains, buying_signals)

        return InterviewAnalysisResponse(
            interview_id=interview.interview_id,
            summary=self._summary(interview, top_insights),
            top_insights=top_insights,
            objections=objections,
            pains=pains,
            evidence=evidence,
            buying_signals=buying_signals,
            risks=risks,
            recommended_updates=recommended_updates,
            next_actions=next_actions,
            supporting_context=self._project_sources(snapshot) + self._interview_sources(interview) + rag_sources,
        )

    def _build_query(self, snapshot: ProjectSnapshot, interview: InterviewSnapshot | None) -> str:
        parts: list[str] = []
        problem_module = snapshot.module("problem_statement")
        if problem_module and problem_module.summary:
            parts.append(problem_module.summary)
        if interview:
            parts.extend(
                filter(
                    None,
                    [
                        interview.research_objectives,
                        interview.hypotheses,
                        interview.notes,
                        interview.key_evidence,
                    ],
                )
            )
        return " ".join(parts).strip() or "analyze interview evidence and objections"

    def _summary(self, interview: InterviewSnapshot, top_insights: list[str]) -> str:
        who = interview.contact_name or "Ce contact"
        role = f" ({interview.contact_role})" if interview.contact_role else ""
        if top_insights:
            return f"{who}{role} confirme surtout: {top_insights[0]}"
        return f"{who}{role} a besoin d'une lecture plus complete de ses reponses pour degager un signal fort."

    def _extract_insights(self, interview: InterviewSnapshot) -> list[str]:
        insights: list[str] = []
        if interview.key_evidence:
            insights.append("Une preuve explicite a deja ete capturee dans l'interview.")
        if interview.result_signal in {"strong_interest", "interest"}:
            insights.append("Le niveau d'interet indique qu'il existe un signal commercial a explorer.")
        if interview.willingness_score and interview.willingness_score >= 7:
            insights.append("Le score de volonte montre une ouverture reelle a un futur achat ou test.")
        if interview.notes:
            insights.append("Les notes d'entretien peuvent nourrir directement le probleme, le BMC et le GTM.")
        return insights[:3]

    def _extract_objections(self, interview: InterviewSnapshot) -> list[str]:
        objections: list[str] = []
        notes = (interview.notes or "").lower()
        if "cher" in notes or "expensive" in notes or "prix" in notes:
            objections.append("Le prix semble etre une objection potentielle.")
        if "temps" in notes or "time" in notes:
            objections.append("Le temps d'adoption ou de mise en place peut bloquer.")
        if "pas sur" in notes or "not sure" in notes or "doute" in notes:
            objections.append("Le contact montre encore du doute ou un manque de confiance.")
        return objections

    def _extract_pains(self, interview: InterviewSnapshot) -> list[str]:
        pains: list[str] = []
        combined = " ".join(filter(None, [interview.transcription, interview.notes, interview.key_evidence])).lower()
        if "perd" in combined or "lose" in combined:
            pains.append("Perte de temps, d'energie ou d'argent mentionnee.")
        if "frustr" in combined:
            pains.append("Frustration explicite detectee.")
        if "manuel" in combined or "manual" in combined:
            pains.append("Le processus actuel reste trop manuel.")
        return pains

    def _extract_evidence(self, interview: InterviewSnapshot) -> list[str]:
        evidence: list[str] = []
        if interview.key_evidence:
            evidence.append(interview.key_evidence)
        if interview.transcription:
            evidence.append("Une transcription existe et peut etre resumee ou citee.")
        if interview.notes:
            evidence.append("Les notes du fondateur contiennent des signaux complementaires.")
        return evidence[:3]

    def _extract_buying_signals(self, interview: InterviewSnapshot) -> list[str]:
        signals: list[str] = []
        if interview.result_signal == "strong_interest":
            signals.append("Signal fort d'interet.")
        elif interview.result_signal == "interest":
            signals.append("Signal positif d'interet.")
        if interview.willingness_score and interview.willingness_score >= 8:
            signals.append("Le score de volonte est eleve.")
        if interview.next_steps:
            signals.append("Des prochaines etapes sont deja notees, ce qui suggere une suite envisagee.")
        return signals

    def _extract_risks(self, interview: InterviewSnapshot) -> list[str]:
        risks: list[str] = []
        if not interview.transcription and not interview.notes and not interview.key_evidence:
            risks.append("Tres peu de matiere exploitable a ete capturee.")
        if interview.status != "completed":
            risks.append("L'interview n'est pas marquee comme terminee.")
        if not interview.research_objectives:
            risks.append("Les objectifs de recherche ne sont pas formalises.")
        return risks

    def _recommended_updates(
        self,
        snapshot: ProjectSnapshot,
        interview: InterviewSnapshot,
        pains: list[str],
        buying_signals: list[str],
    ) -> list[str]:
        updates: list[str] = []
        if pains:
            updates.append("Mettre a jour le bloc probleme avec une douleur plus concrete issue de cette interview.")
        if buying_signals:
            updates.append("Reporter les signaux d'achat dans GTM ou ROI pour mieux prioriser la suite.")
        if snapshot.module("business"):
            updates.append("Comparer ces apprentissages avec les blocs BMC deja remplis.")
        else:
            updates.append("Reinjecter ces apprentissages dans le futur BMC.")
        return updates[:3]

    def _next_actions(self, interview: InterviewSnapshot, pains: list[str], buying_signals: list[str]) -> list[str]:
        actions: list[str] = []
        if pains:
            actions.append("Extraire une citation concrete et l'ajouter comme preuve terrain.")
        if buying_signals:
            actions.append("Planifier une relance ou un test avec ce contact.")
        if interview.next_steps:
            actions.append(interview.next_steps)
        if not actions:
            actions.append("Completer les notes ou la transcription avant toute conclusion.")
        return actions[:3]

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

    def _interview_sources(self, interview: InterviewSnapshot) -> list[SourceChunk]:
        items: list[SourceChunk] = []
        if interview.notes:
            items.append(
                SourceChunk(
                    title="Interview notes",
                    excerpt=interview.notes[:280],
                    source_type="interview_notes",
                )
            )
        if interview.key_evidence:
            items.append(
                SourceChunk(
                    title="Interview evidence",
                    excerpt=interview.key_evidence[:280],
                    source_type="interview_evidence",
                )
            )
        return items
