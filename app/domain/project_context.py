from pydantic import BaseModel, Field


class ModuleSnapshot(BaseModel):
    module: str
    title: str
    summary: str = ""
    raw: dict[str, object] = Field(default_factory=dict)


class ProjectSnapshot(BaseModel):
    workspace_id: str | None = None
    project_id: str | None = None
    project_name: str | None = None
    locale: str = "fr"
    modules: list[ModuleSnapshot] = Field(default_factory=list)

    def module(self, module_key: str) -> ModuleSnapshot | None:
        for item in self.modules:
            if item.module == module_key:
                return item
        return None


class InterviewSnapshot(BaseModel):
    interview_id: str
    interview_type: str | None = None
    status: str | None = None
    contact_name: str | None = None
    contact_role: str | None = None
    contact_company: str | None = None
    research_objectives: str | None = None
    hypotheses: str | None = None
    transcription: str | None = None
    key_evidence: str | None = None
    notes: str | None = None
    result_signal: str | None = None
    willingness_score: float | None = None
    next_steps: str | None = None
