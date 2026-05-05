from pydantic import BaseModel, Field

from app.schemas.common import ProjectRef, SourceChunk


class InterviewAnalysisRequest(BaseModel):
    project: ProjectRef = Field(default_factory=ProjectRef)
    interview_id: str | None = None
    locale: str = "fr"


class InterviewAnalysisResponse(BaseModel):
    interview_id: str
    summary: str
    top_insights: list[str]
    objections: list[str]
    pains: list[str]
    evidence: list[str]
    buying_signals: list[str]
    risks: list[str]
    recommended_updates: list[str]
    next_actions: list[str]
    supporting_context: list[SourceChunk] = Field(default_factory=list)
