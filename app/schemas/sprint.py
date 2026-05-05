from pydantic import BaseModel, Field

from app.schemas.common import ProjectRef, SourceChunk


class SprintTaskProposal(BaseModel):
    title: str
    reason: str
    status: str = "To Do"


class SprintPlanRequest(BaseModel):
    project: ProjectRef = Field(default_factory=ProjectRef)
    locale: str = "fr"
    duration: str = "7 days"
    sprint_name: str | None = None


class SprintPlanResponse(BaseModel):
    name: str
    goal: str
    duration: str
    status: str
    tasks: list[SprintTaskProposal]
    review: str
    retrospective: str
    reasoning: list[str]
    supporting_context: list[SourceChunk] = Field(default_factory=list)
