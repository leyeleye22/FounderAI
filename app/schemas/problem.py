from pydantic import BaseModel, Field

from app.schemas.common import ProjectRef, SourceChunk


class ProblemChallengeRequest(BaseModel):
    project: ProjectRef = Field(default_factory=ProjectRef)
    statement: str = Field(min_length=3)
    locale: str = Field(default="fr")
    include_rewrite: bool = Field(default=True)


class ProblemChallengeResponse(BaseModel):
    score_clarity: float
    score_specificity: float
    strengths: list[str]
    weaknesses: list[str]
    challenge_questions: list[str]
    rewrite_suggestion: str | None
    next_action: str
    supporting_context: list[SourceChunk] = Field(default_factory=list)

