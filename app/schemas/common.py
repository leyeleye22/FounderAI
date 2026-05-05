from pydantic import BaseModel, Field


class ProjectRef(BaseModel):
    workspace_id: str | None = Field(default=None)
    project_id: str | None = Field(default=None)


class SourceChunk(BaseModel):
    title: str
    excerpt: str
    source_type: str

