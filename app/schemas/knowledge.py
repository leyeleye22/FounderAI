from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    document_id: str
    module: str
    field: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    language: str = "fr"
    source_type: str = "rule"


class KnowledgeChunk(BaseModel):
    chunk_id: str
    document_id: str
    module: str
    field: str
    title: str
    content: str
    language: str = "fr"
    source_type: str = "rule"
    tags: list[str] = Field(default_factory=list)
