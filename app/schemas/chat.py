from pydantic import BaseModel, Field

from app.schemas.common import ProjectRef, SourceChunk


class FieldStatus(BaseModel):
    field_name: str
    label: str
    is_filled: bool
    content: str | None = None


class ModuleContext(BaseModel):
    module_key: str
    label: str
    filled_fields: list[FieldStatus] = Field(default_factory=list)
    empty_fields: list[str] = Field(default_factory=list)
    raw_content: str | None = None


class ChatMessageItem(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str


class FieldProposal(BaseModel):
    field_name: str
    label: str
    value: str


class ChatRequest(BaseModel):
    project: ProjectRef = Field(default_factory=ProjectRef)
    module: ModuleContext
    message: str = Field(min_length=1)
    locale: str = Field(default="fr")
    conversation_history: list[ChatMessageItem] = Field(default_factory=list)


class ChatAction(BaseModel):
    type: str = Field(..., description="apply_fields, quick_prompt, or navigate")
    label: str
    target_field: str | None = None
    payload: str | dict | None = None
    field_proposals: list[FieldProposal] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Conversational response text")
    actions: list[ChatAction] = Field(default_factory=list)
    supporting_context: list[SourceChunk] = Field(default_factory=list)
    module_key: str
