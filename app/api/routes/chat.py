from fastapi import APIRouter

from app.registry import get_conversational_agent
from app.schemas.chat import ChatRequest, ChatResponse


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    agent = get_conversational_agent()
    return agent.run(payload)
