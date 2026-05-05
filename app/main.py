from fastapi import FastAPI

from app.api.router import api_router
from app.core.settings import get_settings


settings = get_settings()

app = FastAPI(
    title="FounderAI",
    version="0.1.0",
    description="Local-first agentic AI copilot for FounderPath.",
)

app.include_router(api_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "FounderAI",
        "environment": settings.environment,
        "status": "ok",
    }

