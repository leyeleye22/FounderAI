from fastapi import APIRouter

from app.api.routes.agents import router as agents_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router


api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(agents_router, prefix="/agents", tags=["agents"])
api_router.include_router(chat_router, prefix="/agents", tags=["chat"])

