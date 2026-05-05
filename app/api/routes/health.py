from fastapi import APIRouter

from app.core.settings import get_settings


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.environment,
        "model_dir": str(settings.model_dir),
    }

