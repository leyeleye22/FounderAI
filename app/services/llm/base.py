from dataclasses import dataclass, field


@dataclass(slots=True)
class LLMResponse:
    content: str
    provider: str
    model: str
    error: bool = False


class BaseLLMService:
    provider_name = "base"

    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        raise NotImplementedError


def create_llm_service() -> BaseLLMService:
    """Factory function to create the appropriate LLM service based on settings."""
    from app.core.settings import get_settings

    settings = get_settings()

    # Try fine-tuned model first if enabled
    if settings.use_finetuned_model:
        from app.services.llm.finetuned_qwen import FinetunedQwenService

        service = FinetunedQwenService()
        if service.is_available():
            return service

    # Fall back to API-based Qwen
    from app.services.llm.local_qwen import LocalQwenService

    return LocalQwenService()

