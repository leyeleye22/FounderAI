from dataclasses import dataclass, field
from functools import lru_cache


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
    return _build_llm_service(
        use_finetuned_model=settings.use_finetuned_model,
        llm_provider=settings.llm_provider,
        llm_api_base_url=settings.llm_api_base_url or "",
        hf_inference_model=settings.hf_inference_model or "",
        finetuned_model_path=settings.finetuned_model_path or "",
        lora_adapter_path=settings.lora_adapter_path or "",
    )


@lru_cache(maxsize=4)
def _build_llm_service(
    *,
    use_finetuned_model: bool,
    llm_provider: str,
    llm_api_base_url: str,
    hf_inference_model: str,
    finetuned_model_path: str,
    lora_adapter_path: str,
) -> BaseLLMService:
    # Try fine-tuned model first if enabled
    if use_finetuned_model:
        from app.services.llm.finetuned_qwen import FinetunedQwenService

        service = FinetunedQwenService()
        if service.is_available():
            return service

    if llm_provider == "huggingface" or hf_inference_model:
        from app.services.llm.huggingface_inference import HuggingFaceInferenceService

        return HuggingFaceInferenceService()

    # Fall back to API-based Qwen
    from app.services.llm.local_qwen import LocalQwenService

    return LocalQwenService()
