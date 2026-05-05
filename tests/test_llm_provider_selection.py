from app.core.settings import get_settings
from app.services.llm.base import create_llm_service
from app.services.llm.huggingface_inference import HuggingFaceInferenceService
from app.services.llm.local_qwen import LocalQwenService


def test_create_llm_service_uses_huggingface_when_requested(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_INFERENCE_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.delenv("USE_FINETUNED_MODEL", raising=False)
    get_settings.cache_clear()

    service = create_llm_service()

    assert isinstance(service, HuggingFaceInferenceService)
    get_settings.cache_clear()


def test_create_llm_service_defaults_to_local_qwen(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("HF_INFERENCE_MODEL", raising=False)
    monkeypatch.setenv("USE_FINETUNED_MODEL", "false")
    get_settings.cache_clear()

    service = create_llm_service()

    assert isinstance(service, LocalQwenService)
    get_settings.cache_clear()
