from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = Field(default="development", alias="FOUNDER_AI_ENV")
    host: str = Field(default="0.0.0.0", alias="FOUNDER_AI_HOST")
    port: int = Field(default=8010, alias="FOUNDER_AI_PORT")
    model_dir: Path = Field(default=Path(".").resolve(), alias="FOUNDER_AI_MODEL_DIR")
    default_locale: str = Field(default="fr", alias="FOUNDER_AI_DEFAULT_LOCALE")
    founderpath_api_base_url: str = Field(default="http://127.0.0.1:8000/api", alias="FOUNDERPATH_API_BASE_URL")
    founderpath_api_token: str | None = Field(default=None, alias="FOUNDERPATH_API_TOKEN")
    use_local_heuristics: bool = Field(default=True, alias="FOUNDER_AI_USE_LOCAL_HEURISTICS")
    founderpath_request_timeout_seconds: float = Field(default=20.0, alias="FOUNDERPATH_REQUEST_TIMEOUT_SECONDS")
    rag_dir: Path = Field(default=Path(".rag").resolve(), alias="FOUNDER_AI_RAG_DIR")
    rag_collection: str = Field(default="founderai_knowledge", alias="FOUNDER_AI_RAG_COLLECTION")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="FOUNDER_AI_EMBEDDING_MODEL")
    force_in_memory_retrieval: bool = Field(default=False, alias="FOUNDER_AI_FORCE_IN_MEMORY_RETRIEVAL")

    # LLM settings
    llm_api_base_url: str | None = Field(default=None, alias="LLM_API_BASE_URL")
    llm_model_name: str = Field(default="qwen3-4b-fp8", alias="LLM_MODEL_NAME")
    llm_request_timeout: float = Field(default=60.0, alias="LLM_REQUEST_TIMEOUT")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_presence_penalty: float = Field(default=0.2, alias="LLM_PRESENCE_PENALTY")
    llm_chat_temperature: float = Field(default=0.7, alias="LLM_CHAT_TEMPERATURE")
    llm_chat_top_p: float = Field(default=0.8, alias="LLM_CHAT_TOP_P")
    llm_chat_max_tokens: int = Field(default=700, alias="LLM_CHAT_MAX_TOKENS")
    llm_reasoning_temperature: float = Field(default=0.6, alias="LLM_REASONING_TEMPERATURE")
    llm_reasoning_top_p: float = Field(default=0.95, alias="LLM_REASONING_TOP_P")
    llm_reasoning_max_tokens: int = Field(default=1400, alias="LLM_REASONING_MAX_TOKENS")

    # Fine-tuned model settings
    finetuned_model_path: str | None = Field(default=None, alias="FINETUNED_MODEL_PATH")
    lora_adapter_path: str | None = Field(default=None, alias="LORA_ADAPTER_PATH")
    use_finetuned_model: bool = Field(default=False, alias="USE_FINETUNED_MODEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
