import re

from huggingface_hub import InferenceClient

from app.core.settings import get_settings
from app.services.llm.base import BaseLLMService, LLMResponse


class HuggingFaceInferenceService(BaseLLMService):
    provider_name = "huggingface-inference"

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.hf_inference_model or settings.llm_model_name
        self._provider = settings.hf_inference_provider
        self._token = settings.hf_token
        self._timeout = settings.llm_request_timeout
        self._presence_penalty = settings.llm_presence_penalty
        self._chat_temperature = settings.llm_chat_temperature
        self._chat_top_p = settings.llm_chat_top_p
        self._chat_max_tokens = settings.llm_chat_max_tokens
        self._reasoning_temperature = settings.llm_reasoning_temperature
        self._reasoning_top_p = settings.llm_reasoning_top_p
        self._reasoning_max_tokens = settings.llm_reasoning_max_tokens

        self._client = InferenceClient(
            model=self._model_name,
            provider=self._provider,
            token=self._token,
            timeout=self._timeout,
        )

    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        mode = self._select_mode(system_prompt, user_prompt)
        mode_tag = "/think" if mode["thinking"] else "/no_think"
        system_message = f"{system_prompt.strip()}\n\n{mode_tag}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=mode["max_tokens"],
                temperature=mode["temperature"],
                top_p=mode["top_p"],
                presence_penalty=self._presence_penalty,
            )
            content = response.choices[0].message.content if response.choices else ""
            cleaned = self._clean_response(content)
            if cleaned:
                return LLMResponse(
                    content=cleaned,
                    provider=self.provider_name,
                    model=self._model_name,
                )
        except Exception:
            pass

        from app.services.llm.local_qwen import LocalQwenService

        fallback = LocalQwenService()._local_heuristic(system_prompt, user_prompt)
        return LLMResponse(
            content=fallback.content,
            provider=self.provider_name,
            model=self._model_name,
        )

    def _select_mode(self, system_prompt: str, user_prompt: str) -> dict[str, float | int | bool]:
        system_lower = system_prompt.lower()
        user_lower = user_prompt.lower()
        merged = f"{system_lower} {user_lower}"

        fast_keywords = [
            "reformule",
            "reecris",
            "rends-le plus clair",
            "rends le plus clair",
            "plus clair",
            "meilleure version",
            "trouve ce qui est flou",
            "qu'est-ce qui manque",
            "qu est-ce qui manque",
            "aide-moi a ecrire",
            "aide moi a ecrire",
            "clarify",
            "rewrite",
        ]
        reasoning_keywords = [
            "analyse",
            "analyze",
            "challenge",
            "challenger",
            "compare",
            "comparer",
            "priorise",
            "prioritize",
            "plan",
            "roadmap",
            "sprint",
            "roi",
            "tam",
            "sam",
            "som",
            "business model",
            "bmc",
            "interview",
            "objection",
            "preuve",
            "evidence",
            "hypothese",
            "hypothesis",
            "scoring",
            "diagnostic",
            "market sizing",
            "go-to-market",
        ]
        heavy_modules = [
            "page validation",
            "page go-to-market",
            "page business model canvas",
            "page roi",
            "page parcours client",
            "page sprints",
            "page client ideal",
        ]

        wants_fast = any(keyword in user_lower for keyword in fast_keywords)
        wants_reasoning = any(keyword in merged for keyword in reasoning_keywords)
        in_heavy_module = any(keyword in system_lower for keyword in heavy_modules)

        thinking = wants_reasoning or (in_heavy_module and not wants_fast)

        if thinking:
            return {
                "thinking": True,
                "temperature": self._reasoning_temperature,
                "top_p": self._reasoning_top_p,
                "max_tokens": self._reasoning_max_tokens,
            }

        return {
            "thinking": False,
            "temperature": self._chat_temperature,
            "top_p": self._chat_top_p,
            "max_tokens": self._chat_max_tokens,
        }

    def _clean_response(self, content: str) -> str:
        if not content:
            return ""

        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>")[-1]

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned
