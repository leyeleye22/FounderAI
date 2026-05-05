import httpx
import os
import re

from app.core.settings import get_settings
from app.services.llm.base import BaseLLMService, LLMResponse


class LocalQwenService(BaseLLMService):
    provider_name = "local-qwen"

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.llm_api_base_url
        self._model_name = settings.llm_model_name
        self._timeout = settings.llm_request_timeout
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._presence_penalty = settings.llm_presence_penalty
        self._chat_temperature = settings.llm_chat_temperature
        self._chat_top_p = settings.llm_chat_top_p
        self._chat_max_tokens = settings.llm_chat_max_tokens
        self._reasoning_temperature = settings.llm_reasoning_temperature
        self._reasoning_top_p = settings.llm_reasoning_top_p
        self._reasoning_max_tokens = settings.llm_reasoning_max_tokens

    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self._base_url:
            return self._local_heuristic(system_prompt, user_prompt)

        mode = self._select_mode(system_prompt, user_prompt)
        mode_tag = "/think" if mode["thinking"] else "/no_think"
        system_message = f"{system_prompt.strip()}\n\n{mode_tag}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": mode["temperature"],
            "top_p": mode["top_p"],
            "max_tokens": mode["max_tokens"],
            "presence_penalty": self._presence_penalty,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                cleaned = self._clean_response(content)
                if cleaned:
                    return LLMResponse(
                        content=cleaned,
                        provider=self.provider_name,
                        model=self._model_name,
                    )
        except Exception:
            pass

        return self._local_heuristic(system_prompt, user_prompt)

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

    def _local_heuristic(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        msg_lower = user_prompt.lower()

        help_kw = ["aide", "help", "comment", "how", "explique", "explain", "guide", "commencer", "start"]
        challenge_kw = ["challenger", "challenge", "analyse", "analy", "faible", "weak", "flou", "vague", "coherent", "coher"]
        missing_kw = ["manque", "missing", "quoi remplir", "what to fill", "complet", "complete", "important"]
        rewrite_kw = ["reecris", "reformule", "meilleure version", "better version", "clearer", "plus clair"]
        next_kw = ["prochain", "next", "apres", "after", "etape", "step", "continuer"]

        is_help = any(kw in msg_lower for kw in help_kw)
        is_challenge = any(kw in msg_lower for kw in challenge_kw)
        is_missing = any(kw in msg_lower for kw in missing_kw)
        is_rewrite = any(kw in msg_lower for kw in rewrite_kw)
        is_next = any(kw in msg_lower for kw in next_kw)

        if is_rewrite:
            content = (
                "D'accord. Envoie-moi ta phrase actuelle et je te proposerai une version plus claire, plus concrete et plus facile a defendre."
            )
        elif is_help:
            content = (
                "On peut le construire ensemble. Commence par m'envoyer ta phrase actuelle, ou reponds juste a ceci: qui souffre, dans quel moment, et qu'est-ce que ca coute ?"
            )
        elif is_challenge:
            content = (
                "Je peux te challenger proprement. Envoie-moi ce que tu as ecrit et je te dirai ce qui est clair, ce qui bloque, puis je te proposerai une meilleure version."
            )
        elif is_missing:
            content = (
                "Pour avancer, j'ai surtout besoin de 3 choses: qui souffre exactement, dans quel moment concret le probleme apparait, et quel impact reel cela cree."
            )
        elif is_next:
            content = (
                "Avant d'aller plus loin, assure-toi d'avoir un probleme clair, une cible identifiable, et au moins un signal terrain qui montre que la douleur existe vraiment."
            )
        else:
            content = (
                "Je t'ecoute. Envoie-moi ce qui te bloque ou colle ton brouillon, et je t'aiderai a le clarifier ou a le reecrire."
            )

        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=self._model_name,
        )
