"""
Local Fine-Tuned Qwen3 Service with LoRA Adapter.
Uses the trained LoRA adapter for domain-specific inference.
Falls back to API or heuristics if model cannot be loaded.
"""

import os
import re
import torch
from typing import Optional

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.settings import get_settings
from app.services.llm.base import BaseLLMService, LLMResponse


class FinetunedQwenService(BaseLLMService):
    """Qwen3 model with LoRA adapter for Teranga Power domain knowledge."""

    provider_name = "finetuned-qwen"

    def __init__(self) -> None:
        settings = get_settings()
        self._base_model_path = settings.finetuned_model_path or settings.llm_model_name
        self._lora_adapter_path = settings.lora_adapter_path
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_chat_temperature
        self._top_p = settings.llm_chat_top_p

        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_error = None

        # Try to load model on initialization
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load the fine-tuned model."""
        try:
            if not self._base_model_path:
                self._load_error = "Base model path or model id not configured"
                return

            model_source = "local path" if os.path.exists(self._base_model_path) else "model id"
            print(f"Loading base model from {model_source}: {self._base_model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._base_model_path,
                trust_remote_code=True,
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token

            # Determine device
            torch_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
                if torch.cuda.is_available()
                else torch.float32
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self._base_model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

            # Load LoRA adapter if available
            if self._lora_adapter_path and os.path.exists(self._lora_adapter_path):
                print(f"Loading LoRA adapter from {self._lora_adapter_path}...")
                self._model = PeftModel.from_pretrained(self._model, self._lora_adapter_path)

            self._loaded = True
            print("Fine-tuned Qwen3 model loaded successfully")
        except Exception as e:
            self._load_error = str(e)
            print(f"Failed to load fine-tuned model: {e}")

    def is_available(self) -> bool:
        """Check if the fine-tuned model is loaded and ready."""
        return self._loaded

    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a response using the fine-tuned model."""
        if not self._loaded:
            return LLMResponse(
                content=f"Modele fine-tune non disponible: {self._load_error}",
                provider=self.provider_name,
                model=self._base_model_path or "unknown",
                error=True,
            )

        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self._tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode response
        response_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Clean response
        cleaned = self._clean_response(response_text)

        return LLMResponse(
            content=cleaned,
            provider=self.provider_name,
            model=f"{self._base_model_path}+lora",
        )

    def _clean_response(self, content: str) -> str:
        """Clean up model output."""
        if not content:
            return ""

        # Remove thinking tags
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>")[-1]

        # Normalize whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned
