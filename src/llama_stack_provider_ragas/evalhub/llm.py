"""OpenAI-compatible LLM wrapper for EvalHub adapter."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.language_models.llms import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


def _get_api_key() -> str:
    return (
        os.environ.get("OPENAICOMPATIBLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "DUMMY"
    )


class EvalHubOpenAILLM(BaseRagasLLM):
    """RAGAS LLM that calls an OpenAI-compatible completions endpoint from EvalHub JobSpec."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        run_config: RunConfig | None = None,
    ):
        if run_config is None:
            run_config = RunConfig()
        super().__init__(run_config, multiple_completion_supported=True)
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _client(self) -> Any:
        if not _HAS_OPENAI:
            raise RuntimeError(
                "OpenAI package is required. Install with: pip install llama-stack-provider-ragas[evalhub]"
            )
        return OpenAI(
            base_url=f"{self._base_url}/v1" if not self._base_url.endswith("/v1") else self._base_url,
            api_key=_get_api_key(),
        )

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        client = self._client()
        kwargs = {
            "model": self._model_id,
            "prompt": prompt.to_string(),
            "n": n,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        t = temperature if temperature is not None else self._temperature
        if t is not None:
            kwargs["temperature"] = t
        if stop:
            kwargs["stop"] = stop

        try:
            response = client.completions.create(**kwargs)
        except Exception as e:
            logger.error("Completion request failed: %s", e)
            raise

        generations = []
        for choice in getattr(response, "choices", []) or []:
            text = getattr(choice, "text", "") or ""
            generations.append(Generation(text=text))

        if not generations:
            generations = [Generation(text="")]

        return LLMResult(generations=[generations], llm_output={"provider": "evalhub_openai"})

    def is_finished(self, response: LLMResult) -> bool:
        """Check if the LLM response is finished. Completions API returns full response, so always True."""
        return True

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return self.generate_text(prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks)

    def get_temperature(self, n: int) -> float:
        if self._temperature is not None:
            return self._temperature
        return 0.3 if n > 1 else 1e-8
