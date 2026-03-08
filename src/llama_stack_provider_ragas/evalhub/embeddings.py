"""OpenAI-compatible embeddings wrapper for EvalHub adapter."""

from __future__ import annotations

import logging
import os
from typing import Any

from ragas.embeddings.base import BaseRagasEmbeddings
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


class EvalHubOpenAIEmbeddings(BaseRagasEmbeddings):
    """RAGAS embeddings that call an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        run_config: RunConfig | None = None,
    ):
        super().__init__()
        self._base_url = base_url.rstrip("/")
        if not self._base_url.endswith("/v1"):
            self._base_url = f"{self._base_url}/v1"
        self._model_id = model_id
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def _client(self) -> Any:
        if not _HAS_OPENAI:
            raise RuntimeError(
                "OpenAI package is required. Install with: pip install llama-stack-provider-ragas[evalhub]"
            )
        return OpenAI(base_url=self._base_url, api_key=_get_api_key())

    def _validate_embedding(self, embedding: list[float] | str) -> list[float]:
        if isinstance(embedding, str):
            raise ValueError("Expected float embeddings, got base64 string")
        return embedding

    def embed_query(self, text: str) -> list[float]:
        client = self._client()
        try:
            r = client.embeddings.create(input=text, model=self._model_id)
            data = r.data
            if not data:
                raise ValueError("Embeddings response had no data")
            return self._validate_embedding(data[0].embedding)
        except Exception as e:
            logger.error("Embed query failed: %s", e)
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        try:
            r = client.embeddings.create(input=texts, model=self._model_id)
            return [self._validate_embedding(d.embedding) for d in r.data]
        except Exception as e:
            logger.error("Embed documents failed: %s", e)
            raise

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
