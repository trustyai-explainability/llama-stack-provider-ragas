from typing import Any, Dict

from llama_stack.distribution.datatypes import Api

from .config import RagasEvalProviderConfig
from .ragas_eval import RagasEvaluatorInline, RagasEvaluatorRemote


async def get_adapter_impl(
    config: RagasEvalProviderConfig,
    deps: Dict[Api, Any],
) -> RagasEvaluatorRemote:
    return RagasEvaluatorRemote(config, deps[Api.datasetio], deps[Api.inference])


async def get_provider_impl(
    config: RagasEvalProviderConfig,
    deps: Dict[Api, Any],
) -> RagasEvaluatorInline:
    return RagasEvaluatorInline(config, deps[Api.datasetio], deps[Api.inference])


__all__ = [
    "get_adapter_impl",
    "get_provider_impl",
]
