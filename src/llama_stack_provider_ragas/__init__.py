from typing import Any, Dict

from llama_stack.distribution.datatypes import Api

from .config import RagasEvalProviderConfig
from .ragas_eval import RagasEvaluator


async def get_adapter_impl(
    config: RagasEvalProviderConfig,
    deps: Dict[Api, Any],
) -> RagasEvaluator:
    return RagasEvaluator(config, deps[Api.datasetio], deps[Api.inference])


__all__ = [
    "get_adapter_impl",
    "RagasEvaluator",
]
