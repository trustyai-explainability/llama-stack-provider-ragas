from typing import Any

from llama_stack.distribution.datatypes import Api

from .config import RagasProviderInlineConfig, RagasProviderRemoteConfig
from .eval_inline import RagasEvaluatorInline
from .eval_remote import RagasEvaluatorRemote


async def get_adapter_impl(
    config: RagasProviderRemoteConfig,
    deps: dict[Api, Any],
) -> RagasEvaluatorRemote:
    return RagasEvaluatorRemote(config)


async def get_provider_impl(
    config: RagasProviderInlineConfig,
    deps: dict[Api, Any],
) -> RagasEvaluatorInline:
    return RagasEvaluatorInline(config, deps[Api.datasetio], deps[Api.inference])


__all__ = [
    "get_adapter_impl",
    "get_provider_impl",
]
