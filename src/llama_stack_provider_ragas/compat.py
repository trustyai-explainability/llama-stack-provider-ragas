"""
Compatibility layer for llama_stack imports.

This module provides backward compatibility by attempting to import from
the legacy llama_stack package first, then falling back to the newer
llama_stack_api package structure.
"""

# Provider datatypes and API definitions
try:  # Legacy llama_stack layout
    from llama_stack.apis.datatypes import Api
    from llama_stack.providers.datatypes import (
        BenchmarksProtocolPrivate,
        InlineProviderSpec,
        ProviderSpec,
        RemoteProviderSpec,
    )
except (ImportError, ModuleNotFoundError):
    # Newer llama_stack_api layout
    from llama_stack_api import (
        Api,
        BenchmarksProtocolPrivate,
        InlineProviderSpec,
        ProviderSpec,
        RemoteProviderSpec,
    )

# Benchmarks
try:
    from llama_stack.apis.benchmarks import Benchmark
except (ImportError, ModuleNotFoundError):
    from llama_stack_api import Benchmark

# Common job types
try:
    from llama_stack.apis.common.job_types import Job, JobStatus
except (ImportError, ModuleNotFoundError):
    from llama_stack_api import Job, JobStatus

# DatasetIO
try:
    from llama_stack.apis.datasetio import DatasetIO
except (ImportError, ModuleNotFoundError):
    from llama_stack_api import DatasetIO

# Eval
try:
    from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
except (ImportError, ModuleNotFoundError):
    from llama_stack_api import BenchmarkConfig, Eval, EvaluateResponse

# Inference
try:
    from llama_stack.apis.inference import (
        Inference,
        OpenAICompletionRequestWithExtraBody,
        OpenAIEmbeddingsRequestWithExtraBody,
        SamplingParams,
        TopPSamplingStrategy,
    )
except (ImportError, ModuleNotFoundError):
    from llama_stack_api import (
        Inference,
        OpenAICompletionRequestWithExtraBody,
        OpenAIEmbeddingsRequestWithExtraBody,
        SamplingParams,
        TopPSamplingStrategy,
    )

# Scoring
try:
    from llama_stack.apis.scoring import ScoringResult
except (ImportError, ModuleNotFoundError):
    from llama_stack_api import ScoringResult

# Schema utils
try:
    from llama_stack.schema_utils import json_schema_type
except (ImportError, ModuleNotFoundError):
    from llama_stack_api import json_schema_type

__all__ = [
    # API and Provider types
    "Api",
    "BenchmarksProtocolPrivate",
    "InlineProviderSpec",
    "ProviderSpec",
    "RemoteProviderSpec",
    # Benchmarks
    "Benchmark",
    # Job types
    "Job",
    "JobStatus",
    # DatasetIO
    "DatasetIO",
    # Eval
    "BenchmarkConfig",
    "Eval",
    "EvaluateResponse",
    # Inference
    "Inference",
    "OpenAICompletionRequestWithExtraBody",
    "OpenAIEmbeddingsRequestWithExtraBody",
    "SamplingParams",
    "TopPSamplingStrategy",
    # Scoring
    "ScoringResult",
    # Schema utils
    "json_schema_type",
]
