from typing import Any, Dict, List

from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.common.job_types import Job
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.apis.inference import Inference
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate

from .config import RagasEvalProviderConfig


class RagasEvaluatorRemote(Eval, BenchmarksProtocolPrivate):
    """Forward eval requests to a remote Ragas server."""

    def __init__(
        self,
        config: RagasEvalProviderConfig,
        datasetio_api: DatasetIO,
        inference_api: Inference,
    ):
        self.config = config
        self.ragas_client = None  # RagasClient()

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        raise NotImplementedError("run_eval is not implemented yet")

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        raise NotImplementedError("evaluate_rows is not implemented yet")

    async def job_status(self, benchmark_id: str, job_id: str) -> Job:
        raise NotImplementedError("job_status is not implemented yet")

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        raise NotImplementedError("job_cancel is not implemented yet")

    async def job_result(
        self, benchmark_id: str, job_id: str
    ) -> EvaluateResponse | None:
        raise NotImplementedError("job_result is not implemented yet")

    async def register_benchmark(self, task_def: Benchmark) -> None:
        raise NotImplementedError("register_benchmark is not implemented yet")
