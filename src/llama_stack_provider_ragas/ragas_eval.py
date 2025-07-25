import logging
from typing import Any, Dict, List, Optional

from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.apis.inference import Inference
from llama_stack.apis.scoring import ScoringResult
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from ragas import EvaluationDataset
from ragas import evaluate as ragas_evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AspectCritic,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from .config import RagasEvalProviderConfig
from .constants import METRIC_MAPPING
from .errors import RagasConfigError, RagasEvaluationError
from .logging_utils import render_dataframe_as_table
from .wrappers_inline import LlamaStackInlineEmbeddings, LlamaStackInlineLLM

logger = logging.getLogger(__name__)


class RagasEvaluator(Eval, BenchmarksProtocolPrivate):
    """Ragas evaluation provider for Llama Stack."""

    def __init__(
        self,
        config: RagasEvalProviderConfig,
        datasetio_api: DatasetIO,
        inference_api: Inference,
    ):
        self.config = config
        self.datasetio_api = datasetio_api
        self.inference_api = inference_api
        self.job_results: Dict[str, EvaluateResponse] = {}
        self.benchmarks: Dict[str, Benchmark] = {}

    def _get_metrics(self, scoring_functions: List[str]) -> List:
        """Get the list of metrics to run based on scoring functions.

        Args:
            scoring_functions: List of scoring function names to use

        Returns:
            List of metrics (unconfigured - ragas_evaluate will configure them)
        """
        metrics = []

        for metric_name in scoring_functions:
            if metric_name in METRIC_MAPPING:
                metric = METRIC_MAPPING[metric_name]
                metrics.append(metric)
            else:
                logger.warning(f"Unknown metric: {metric_name}")

        if not metrics:
            # Use default metrics if none specified or all invalid
            logger.info("Using default metrics")
            metrics = [
                answer_relevancy,
                context_precision,
                faithfulness,
                context_recall,
            ]

        return metrics

    async def _prepare_dataset(
        self, dataset_id: str, limit: int = -1
    ) -> EvaluationDataset:
        all_rows = await self.datasetio_api.iterrows(
            dataset_id=dataset_id,
            limit=limit,
        )
        return EvaluationDataset.from_list(all_rows.data)

    async def register_benchmark(self, task_def: Benchmark) -> None:
        self.benchmarks[task_def.identifier] = task_def
        logger.info(f"Registered benchmark: {task_def.identifier}")

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        eval_candidate = benchmark_config.eval_candidate
        if eval_candidate.type != "model":
            raise RagasEvaluationError(
                "Ragas currently only supports model candidates. "
                "We will add support for agents soon!"
            )

        model_id = benchmark_config.eval_candidate.model
        sampling_params = eval_candidate.sampling_params

        ragas_run_config = RunConfig(max_workers=self.config.ragas_max_workers)
        if self.config.additional_config:
            for key, value in self.config.additional_config.items():
                if hasattr(ragas_run_config, key):
                    setattr(ragas_run_config, key, value)

        llm_wrapper = LlamaStackInlineLLM(
            self.inference_api, model_id, sampling_params, run_config=ragas_run_config
        )

        embeddings_wrapper = LlamaStackInlineEmbeddings(
            self.inference_api, self.config.embedding_model, run_config=ragas_run_config
        )
        task_def = self.benchmarks[benchmark_id]
        dataset_id = task_def.dataset_id
        scoring_functions = task_def.scoring_functions

        try:
            eval_dataset = await self._prepare_dataset(
                dataset_id, benchmark_config.num_examples
            )
            metrics = self._get_metrics(scoring_functions)

            result = ragas_evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                llm=llm_wrapper,
                embeddings=embeddings_wrapper,
                experiment_name=self.config.experiment_name,
                run_config=ragas_run_config,
                raise_exceptions=self.config.raise_exceptions,
                column_map=self.config.column_map,
                show_progress=self.config.show_progress,
                batch_size=self.config.batch_size,
            )

            # Render evaluation results as a rich table for better readability
            result_df = result.to_pandas()
            table_output = render_dataframe_as_table(
                result_df, "Ragas Evaluation Results"
            )
            logger.info(f"Ragas evaluation completed:\n{table_output}")

            # Convert scores to ScoringResult format
            scores = {}
            for metric_name in scoring_functions:
                metric_scores = result[metric_name]
                score_rows = [{"score": score} for score in metric_scores]

                if metric_scores:
                    aggregated_score = sum(metric_scores) / len(metric_scores)
                else:
                    aggregated_score = 0.0

                scores[metric_name] = ScoringResult(
                    score_rows=score_rows,
                    aggregated_results={metric_name: aggregated_score},
                )

            logger.info(f"Evaluation completed for model {model_id}. Scores: {scores}")
            res = EvaluateResponse(generations=eval_dataset.to_list(), scores=scores)

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise RagasEvaluationError(f"Evaluation failed: {str(e)}")

        job_id = str(len(self.job_results))
        self.job_results[job_id] = res
        return Job(job_id=job_id, status=JobStatus.completed)

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark."""
        raise NotImplementedError(
            "evaluate_rows is not implemented, use run_eval instead"
        )

    async def job_status(self, benchmark_id: str, job_id: str) -> Job:
        """Get the status of a job.

        Args:
            benchmark_id: The ID of the benchmark to run the evaluation on.
            job_id: The ID of the job to get the status of.

        Returns:
            The status of the evaluation job.
        """
        if job_id not in self.job_results:
            raise RagasEvaluationError(f"Job {job_id} not found")

        return Job(job_id=job_id, status=JobStatus.completed)

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        if job_id not in self.job_results:
            raise RagasEvaluationError(f"Job {job_id} not found")

        return self.job_results[job_id]

    def get_available_metrics(self) -> List[str]:
        """Get list of available metric names.

        Returns:
            List of available metric names
        """
        return list(METRIC_MAPPING.keys())

    def create_custom_metric(
        self, metric_name: str, definition: str, llm_model: Optional[str] = None
    ) -> AspectCritic:
        """Create a custom AspectCritic metric.

        Args:
            metric_name: Name for the metric
            definition: Definition/criteria for the metric
            llm_model: Optional LLM model to use (overrides config)

        Returns:
            Configured AspectCritic metric
        """
        llm = None
        if llm_model:
            llm = llm_factory(model=llm_model)
        elif self.llm:
            llm = self.llm
        else:
            raise RagasConfigError("No LLM available for custom metric")

        metric = AspectCritic(name=metric_name, llm=llm, definition=definition)
        return metric
