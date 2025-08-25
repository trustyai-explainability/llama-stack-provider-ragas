# TODO: decide how to treat these imports & possibly an extras_require
import logging
import subprocess
import uuid
from typing import Any

import pandas as pd
import requests
from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.apis.scoring import ScoringResult
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate

from llama_stack_provider_ragas.config import RagasProviderRemoteConfig
from llama_stack_provider_ragas.errors import RagasEvaluationError
from llama_stack_provider_ragas.logging_utils import render_dataframe_as_table

logger = logging.getLogger(__name__)


class RagasEvaluationJob(Job):
    result: EvaluateResponse | None
    kubeflow_run_id: str | None = None
    pipeline_status: str | None = None


class RagasEvaluatorRemote(Eval, BenchmarksProtocolPrivate):
    """Execute Ragas evaluations using Kubeflow Pipelines."""

    def __init__(
        self,
        config: RagasProviderRemoteConfig,
    ):
        self.config = config
        self.evaluation_jobs: dict[str, RagasEvaluationJob] = {}
        self.benchmarks: dict[str, Benchmark] = {}
        try:
            import kfp

            result = subprocess.run(
                ["oc", "whoami", "-t"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            token = result.stdout.strip()
            if not token:
                raise RagasEvaluationError(
                    "No token found. Please run `oc login` and try again."
                )

            # the kfp.Client handles the healthz endpoint poorly, run a pre-flight check manually
            response = requests.get(
                f"{self.config.kubeflow_config.pipelines_endpoint}/apis/v2beta1/healthz",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                timeout=5,
            )
            response.raise_for_status()

            self.kfp_client = kfp.Client(
                host=self.config.kubeflow_config.pipelines_endpoint,
                existing_token=token,
            )
        except ImportError as e:
            raise RagasEvaluationError(
                "Kubeflow Pipelines SDK not available. Install with: pip install .[remote]"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RagasEvaluationError(
                f"Failed to get OpenShift token. Command failed with exit code {e.returncode}: {e.stderr.strip()}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RagasEvaluationError(
                f"Failed to connect to Kubeflow Pipelines server at {self.config.kubeflow_config.pipelines_endpoint}, "
                "do you need a new token?"
            ) from e
        except Exception as e:
            raise RagasEvaluationError(
                "Failed to initialize Kubeflow Pipelines client."
            ) from e

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        """Submit a Ragas evaluation job to Kubeflow Pipelines."""
        try:
            eval_candidate = benchmark_config.eval_candidate
            if eval_candidate.type != "model":
                raise RagasEvaluationError(
                    "Ragas currently only supports model candidates. "
                    "We will add support for agents soon!"
                )

            if benchmark_id not in self.benchmarks:
                raise RagasEvaluationError(f"Benchmark {benchmark_id} not found")

            task_def = self.benchmarks[benchmark_id]

            job_id = str(uuid.uuid4())
            job = RagasEvaluationJob(
                job_id=job_id,
                status=JobStatus.in_progress,
                result=None,
                kubeflow_run_id=None,
                pipeline_status="submitted",
            )

            kubeflow_run_id = await self._submit_to_kubeflow(
                benchmark_id=benchmark_id,
                benchmark_config=benchmark_config,
                task_def=task_def,
                job_id=job_id,
            )

            job.kubeflow_run_id = kubeflow_run_id
            self.evaluation_jobs[job_id] = job

            logger.info(
                f"Submitted Ragas evaluation job {job_id} to Kubeflow with run ID {kubeflow_run_id}"
            )

            return job

        except Exception as e:
            logger.error(f"Failed to submit evaluation job: {str(e)}")
            raise RagasEvaluationError(f"Failed to submit evaluation: {str(e)}") from e

    async def _submit_to_kubeflow(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
        task_def: Benchmark,
        job_id: str,
    ) -> str:
        from .kubeflow.pipeline import ragas_evaluation_pipeline

        temperature = (
            benchmark_config.eval_candidate.sampling_params.temperature
            if benchmark_config.eval_candidate.sampling_params.strategy.type == "top_p"
            else None
        )

        sampling_params = {
            "temperature": temperature,
            "max_tokens": benchmark_config.eval_candidate.sampling_params.max_tokens,
        }

        pipeline_args = {
            "dataset_id": task_def.dataset_id,
            "llama_stack_base_url": self.config.kubeflow_config.llama_stack_url,
            "num_examples": (
                benchmark_config.num_examples
                if benchmark_config.num_examples is not None
                else -1
            ),
            "model": benchmark_config.eval_candidate.model,
            "sampling_params": sampling_params,
            "embedding_model": self.config.embedding_model,
            "metrics": self.config.metric_names,
        }

        run_result = self.kfp_client.create_run_from_pipeline_func(
            pipeline_func=ragas_evaluation_pipeline,
            arguments=pipeline_args,
            run_name=f"ragas-eval-{benchmark_id}-{job_id[:8]}",
            namespace=self.config.kubeflow_config.namespace,
            experiment_name="lls-provider-ragas-runs",
        )

        return run_result.run_id

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        raise NotImplementedError("Not implemented yet -- use run_eval instead")

    async def job_status(self, benchmark_id: str, job_id: str) -> Job:
        # TODO: replace inmem dict with kubeflow client
        if (job := self.evaluation_jobs.get(job_id)) is None:
            raise RagasEvaluationError(f"Job {job_id} not found")

        if job.kubeflow_run_id is None:
            raise RagasEvaluationError(f"Job {job_id} has no Kubeflow run ID")

        try:
            run_detail = self.kfp_client.get_run(job.kubeflow_run_id)
            if run_detail.state == "FAILED":
                # TODO: add error message
                job.status = JobStatus.failed
            elif run_detail.state == "SUCCEEDED":
                job.status = JobStatus.completed
                await self._fetch_kubeflow_results(job)
            elif run_detail.state == "RUNNING" or run_detail.state == "PENDING":
                job.status = JobStatus.in_progress
            else:
                raise RagasEvaluationError(
                    f"Unknown Kubeflow run state: {run_detail.state}"
                )

            return job

        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}")
            return job

    async def _fetch_kubeflow_results(self, job: RagasEvaluationJob) -> None:
        """Fetch results directly from S3."""
        s3_url = "s3://public-rhods/ragas-evaluation-pipeline/results.jsonl"

        try:
            df = pd.read_json(s3_url, lines=True)
            logger.info(f"Successfully fetched results from {s3_url}")
        except Exception as e:
            raise RagasEvaluationError(
                f"Failed to fetch results from {s3_url}: {str(e)}"
            ) from e

        table_output = render_dataframe_as_table(df, "Fetched Evaluation Results")
        logger.info(f"Fetched Evaluation Results:\n{table_output}")

        # TODO: move the rest into a conversion function
        generation_columns = [
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
        ]
        generations = df[generation_columns].to_dict("records")

        metric_columns = [col for col in df.columns if col in self.config.metric_names]
        scores = {}

        for metric_name in metric_columns:
            metric_scores = df[metric_name].dropna().tolist()
            score_rows = [{"score": score} for score in metric_scores]

            scores[metric_name] = ScoringResult(
                score_rows=score_rows,
                aggregated_results={
                    "average": sum(metric_scores) / len(metric_scores)
                    if metric_scores
                    else 0.0,
                    "count": len(metric_scores),
                    "min": min(metric_scores) if metric_scores else 0.0,
                    "max": max(metric_scores) if metric_scores else 0.0,
                },
            )

        job.result = EvaluateResponse(generations=generations, scores=scores)

        logger.info(
            f"Successfully fetched results for job {job.job_id}: {len(generations)} generations, {len(scores)} metrics"
        )

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a running Kubeflow pipeline."""
        if job_id not in self.evaluation_jobs:
            raise RagasEvaluationError(f"Job {job_id} not found")

        job = self.evaluation_jobs[job_id]

        if not job.kubeflow_run_id:
            job.job_status = JobStatus.failed
            return

        try:
            self.kfp_client.runs.terminate_run(job.kubeflow_run_id)
            job.status = JobStatus.failed
            job.pipeline_status = "cancelled"

            logger.info(
                f"Cancelled Kubeflow run {job.kubeflow_run_id} for job {job_id}"
            )

        except ImportError as e:
            raise RagasEvaluationError(
                "Kubeflow Pipelines SDK not available. Install with: pip install kfp"
            ) from e
        except Exception as e:
            logger.error(f"Failed to cancel job: {str(e)}")
            raise RagasEvaluationError(f"Failed to cancel job: {str(e)}") from e

    async def job_result(
        self, benchmark_id: str, job_id: str
    ) -> EvaluateResponse | None:
        job = await self.job_status(benchmark_id, job_id)

        if job.status == JobStatus.completed:
            return job.result
        elif job.status == JobStatus.failed:
            raise RagasEvaluationError(f"Job {job_id} failed")
        else:
            return None  # Job still running

    async def register_benchmark(self, task_def: Benchmark) -> None:
        """Register a benchmark for evaluation."""
        self.benchmarks[task_def.benchmark_id] = task_def
        logger.info(f"Registered benchmark {task_def.benchmark_id}")
