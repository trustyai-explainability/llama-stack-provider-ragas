# TODO: decide how to treat these imports & possibly an extras_require
import io
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.apis.scoring import ScoringResult
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate

from llama_stack_provider_ragas.config import RagasProviderRemoteConfig
from llama_stack_provider_ragas.errors import RagasEvaluationError

logger = logging.getLogger(__name__)


class RagasEvaluationJob(Job):
    result: EvaluateResponse | None
    kubeflow_run_id: Optional[str] = None
    pipeline_status: Optional[str] = None


class RagasEvaluatorRemote(Eval, BenchmarksProtocolPrivate):
    """Execute Ragas evaluations using Kubeflow Pipelines."""

    def __init__(
        self,
        config: RagasProviderRemoteConfig,
    ):
        self.config = config
        self.evaluation_jobs: Dict[str, RagasEvaluationJob] = {}
        self.benchmarks: Dict[str, Benchmark] = {}
        try:
            from kfp import Client

            token = os.popen("oc whoami -t").read().strip()
            self.kfp_client = Client(
                host=self.config.kubeflow_config.pipelines_endpoint,
                existing_token=token,
            )
        except ImportError:
            raise RagasEvaluationError(
                "Kubeflow Pipelines SDK not available. Install with: pip install -e .[remote]"
            )

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
            raise RagasEvaluationError(f"Failed to submit evaluation: {str(e)}")

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
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
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
        # TODO: this does not work yet
        run_detail = self.kfp_client.get_run(job.kubeflow_run_id)

        pipeline_name = run_detail.pipeline_spec.pipeline_info.name

        ragas_eval_task_id = None
        for task_detail in run_detail.run_details.task_details:
            if task_detail.display_name == "run-ragas-evaluation":
                ragas_eval_task_id = task_detail.task_id
                break

        if not ragas_eval_task_id:
            raise RagasEvaluationError(
                f"Could not find 'run-ragas-evaluation' task in run {job.kubeflow_run_id}"
            )

        bucket = "public-rhods"  # TODO: make this configurable
        key = f"{pipeline_name}/{job.kubeflow_run_id}/run-ragas-evaluation/{ragas_eval_task_id}/output_results"
        s3_client = boto3.client("s3")

        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            artifact_content = response["Body"].read().decode("utf-8")
        except ClientError as e:
            logger.warning(
                f"Failed to fetch from constructed path, trying to discover: {e}"
            )

            prefix = f"{pipeline_name}/{job.kubeflow_run_id}/"
            paginator = s3_client.get_paginator("list_objects_v2")

            found_key = None
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith("/output_results"):
                        found_key = obj["Key"]
                        break
                if found_key:
                    break

            if found_key:
                logger.info(f"Found results at: s3://{bucket}/{found_key}")
                response = s3_client.get_object(Bucket=bucket, Key=found_key)
                artifact_content = response["Body"].read().decode("utf-8")
            else:
                raise RagasEvaluationError(
                    f"Could not find output_results artifact in S3 for run {job.kubeflow_run_id}"
                )

        df = pd.read_json(io.StringIO(artifact_content), lines=True)

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

        except ImportError:
            raise RagasEvaluationError(
                "Kubeflow Pipelines SDK not available. Install with: pip install kfp"
            )
        except Exception as e:
            logger.error(f"Failed to cancel job: {str(e)}")
            raise RagasEvaluationError(f"Failed to cancel job: {str(e)}")

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
