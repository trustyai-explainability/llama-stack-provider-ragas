"""RAGAS framework adapter for EvalHub.

Reads the evaluation dataset from /test_data (when populated by S3 init container)
or /data and runs RAGAS metrics against the model in the job spec.
Uses eval-hub-sdk (installed with the evalhub extra).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragas import EvaluationDataset, evaluate as ragas_evaluate
from ragas.run_config import RunConfig

from evalhub.adapter import (
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    OCIArtifactSpec,
)
from evalhub.adapter.models.job import ErrorInfo, MessageInfo
from evalhub.models.api import EvaluationResult

from llama_stack_provider_ragas.constants import METRIC_MAPPING

from .embeddings import EvalHubOpenAIEmbeddings
from .llm import EvalHubOpenAILLM

logger = logging.getLogger(__name__)

# When test_data_ref.s3 is set, EvalHub's init container downloads objects here
TEST_DATA_DIR = Path("/test_data")
DEFAULT_DATA_DIR = Path("/data")
DEFAULT_DATASET_FILENAME = "dataset.jsonl"
_DATA_SUFFIXES = (".jsonl", ".json")


def _first_dataset_in_dir(path: Path) -> Path | None:
    """Return the first dataset file (.jsonl or .json) in path, or None."""
    if not path.exists() or not path.is_dir():
        return None
    for f in sorted(path.iterdir()):
        if f.suffix.lower() in _DATA_SUFFIXES and f.is_file():
            return f
    return None


def _resolve_data_path(config: JobSpec) -> Path:
    bc = config.benchmark_config or {}
    explicit = bc.get("data_path")
    if explicit:
        p = Path(explicit)
        if p.is_absolute():
            return p
        return DEFAULT_DATA_DIR / explicit
    # Prefer /test_data when populated by S3 init container (custom data)
    test_data_file = TEST_DATA_DIR / DEFAULT_DATASET_FILENAME
    if test_data_file.exists():
        return test_data_file
    first_in_test = _first_dataset_in_dir(TEST_DATA_DIR)
    if first_in_test is not None:
        return first_in_test
    # Fall back to /data (e.g. when no test_data_ref)
    default_file = DEFAULT_DATA_DIR / DEFAULT_DATASET_FILENAME
    if default_file.exists():
        return default_file
    first_in_data = _first_dataset_in_dir(DEFAULT_DATA_DIR)
    if first_in_data is not None:
        return first_in_data
    return default_file


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    with open(path) as f:
        if suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if suffix == ".json":
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            raise ValueError(f"JSON dataset must be a list or {{'data': list}}, got {type(data)}")
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _apply_column_map(records: list[dict[str, Any]], column_map: dict[str, str] | None) -> list[dict[str, Any]]:
    if not column_map:
        return records
    return [{column_map.get(k, k): v for k, v in row.items()} for row in records]


def _limit_records(records: list[dict[str, Any]], num_examples: int | None) -> list[dict[str, Any]]:
    if num_examples is None or num_examples <= 0:
        return records
    return records[:num_examples]


class RagasEvalHubAdapter(FrameworkAdapter):
    """EvalHub framework adapter that runs RAGAS evaluation using data from /test_data or /data."""

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.time()
        logger.info("Starting RAGAS EvalHub job %s for benchmark %s", config.id, config.benchmark_id)

        try:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Validating configuration and resolving data path",
                        message_code="initializing",
                    ),
                )
            )
            self._validate_config(config)
            bc = config.benchmark_config or {}

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.15,
                    message=MessageInfo(
                        message="Loading dataset from /test_data or /data",
                        message_code="loading_data",
                    ),
                )
            )
            data_path = _resolve_data_path(config)
            records = _load_dataset(data_path)
            column_map = bc.get("column_map")
            if isinstance(column_map, dict):
                records = _apply_column_map(records, column_map)
            records = _limit_records(records, config.num_examples)
            if not records:
                raise ValueError(f"No records in dataset at {data_path} (or after limit)")
            eval_dataset = EvaluationDataset.from_list(records)
            logger.info(
                "Dataset loaded: path=%s records=%d columns=%s",
                data_path,
                len(records),
                list(records[0].keys()) if records else [],
            )

            metric_names = bc.get("metrics") or bc.get("scoring_functions") or list(METRIC_MAPPING.keys())
            metrics = [METRIC_MAPPING[name] for name in metric_names if name in METRIC_MAPPING]
            if not metrics:
                metrics = list(METRIC_MAPPING.values())
                logger.info("Using default RAGAS metrics")

            model_url = config.model.url.strip().rstrip("/")
            model_name = config.model.name
            embedding_model = bc.get("embedding_model") or model_name
            embedding_url = bc.get("embedding_url") or model_url
            run_config = RunConfig(max_workers=1)
            llm = EvalHubOpenAILLM(
                base_url=model_url,
                model_id=model_name,
                max_tokens=bc.get("max_tokens"),
                temperature=bc.get("temperature"),
                run_config=run_config,
            )
            embeddings = EvalHubOpenAIEmbeddings(
                base_url=embedding_url,
                model_id=embedding_model,
                run_config=run_config,
            )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=MessageInfo(
                        message=f"Running RAGAS evaluation ({len(metrics)} metrics)",
                        message_code="running_evaluation",
                    ),
                )
            )
            ragas_result = ragas_evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
            )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.85,
                    message=MessageInfo(
                        message="Processing RAGAS results",
                        message_code="post_processing",
                    ),
                )
            )
            result_df = ragas_result.to_pandas()
            n_evaluated = len(result_df)
            evaluation_results: list[EvaluationResult] = []
            scores_for_overall: list[float] = []
            for metric_name in [m.name for m in metrics]:
                if metric_name not in result_df.columns:
                    continue
                series = result_df[metric_name].dropna()
                values = series.tolist()
                if not values:
                    continue
                avg = sum(values) / len(values)
                scores_for_overall.append(avg)
                evaluation_results.append(
                    EvaluationResult(
                        metric_name=metric_name,
                        metric_value=round(avg, 6),
                        metric_type="float",
                        num_samples=len(values),
                        metadata={"min": min(values), "max": max(values)},
                    )
                )
            overall_score = sum(scores_for_overall) / len(scores_for_overall) if scores_for_overall else None

            oci_artifact = None
            if config.exports and config.exports.oci:
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.PERSISTING_ARTIFACTS,
                        progress=0.9,
                        message=MessageInfo(
                            message="Persisting results to OCI",
                            message_code="persisting_artifacts",
                        ),
                    )
                )
                results_dir = Path("/tmp/ragas_evalhub_results") / config.id
                results_dir.mkdir(parents=True, exist_ok=True)
                results_file = results_dir / "results.jsonl"
                result_df.to_json(results_file, orient="records", lines=True)
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=results_dir,
                        coordinates=config.exports.oci.coordinates,
                    )
                )

            duration = time.time() - start_time
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=n_evaluated,
                duration_seconds=round(duration, 2),
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "ragas",
                    "data_path": str(data_path),
                    "metrics": [m.name for m in metrics],
                },
                oci_artifact=oci_artifact,
            )

        except Exception as e:
            logger.exception("RAGAS EvalHub job %s failed", config.id)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error=ErrorInfo(message=str(e), message_code="job_failed"),
                    error_details={"exception_type": type(e).__name__},
                )
            )
            raise

    def _validate_config(self, config: JobSpec) -> None:
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")
        if not config.model or not config.model.url:
            raise ValueError("model.url is required")
        if not config.model.name:
            raise ValueError("model.name is required")


def main() -> None:
    """Entry point for running the adapter as an EvalHub K8s Job."""
    import os
    import sys

    from evalhub.adapter import DefaultCallbacks

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    job_spec_path = os.environ.get("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
    try:
        adapter = RagasEvalHubAdapter(job_spec_path=job_spec_path)
        logger.info("Loaded job %s", adapter.job_spec.id)
        logger.info("Benchmark: %s", adapter.job_spec.benchmark_id)

        oci_auth = os.environ.get("OCI_AUTH_CONFIG_PATH")
        callbacks = DefaultCallbacks(
            job_id=adapter.job_spec.id,
            benchmark_id=adapter.job_spec.benchmark_id,
            benchmark_index=adapter.job_spec.benchmark_index,
            provider_id=adapter.job_spec.provider_id,
            sidecar_url=adapter.job_spec.callback_url,
            oci_auth_config_path=Path(oci_auth) if oci_auth else None,
            oci_insecure=os.environ.get("OCI_REGISTRY_INSECURE", "false").lower() == "true",
        )

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info("Job completed: %s", results.id)
        callbacks.report_results(results)
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error("Job spec not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)
