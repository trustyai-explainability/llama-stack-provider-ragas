"""Integration tests for Ragas evaluation using remote Llama Stack wrappers."""

import logging
import os

import pytest
from ragas import evaluate
from ragas.evaluation import EvaluationResult
from ragas.metrics import answer_relevancy
from ragas.run_config import RunConfig

from llama_stack_provider_ragas.config import KubeflowConfig, RagasProviderRemoteConfig
from llama_stack_provider_ragas.logging_utils import render_dataframe_as_table
from llama_stack_provider_ragas.wrappers_remote import (
    LlamaStackRemoteEmbeddings,
    LlamaStackRemoteLLM,
)

logger = logging.getLogger(__name__)
# mark as integration, see tool.pytest.ini_options in pyproject.toml
pytestmark = pytest.mark.integration_test


@pytest.fixture
def remote_eval_config():
    return RagasProviderRemoteConfig(
        model="granite3.3:2b",
        sampling_params={"temperature": 0.1, "max_tokens": 100},
        embedding_model="all-MiniLM-L6-v2",
        metric_names=["answer_relevancy"],
        kubeflow_config=KubeflowConfig(
            pipelines_endpoint=os.environ["KUBEFLOW_PIPELINES_ENDPOINT"],
            namespace=os.environ["KUBEFLOW_NAMESPACE"],
            llama_stack_url=os.environ["LLAMA_STACK_URL"],
            base_image=os.environ["KUBEFLOW_BASE_IMAGE"],
        ),
    )


@pytest.fixture
def remote_llm(remote_eval_config):
    """Remote LLM wrapper for evaluation."""
    return LlamaStackRemoteLLM(
        base_url="http://localhost:8321",
        model_id=remote_eval_config.model,
        sampling_params=remote_eval_config.sampling_params,
    )


@pytest.fixture
def remote_embeddings(remote_eval_config):
    """Remote embeddings wrapper for evaluation."""
    return LlamaStackRemoteEmbeddings(
        base_url="http://localhost:8321",
        embedding_model_id=remote_eval_config.embedding_model,
    )


def test_client_connection(lls_client):
    models = lls_client.models.list()
    assert models


@pytest.mark.parametrize(
    "metric_to_test",
    [
        pytest.param(m, id=m.name) for m in [answer_relevancy]
    ],  # , context_precision, faithfulness, context_recall]
)
def test_single_metric_evaluation(
    remote_eval_config,
    evaluation_dataset,
    remote_llm,
    remote_embeddings,
    metric_to_test,
) -> None:
    result: EvaluationResult = evaluate(
        dataset=evaluation_dataset,
        metrics=[metric_to_test],
        llm=remote_llm,
        embeddings=remote_embeddings,
        run_config=RunConfig(max_workers=1),
        show_progress=True,
    )

    assert isinstance(result, EvaluationResult)
    pandas_result = result.to_pandas()
    logger.info(render_dataframe_as_table(pandas_result))
    assert metric_to_test.name in pandas_result.columns
    assert len(pandas_result) == len(evaluation_dataset)
    assert pandas_result[metric_to_test.name].dtype == float

    # Use small tolerance for floating point comparisons
    tolerance = 1e-10
    assert pandas_result[metric_to_test.name].min() >= -tolerance
    assert pandas_result[metric_to_test.name].max() <= 1 + tolerance
