import pytest
from llama_stack_client import LlamaStackClient
from ragas import EvaluationDataset

from llama_stack_provider_ragas.config import RagasEvalProviderConfig


@pytest.fixture
def lls_client():
    return LlamaStackClient(base_url="http://localhost:8321")


@pytest.fixture
def eval_config():
    return RagasEvalProviderConfig(
        model="meta-llama/Llama-3.2-3B-Instruct",
        sampling_params={"temperature": 0.1, "max_tokens": 100},
        embedding_model="all-MiniLM-L6-v2",
        metric_names=["answer_relevancy"],
    )


@pytest.fixture
def raw_evaluation_data():
    """Sample data for Ragas evaluation."""
    return [
        {
            "user_input": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "retrieved_contexts": [
                "Paris is the capital and most populous city of France."
            ],
            "reference": "Paris",
        },
        {
            "user_input": "Who invented the telephone?",
            "response": "Alexander Graham Bell invented the telephone in 1876.",
            "retrieved_contexts": [
                "Alexander Graham Bell was a Scottish-American inventor who patented the first practical telephone."
            ],
            "reference": "Alexander Graham Bell",
        },
        {
            "user_input": "What is photosynthesis?",
            "response": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "retrieved_contexts": [
                "Photosynthesis is a process used by plants to convert light energy into chemical energy."
            ],
            "reference": "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy.",
        },
    ]


@pytest.fixture
def evaluation_dataset(raw_evaluation_data):
    """Create EvaluationDataset from sample data."""
    return EvaluationDataset.from_list(raw_evaluation_data)
