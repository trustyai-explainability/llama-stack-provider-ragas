import pytest
from langchain_core.prompt_values import StringPromptValue

from llama_stack_provider_ragas.wrappers_remote import (
    LlamaStackRemoteEmbeddings,
    LlamaStackRemoteLLM,
)

# mark as integration, see tool.pytest.ini_options in pyproject.toml
pytestmark = pytest.mark.integration_test


@pytest.fixture
def lls_remote_embeddings():
    return LlamaStackRemoteEmbeddings(
        base_url="http://localhost:8321",
        embedding_model_id="all-MiniLM-L6-v2",
    )


@pytest.fixture
def lls_remote_llm():
    """Remote LLM wrapper for evaluation."""
    return LlamaStackRemoteLLM(
        base_url="http://localhost:8321",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        sampling_params={"temperature": 0.1, "max_tokens": 100},
    )


def test_remote_embeddings_sync(lls_remote_embeddings):
    embeddings = lls_remote_embeddings.embed_query("Hello, world!")
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], float)

    embeddings = lls_remote_embeddings.embed_documents(["Hello, world!"])
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], list)
    assert isinstance(embeddings[0][0], float)


@pytest.mark.asyncio
async def test_remote_embeddings_async(lls_remote_embeddings):
    embeddings = await lls_remote_embeddings.aembed_query("Hello, world!")
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], float)

    embeddings = await lls_remote_embeddings.aembed_documents(
        ["Hello, world!", "How are you?"]
    )
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], list)
    assert isinstance(embeddings[0][0], float)
    assert len(embeddings) == 2  # Two input texts


def test_remote_llm_sync(lls_remote_llm):
    prompt = StringPromptValue(text="What is the capital of France?")
    result = lls_remote_llm.generate_text(prompt, n=1)

    assert hasattr(result, "generations")
    assert len(result.generations) == 1
    assert len(result.generations[0]) == 1
    assert isinstance(result.generations[0][0].text, str)
    assert len(result.generations[0][0].text) > 0

    assert hasattr(result, "llm_output")
    assert result.llm_output["model_id"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert result.llm_output["provider"] == "llama_stack_remote"
    assert len(result.llm_output["llama_stack_responses"]) == 1


@pytest.mark.asyncio
async def test_remote_llm_async(lls_remote_llm):
    prompt = StringPromptValue(text="What is the capital of France?")
    result = await lls_remote_llm.agenerate_text(prompt, n=1)

    assert hasattr(result, "generations")
    assert len(result.generations) == 1
    assert len(result.generations[0]) == 1
    assert isinstance(result.generations[0][0].text, str)
    assert len(result.generations[0][0].text) > 0

    assert hasattr(result, "llm_output")
    assert result.llm_output["model_id"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert result.llm_output["provider"] == "llama_stack_remote"
    assert len(result.llm_output["llama_stack_responses"]) == 1
