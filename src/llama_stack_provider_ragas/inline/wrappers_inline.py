import asyncio
import logging

from langchain_core.language_models.llms import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from llama_stack.apis.inference import (
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    SamplingParams,
    TopPSamplingStrategy,
)
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


class LlamaStackInlineEmbeddings(BaseRagasEmbeddings):
    """Wrapper that makes Llama Stack inference API embeddings compatible with Ragas."""

    def __init__(
        self, inference_api, embedding_model_id, run_config: RunConfig | None = None
    ):
        super().__init__()
        self.inference_api = inference_api
        self.embedding_model_id = embedding_model_id
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> list[float]:
        """Embed query using asyncio.get_event_loop() to call async version."""
        # TODO: propose a way to configure BaseRagasEmbeddings to use sync or async
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aembed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using asyncio.get_event_loop() to call async version."""
        # TODO: propose a way to configure BaseRagasEmbeddings to use sync or async
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aembed_documents(texts))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using Llama Stack inference API."""
        try:
            request = OpenAIEmbeddingsRequestWithExtraBody(
                model=self.embedding_model_id,
                input=texts,
            )
            response = await self.inference_api.openai_embeddings(request)
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Document embedding failed: {str(e)}")
            raise

    async def aembed_query(self, text: str) -> list[float]:
        """Embed query using Llama Stack inference API."""
        try:
            request = OpenAIEmbeddingsRequestWithExtraBody(
                model=self.embedding_model_id,
                input=text,
            )
            response = await self.inference_api.openai_embeddings(request)
            return response.data[0].embedding  # type: ignore
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise


class LlamaStackInlineLLM(BaseRagasLLM):
    """Wrapper that makes Llama Stack inference API compatible with Ragas."""

    def __init__(
        self,
        inference_api,
        model_id: str,
        sampling_params: SamplingParams | None = None,
        run_config: RunConfig = RunConfig(),
        multiple_completion_supported: bool = True,
    ):
        super().__init__(run_config, multiple_completion_supported)
        self.inference_api = inference_api
        self.model_id = model_id
        self.sampling_params = sampling_params

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        raise NotImplementedError(
            "Sync inline LLMs are not supported, use agenerate_text instead"
        )

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        """Asynchronous text generation using Llama Stack inference API."""
        try:
            generations = []
            llm_output = {
                "llama_stack_responses": [],
                "model_id": self.model_id,
                "provider": "llama_stack",
            }

            # sampling params for this generation should be set via the benchmark config
            # we will ignore the temperature and stop params passed in here
            for _ in range(n):
                request = OpenAICompletionRequestWithExtraBody(
                    model=self.model_id,
                    prompt=prompt.to_string(),
                    max_tokens=self.sampling_params.max_tokens
                    if self.sampling_params
                    else None,
                    temperature=self.sampling_params.strategy.temperature
                    if self.sampling_params
                    and isinstance(self.sampling_params.strategy, TopPSamplingStrategy)
                    else None,
                    top_p=self.sampling_params.strategy.top_p
                    if self.sampling_params
                    and isinstance(self.sampling_params.strategy, TopPSamplingStrategy)
                    else None,
                    stop=self.sampling_params.stop if self.sampling_params else None,
                )
                response = await self.inference_api.openai_completion(request)

                if not response.choices:
                    logger.warning("Completion response returned no choices")

                # Extract text from OpenAI completion response
                choice = response.choices[0] if response.choices else None
                text = choice.text if choice else ""

                # Store Llama Stack response info in llm_output
                llama_stack_info = {
                    "stop_reason": (choice.finish_reason if choice else None),
                    "content_length": len(text),
                    "has_logprobs": choice.logprobs is not None if choice else False,
                }
                llm_output["llama_stack_responses"].append(llama_stack_info)  # type: ignore

                generations.append(Generation(text=text))

            return LLMResult(generations=[generations], llm_output=llm_output)

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    # TODO: revisit this
    # def is_finished(self, response: LLMResult) -> bool:
    #     """
    #     Check if the LLM generation completed successfully.

    #     For Llama Stack responses, we check if the generation was completed
    #     without hitting token limits or other issues.
    #     """
    #     try:
    #         # First, check if we have Llama Stack specific information in llm_output
    #         if response.llm_output and "llama_stack_responses" in response.llm_output:
    #             llama_stack_responses = response.llm_output["llama_stack_responses"]

    #             for i, llama_response in enumerate(llama_stack_responses):
    #                 stop_reason = llama_response.get("stop_reason")
    #                 content_length = llama_response.get("content_length", 0)

    #                 # Check stop_reason from Llama Stack response
    #                 if stop_reason == "out_of_tokens":
    #                     logger.warning(
    #                         f"Generation {i} hit token limit (stop_reason: {stop_reason})"
    #                     )
    #                     return False
    #                 elif stop_reason == "end_of_message":
    #                     # This is usually fine for tool calls, but might indicate incomplete generation
    #                     logger.info(
    #                         f"Generation {i} ended with end_of_message (stop_reason: {stop_reason})"
    #                     )
    #                 elif stop_reason == "end_of_turn":
    #                     # This is the ideal case - normal completion
    #                     logger.debug(
    #                         f"Generation {i} completed normally (stop_reason: {stop_reason})"
    #                     )
    #                 elif stop_reason is None:
    #                     logger.warning(f"Generation {i} has no stop_reason")
    #                     return False

    #                 # Check content length
    #                 if content_length == 0:
    #                     logger.warning(f"Generation {i} has empty content")
    #                     return False
    #                 elif content_length < 10:
    #                     logger.warning(
    #                         f"Generation {i} has very short content ({content_length} chars)"
    #                     )
    #                     return False

    #             # If we have Llama Stack info and all checks pass, we're done
    #             return True

    #         # Fallback to content-based validation if no Llama Stack info
    #         for generation_list in response.generations:
    #             for generation in generation_list:
    #                 # Check if the generated text is empty or None
    #                 if not generation.text or generation.text.strip() == "":
    #                     logger.warning("Empty response from Llama Stack LLM")
    #                     return False

    #                 # Check if the response indicates an error or incomplete generation
    #                 if any(
    #                     error_indicator in generation.text.lower()
    #                     for error_indicator in [
    #                         "error",
    #                         "failed",
    #                         "timeout",
    #                         "incomplete",
    #                         "truncated",
    #                     ]
    #                 ):
    #                     logger.warning(
    #                         f"Response indicates error or incomplete generation: {generation.text[:100]}..."
    #                     )
    #                     return False

    #                 # Check for common truncation indicators
    #                 if generation.text.endswith("...") or generation.text.endswith("…"):
    #                     logger.warning("Response appears to be truncated")
    #                     return False

    #                 # Check if the response is too short (might indicate truncation)
    #                 if len(generation.text.strip()) < 10:
    #                     logger.warning("Response is very short, might be incomplete")
    #                     return False

    #         # If we get here, all generations look good
    #         return True

    #     except Exception as e:
    #         logger.error(f"Error checking if LLM generation is finished: {str(e)}")
    #         # Default to True to avoid false positives, but log the error
    #         return True
