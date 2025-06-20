from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    ProviderSpec,
    remote_provider_spec,
)


def get_provider_spec() -> ProviderSpec:
    return remote_provider_spec(
        api=Api.eval,
        adapter=AdapterSpec(
            adapter_type="trustyai_ragas",
            pip_packages=["ragas"],  # ["datasets", "langchain-core"],
            config_class="config.RagasEvalProviderConfig",
            module="ragas_eval",
        ),
    )
