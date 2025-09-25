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
            module="llama_stack_provider_ragas.remote",
            pip_packages=["ragas", "kfp", "kfp-kubernetes", "kfp-server-api", "boto3"],
            config_class="llama_stack_provider_ragas.config.RagasProviderRemoteConfig",
        ),
        api_dependencies=[
            Api.inference,
            Api.files,
            Api.benchmarks,
            Api.datasetio,
            Api.telemetry,
        ],
    )
