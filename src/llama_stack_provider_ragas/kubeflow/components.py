import os
from typing import List

from dotenv import load_dotenv
from kfp import dsl

load_dotenv()


@dsl.component(base_image=os.environ["KUBEFLOW_BASE_IMAGE"])
def retrieve_data_from_llama_stack(
    dataset_id: str,
    llama_stack_base_url: str,
    output_dataset: dsl.Output[dsl.Dataset],
    num_examples: int = -1,  # TODO: parse this
):
    import pandas as pd
    from llama_stack_client import LlamaStackClient

    client = LlamaStackClient(base_url=llama_stack_base_url)
    dataset = client.datasets.retrieve(dataset_id=dataset_id)
    df = pd.DataFrame(dataset.source.rows)
    df.to_json(output_dataset.path, orient="records", lines=True)


@dsl.component(
    base_image=os.environ["KUBEFLOW_BASE_IMAGE"],
    packages_to_install=["s3fs", "kubernetes"],
)
def run_ragas_evaluation(
    model: str,
    sampling_params: dict,
    embedding_model: str,
    metrics: List[str],
    llama_stack_base_url: str,
    input_dataset: dsl.Input[dsl.Dataset],
    output_results: dsl.Output[dsl.Dataset],
):
    import base64
    import logging
    import os

    import pandas as pd
    from kubernetes import client, config
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.run_config import RunConfig

    from llama_stack_provider_ragas.constants import METRIC_MAPPING
    from llama_stack_provider_ragas.logging_utils import render_dataframe_as_table
    from llama_stack_provider_ragas.wrappers_remote import (
        LlamaStackRemoteEmbeddings,
        LlamaStackRemoteLLM,
    )

    def set_aws_credentials_from_k8s_secret():
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        secret = v1.read_namespaced_secret(
            name="aws-credentials", namespace="ragas-eval-v3"
        )

        required_keys = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
        missing_keys = [key for key in required_keys if key not in secret.data]
        
        if missing_keys:
            raise ValueError(f"Missing required AWS credentials in secret: {missing_keys}")

        for key in required_keys:
            os.environ[key] = base64.b64decode(secret.data[key]).decode("utf-8")

    set_aws_credentials_from_k8s_secret()

    logger = logging.getLogger(__name__)

    llm = LlamaStackRemoteLLM(
        base_url=llama_stack_base_url,
        model_id=model,
        sampling_params=sampling_params,
    )
    embeddings = LlamaStackRemoteEmbeddings(
        base_url=llama_stack_base_url,
        embedding_model_id=embedding_model,
    )

    metrics = [METRIC_MAPPING[m] for m in metrics]
    run_config = RunConfig(max_workers=1)

    with open(input_dataset.path) as f:
        df_input = pd.read_json(f, lines=True)
        eval_dataset = EvaluationDataset.from_list(df_input.to_dict(orient="records"))

    ragas_output: EvaluationResult = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    df_output = ragas_output.to_pandas()
    table_output = render_dataframe_as_table(df_output, "Ragas Evaluation Results")
    logger.info(f"Ragas evaluation completed:\n{table_output}")

    s3_location = "s3://public-rhods/ragas-evaluation-pipeline/results.jsonl"
    df_output.to_json(s3_location, orient="records", lines=True)

    df_output.to_json(output_results.path, orient="records", lines=True)
