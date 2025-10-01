import os
from typing import List  # noqa

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


@dsl.component(base_image=os.environ["KUBEFLOW_BASE_IMAGE"])
def run_ragas_evaluation(
    model: str,
    sampling_params: dict,
    embedding_model: str,
    metrics: List[str],  # noqa
    llama_stack_base_url: str,
    input_dataset: dsl.Input[dsl.Dataset],
    result_s3_location: str,
):
    import logging

    import pandas as pd
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.run_config import RunConfig

    from llama_stack_provider_ragas.constants import METRIC_MAPPING
    from llama_stack_provider_ragas.logging_utils import render_dataframe_as_table
    from llama_stack_provider_ragas.remote.wrappers_remote import (
        LlamaStackRemoteEmbeddings,
        LlamaStackRemoteLLM,
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

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

    logger.info(f"Saving results to {result_s3_location}")
    df_output.to_json(result_s3_location, orient="records", lines=True)
