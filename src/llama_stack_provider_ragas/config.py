import json
from typing import Any, Dict, List, Optional

from llama_stack.schema_utils import json_schema_type
from pydantic import BaseModel, Field, computed_field, field_validator
from ragas.metrics import Metric

from .constants import METRIC_MAPPING


@json_schema_type
class RagasEvalProviderConfig(BaseModel):
    """Configuration for Ragas evaluation provider."""

    model: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        description=(
            "Model to use for evaluation. "
            "Adding here for completeness, as it is already provided in the benchmark config's eval_candidate. "
            "It must match the identifier of the model in Llama Stack."
        ),
    )

    sampling_params: dict = Field(
        default={"temperature": 0.1, "max_tokens": 100},
        description=(
            "Sampling parameters for the model. "
            "Also here for completeness, as it is already provided in the benchmark config's eval_candidate. "
        ),
    )

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "Embedding model for Ragas evaluation. "
            "At the moment, this cannot be set in the benchmark config, so it must be set here. "
            "It must match the identifier of the embedding model in Llama Stack."
        ),
    )

    metric_names: List[str] = Field(
        default=[
            "answer_relevancy",
            "context_precision",
            "faithfulness",
            "context_recall",
        ],
        description="Metrics to use for evaluation",
    )

    @field_validator("metric_names", mode="before")
    @classmethod
    def parse_metrics(cls, v):
        """Parse metrics from string if needed (for YAML env var substitution)."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @computed_field
    @property
    def metric_functions(self) -> List[Metric]:
        return [METRIC_MAPPING[metric] for metric in self.metric_names]

    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size for evaluation. If None, no batching is done.",
    )

    show_progress: bool = Field(
        default=True, description="Whether to show progress bar during evaluation"
    )

    raise_exceptions: bool = Field(
        default=True,
        description="Whether to raise exceptions or return NaN for failed evaluations",
    )

    experiment_name: Optional[str] = Field(
        default=None, description="Name for experiment tracking"
    )

    column_map: Optional[Dict[str, str]] = Field(
        default=None, description="Mapping of dataset column names to expected names"
    )

    additional_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional configuration parameters for Ragas"
    )

    ragas_max_workers: int = Field(
        default=1,
        description="Maximum number of concurrent workers for Ragas evaluation. Controls the level of parallelism.",
    )
