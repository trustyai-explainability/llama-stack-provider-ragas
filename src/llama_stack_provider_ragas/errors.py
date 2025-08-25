class RagasConfigError(Exception):
    """Raised when there is an error in the Ragas configuration."""

    pass


class RagasEvaluationError(Exception):
    """Raised when there is an error during evaluation."""

    pass


class RagasDatasetError(Exception):
    """Raised when there is an error with the dataset format."""

    pass
