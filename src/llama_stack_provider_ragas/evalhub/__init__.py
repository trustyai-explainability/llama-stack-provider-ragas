"""EvalHub RAGAS adapter — standalone module run as Job entrypoint.

This package is part of llama-stack-provider-ragas but is not part of the
Llama Stack provider. It shares the container and dependencies (ragas, etc.)
and is invoked by EvalHub when running RAGAS evaluation jobs.
"""

from .adapter import main

__all__ = ["main"]
