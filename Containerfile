# Single image for both Llama Stack RAGAS provider (KFP) and EvalHub RAGAS adapter.
# Install [remote] for Kubeflow Pipelines; [evalhub] for EvalHub job entrypoint (ragas-evalhub-adapter).
# Use UBI Python to avoid Docker Hub rate limits and for OpenShift compatibility.
FROM registry.access.redhat.com/ubi9/python-312:latest

USER 0

WORKDIR /usr/local/src/kfp/components

COPY . .

RUN pip install --no-cache-dir -e ".[remote,evalhub]"
