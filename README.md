# `trustyai-ragas` <br> Ragas as an Out-of-Tree Llama Stack Provider

⚠️ Warning! This project is in early stages of development!

## About
This repository implements [Ragas](https://github.com/explodinggradients/ragas) as an out-of-tree Llama Stack evaluation provider. Ragas is a toolkit for evaluating and optimizing Large Language Model (LLM) applications with objective metrics.

## Features
The goal is to provide all of Ragas' evaluation functionality over Llama Stack's eval API, while leveraging the Llama Stack's built-in APIs for inference (llms and embeddings), datasets & benchmarks.

## Installation

### Prerequisites
* Python 3.12
* `uv`

### Setup
Clone and use `uv` (see below). I will update this README with alternative instructions using the build.yaml file and llama stack's build command.


1. Clone this repository
    ```bash
    git clone <repository-url>
    cd llama-stack-provider-ragas
    ```

2. Create and activate a virtual environment
    ```bash
    uv venv 
    source .venv/bin/activate
    ```

3. Install as an editable package. There's `distro` and `dev` optional dependencies to run the sample LS distribution:
    ```bash
    uv pip install -e ".[distro]"
    uv pip install -e ".[dev]"
    ```

## Usage

### Inline provider
This version of the provider runs the Ragas evaluation in the same process as the Llama Stack server.

### Remote provider
This version of the provider runs the Ragas evaluation in a remote process, using Kubeflow Pipelines.

You will need:

- a Llama Stack server running locally
- a Kubeflow Pipelines server
- a `.env` file with the following variables:
```
KUBEFLOW_PIPELINES_ENDPOINT=[...] # get this via oc get routes -A | grep -i pipeline
KUBEFLOW_NAMESPACE=[...] # this is the name of the data science project
LLAMA_STACK_URL=[...] # this is the url of the llama stack server -- use ngrok with a locally running llama stack server
KUBEFLOW_BASE_IMAGE=quay.io/diegosquayorg/my-ragas-provider-image:latest # see Containerfile for details
```







