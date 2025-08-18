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
    uv sync
    ```

3. Install as an editable package. There's `distro` and `dev` optional dependencies to run the sample LS distribution:
    ```bash
    uv pip install -e ".[distro]"
    uv pip install -e ".[dev]"
    ```

## Usage

### Basic Usage
See the [demo notebook](demos/ragas_evaluation_demo.ipynb) for a complete example of using the Ragas provider with Llama Stack.

### Configuration

See config params in the `run.yaml` file. These are still changing.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.