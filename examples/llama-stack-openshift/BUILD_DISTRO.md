# Building a Llama Stack Distribution with the remote llama stack ragas provider

## Prerequisites
* Cloned [llama-stack-distribution](https://github.com/opendatahub-io/llama-stack-distribution.git) locally and checked out the rhoai-v2.22 branch
* The llama-stack-provider-ragas project cloned locally

> [!NOTE]
> Using rhoai-v2.22 as it is using a comparable Llama Stack version as the provider

## Structure setup
* Copy the [llama-stack-provider-ragas](https://github.com/trustyai-explainability/llama-stack-provider-ragas) folder into the [llama-stack-distribution/redhat-distribution](https://github.com/opendatahub-io/llama-stack-distribution/tree/rhoai-v2.22/redhat-distribution) directory. 
* Copy the [providers.d/remote/evaltrustyai_ragas.yaml](https://github.com/trustyai-explainability/llama-stack-provider-ragas/tree/dev/providers.d/remote/eval) file into the [llama-stack-distribution/redhat-distribution/providers.d/remote/eval](https://github.com/opendatahub-io/llama-stack-distribution/tree/rhoai-v2.22/redhat-distribution/providers.d/remote/eval) directory

Your llama-stack-distribution file structure should look like this:
```
llama-stack-distribution/
└── redhat-distribution/
    ├── run.yaml
    ├── llama-stack-provider-ragas/
    └── providers.d/
        ├── inline/
        └── remote/
            └── eval/
                ├── trustyai_lmeval.yaml
                └── trustyai_ragas.yaml
```

## Configuring the run.yaml file
Add the remote eval provider to the run.yaml file like below so it will be available to the Llama Stack server when the image is run:
```
...
eval:
    ...
  - provider_id: trustyai_ragas_remote
    provider_type: remote::trustyai_ragas
    config:
      metrics: ["answer_relevancy"]
      kubeflow_config:
        pipelines_endpoint: ${env.KUBEFLOW_PIPELINES_ENDPOINT}
        namespace: ${env.KUBEFLOW_NAMESPACE}
        llama_stack_url: ${env.LLAMA_STACK_URL}
        base_image: ${env.KUBEFLOW_BASE_IMAGE}
...
```

## Configuring the Containerfile
In order to install the dependencies and remote llama-stack ragas provider the following edits need to be made to [Containerfile.in](https://github.com/opendatahub-io/llama-stack-distribution/blob/rhoai-v2.22/redhat-distribution/Containerfile.in).

Add the following build instructions to the file past the `RUN pip install --no-cache llama-stack==0.2.10` step.

```
COPY --chown=1001:0 redhat-distribution/llama-stack-provider-ragas/ ./llama-stack-provider-ragas/
WORKDIR /opt/app-root/llama-stack-provider-ragas
RUN pip install -e ".[dev]"
WORKDIR /opt/app-root
```

Now run the build.py script to generate a modified Containerfile.

* Run `python build.py`
* Inspect that the new Containerfile includes the above added build steps.

## Build the distribution image
To build the Llama Stack Distribution run the following commands.
``` bash
podman build --platform linux/amd64 -f redhat-distribution/Containerfile -t rh .
podman tag localhost/rh:latest quay.io/<your_quay_repo>/llama-stack:<tag>
podman push quay.io/<your_quay_repo>/llama-stack:<tag>
```

You should now be able to make use of your own custom Llama Stack distribution image.