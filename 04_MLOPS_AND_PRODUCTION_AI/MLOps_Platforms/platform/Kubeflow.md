# Kubeflow

## Description

Kubeflow is an **open-source, Kubernetes-native Machine Learning (ML) platform**, dedicated to making deployments of ML workflows simple, portable, and scalable. Its unique value proposition lies in providing a **unified and modular platform** for the entire ML lifecycle (MLOps), from experimentation to production, leveraging the orchestration, scalability, and portability inherent to Kubernetes. It allows data scientists and ML engineers to build, train, and deploy models on any infrastructure where Kubernetes can run (public cloud, private cloud, on-premises).

## Statistics

*   **Production Adoption:** 44% of users reported running Kubeflow in production (2022 User Survey).
*   **Use of Multiple Components:** 84% to 85% of users use more than one Kubeflow component, indicating adoption of the complete, modular platform.
*   **Most Used Components (2023 Survey):** Pipelines (90%), Notebooks (76%), and Katib (47%).
*   **Community:** Large and active, with use by Global 500 companies.

## Features

Kubeflow is composed of several modular components that cover the complete ML lifecycle:
*   **Kubeflow Pipelines (KFP):** Creation and orchestration of portable, scalable, container-based ML workflows.
*   **Notebooks:** Spawning and management of Jupyter Notebook instances (and others) on Kubernetes for experimentation and development.
*   **Katib:** Hyperparameter tuning and neural architecture search service (AutoML) to optimize models.
*   **Training Operators:** Distributed training of ML models using popular frameworks such as TensorFlow, PyTorch, MXNet, and XGBoost.
*   **KFServing (KServe):** Deployment, serving, and management of ML models at scale, with features such as auto-scaling, canary rollouts, and A/B testing.
*   **Metadata:** Tracking and management of metadata for ML artifacts (datasets, models, pipelines).

## Use Cases

*   **Building MLOps Platforms:** Companies use Kubeflow as the backbone to build their own internal MLOps platforms, standardizing ML development and deployment.
*   **Distributed Training:** Execution of large-scale model training tasks that require multiple GPUs or CPUs, such as training Large Language Models (LLMs) or Computer Vision models.
*   **Experimentation and Reproduction:** Use of Notebooks and Pipelines to ensure that ML experiments are reproducible and can be easily moved from the R&D phase to production.
*   **Model Serving at Scale:** Deployment of ML models as scalable microservices with KServe, enabling low-latency, high-throughput inference.
*   **Adoption Sectors:** Telecommunications (Verizon), Transportation (Delta), Healthcare, Finance (Goldman Sachs).

## Integration

The primary integration is done through the **Kubeflow Pipelines SDK (KFP SDK)**, which allows ML workflows to be defined in Python. The pipeline is then compiled into a YAML file that is deployed to the Kubeflow cluster.

**Code Example (Defining a Simple Pipeline Component):**

```python
from kfp.v2.dsl import component
from kfp.v2 import dsl
from kfp.v2.compiler import Compiler

# 1. Define a pipeline component
@component(
    packages_to_install=['pandas'],
    base_image='python:3.9'
)
def load_data(data_path: str) -> str:
    """Loads data from a path and returns a summary."""
    import pandas as pd
    df = pd.read_csv(data_path)
    summary = f"Data loaded with {len(df)} rows and {len(df.columns)} columns."
    print(summary)
    return summary

# 2. Define the complete pipeline
@dsl.pipeline(
    name='simple-kubeflow-pipeline',
    description='A simple pipeline to load data.'
)
def data_pipeline(data_file: str = 'gs://my-bucket/data.csv'):
    # Use the defined component
    load_data_task = load_data(data_path=data_file)
    
# 3. Compile the pipeline into a YAML file (for deployment to Kubeflow)
# Compiler().compile(pipeline_func=data_pipeline, package_path='data_pipeline.yaml')
```

## URL

https://www.kubeflow.org/