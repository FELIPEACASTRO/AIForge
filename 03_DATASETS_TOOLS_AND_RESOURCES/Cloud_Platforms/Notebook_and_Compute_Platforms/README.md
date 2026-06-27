# Notebook and Compute Platforms

> Hosted notebook environments and rentable GPU/TPU compute — the practical on-ramp for running ML code on the datasets and models indexed across AIForge, from free browser notebooks to serverless GPU clouds and fully managed MLOps platforms.

## Why it matters

Most ML work starts not with a local GPU but with a hosted notebook (Colab, Kaggle) or a rented cloud GPU. These platforms remove the friction of CUDA setup, dependency management, and hardware procurement, letting practitioners go from "found a dataset" to "training a model" in minutes. Choosing the right tier — free notebook, on-demand GPU marketplace, serverless inference, or managed MLOps suite — is a recurring decision that shapes cost, reproducibility, and time-to-result.

## Taxonomy

| Category | What it is | Representative platforms |
|---|---|---|
| **Free hosted notebooks** | Browser Jupyter with a free GPU/TPU quota; zero setup | Google Colab, Kaggle Notebooks, SageMaker Studio Lab |
| **Collaborative notebooks** | Team-oriented notebooks, real-time editing, data apps | Deepnote, Hex, Lightning AI Studios |
| **GPU rental clouds** | On-demand / reserved VMs and pods billed per hour | Paperspace, Lambda, CoreWeave, Crusoe |
| **GPU marketplaces** | P2P / aggregated spot capacity at lowest cost | Vast.ai, RunPod, Clore.ai |
| **Serverless GPU** | Autoscaling, pay-per-second inference & jobs | Modal, RunPod Serverless, Replicate, Baseten |
| **Managed MLOps platforms** | End-to-end train/deploy/monitor on a hyperscaler | SageMaker, Vertex AI, Azure ML, Databricks |

## Key platforms — free & collaborative notebooks

| Platform | Free GPU offering | Notes | Link |
|---|---|---|---|
| Google Colaboratory | T4 (variable), ~12 h sessions | Ubiquitous; Pro/Pro+ for longer runtimes, A100/L4 | https://colab.research.google.com/ |
| Kaggle Notebooks | ~30 GPU h/week, P100/T4, 9 h max | Tightly coupled to Kaggle datasets & competitions | https://www.kaggle.com/docs/notebooks |
| Amazon SageMaker Studio Lab | 4 GPU h/day (T4), free, no AWS account | Persistent 15 GB storage; standalone from AWS | https://studiolab.sagemaker.aws/ |
| Deepnote | Free CPU tier; collaborative cells | Real-time multiplayer, SQL blocks, data apps | https://deepnote.com/ |
| Lightning AI Studios | Free T4 credits, then pay-as-you-go | VS Code + Jupyter studio from the PyTorch Lightning team | https://lightning.ai/ |
| Hex | Notebook + BI / data-app hybrid | Polished collaborative analytics workspace | https://hex.tech/ |

## Key platforms — GPU clouds, marketplaces & serverless

| Platform | Model | Strengths | Link |
|---|---|---|---|
| Paperspace Gradient | On-demand VMs + notebooks | Free shared M4000 tier; A100/H100 on demand | https://www.paperspace.com/gradient |
| Lambda | Managed GPU cloud | NVLink/InfiniBand clusters, Lambda Stack preinstalled, uptime SLA | https://lambda.ai/ |
| CoreWeave | Kubernetes-native GPU cloud | Large H100/H200/GB200 fleets for training & inference | https://www.coreweave.com/ |
| Crusoe | GPU cloud | Energy-aware data centers, H100/H200 capacity | https://crusoe.ai/ |
| RunPod | Pods + serverless GPU | Container Pods (Secure/Community Cloud) + autoscaling endpoints | https://www.runpod.io/ |
| Vast.ai | P2P GPU marketplace | Lowest spot prices; interruptible, dynamic pricing | https://vast.ai/ |
| Modal | Serverless compute | Python-native, per-second billing, autoscale to zero | https://modal.com/ |
| Replicate | Serverless model hosting | Run/fine-tune open models via API, pay-per-prediction | https://replicate.com/ |
| Baseten | Serverless inference | Truss packaging, autoscaling model deployment | https://www.baseten.co/ |
| Saturn Cloud | Hosted Dask/notebooks | Scalable data-science clusters | https://saturncloud.io/ |

## Key platforms — managed MLOps suites

| Platform | Scope | Notebook surface | Link |
|---|---|---|---|
| Amazon SageMaker | Full ML lifecycle (Studio, Training, Pipelines, Endpoints, Clarify) | SageMaker Studio / Notebooks | https://aws.amazon.com/sagemaker/ |
| Google Vertex AI | Managed MLOps, BigQuery + GenAI integration | Vertex AI Workbench | https://cloud.google.com/vertex-ai |
| Azure Machine Learning | AutoML, pipelines, designer, endpoints | Azure ML Studio notebooks | https://azure.microsoft.com/products/machine-learning |
| Databricks | Lakehouse + ML, MLflow, Spark distributed training | Databricks notebooks | https://www.databricks.com/product/machine-learning |

## Underlying open-source stack

| Project | Role | Link |
|---|---|---|
| Project Jupyter / JupyterLab | Core notebook protocol & IDE behind nearly every platform above | https://jupyter.org/ |
| JupyterHub | Multi-user notebook server (self-hosted teams/classes) | https://jupyter.org/hub |
| Papermill | Parameterize & execute notebooks as pipelines | https://github.com/nteract/papermill |
| nbconvert | Convert notebooks to scripts, HTML, PDF | https://github.com/jupyter/nbconvert |
| Binder (mybinder.org) | Turn a Git repo into a live, reproducible notebook | https://mybinder.org/ |

## Key papers

| Title | Venue / ID | Link |
|---|---|---|
| Jupyter Notebooks — a publishing format for reproducible computational workflows | ELPUB 2016 (Kluyver et al.) | https://eprints.soton.ac.uk/403913/ |
| Using Jupyter for reproducible scientific workflows | arXiv:2102.09562 | https://arxiv.org/abs/2102.09562 |
| Computational reproducibility of Jupyter notebooks from biomedical publications | arXiv:2209.04308 | https://arxiv.org/abs/2209.04308 |
| A Large-Scale Study About Quality and Reproducibility of Jupyter Notebooks | MSR 2019 (Pimentel et al.) | https://doi.org/10.1109/MSR.2019.00077 |
| Numerical simulation projects in micromagnetics with Jupyter | arXiv:2303.01784 | https://arxiv.org/abs/2303.01784 |

## Cross-references in AIForge

- [AWS Cloud Platform resources](../AWS/) — deeper AWS-specific ML tooling (SageMaker, EC2 GPU instances)
- [Other Clouds](../Other_Clouds/) — additional hyperscaler and niche cloud providers
- [HuggingFace Hub](../../HuggingFace_Hub/) — Spaces (free hosted GPU demos) and model/dataset hosting
- [Datasets](../../Datasets/) — the data you run on these compute platforms

## Sources

- https://zackproser.com/blog/cloud-gpu-services-jupyter-notebook-reviewed
- https://www.muratkarakaya.net/2025/03/free-gpu-services-for-llm-enthusiasts.html
- https://lyceum.technology/magazine/lambda-labs-vs-runpod-vs-vast-ai/
- https://www.runpod.io/articles/guides/top-serverless-gpu-clouds
- https://mikaelahonen.com/en/data/ml-platforms-comparison-major-clouds/
- https://arxiv.org/abs/2102.09562
- https://eprints.soton.ac.uk/403913/
- https://jupyter.org/

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
