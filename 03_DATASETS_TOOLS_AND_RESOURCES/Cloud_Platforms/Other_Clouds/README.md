# Other Clouds

This directory covers AI and ML infrastructure outside the main hyperscaler folders, including specialized GPU clouds, regional clouds, sovereign clouds, HPC platforms, and managed AI providers.

## Content Map

| Subdirectory | Scope |
|---|---|
| `AI-ML/` | Managed AI, model hosting, inference, training, GPUs, notebooks, vector services, and MLOps features on non-core cloud providers. |

## Source Families

- GPU and AI compute providers such as CoreWeave, Lambda, Paperspace, Crusoe, RunPod, Together, Replicate, Modal, Baseten, and Fireworks.
- Regional or sovereign cloud AI offerings.
- HPC and research cloud resources for AI training, evaluation, and open-science workloads.

## Reference Links

- CoreWeave Kubernetes Service: https://docs.coreweave.com/docs/products/cks
- Lambda Cloud: https://docs.lambda.ai/public-cloud/
- RunPod: https://docs.runpod.io/
- Replicate: https://replicate.com/docs
- Modal: https://modal.com/docs
- Baseten: https://docs.baseten.co/

## Evaluation Checklist

- GPU/accelerator availability and quota model.
- Data residency, compliance, networking, and storage.
- Model-serving latency, scaling, observability, and security boundaries.
- Cost model, spot/preemptible behavior, and reproducibility.
- API compatibility with PyTorch, TensorFlow, JAX, Hugging Face, Kubernetes, or OpenAI-compatible serving.

## Routing Rules

- Put AWS, Azure, and Google-specific resources in their dedicated cloud folders if present.
- Put platform-neutral deployment patterns in `../../../04_MLOPS_AND_PRODUCTION_AI/Deployment/`.
- Put model hubs and catalogs in `../../Global_AI_Ecosystem/` or `../../../02_LLM_AND_AI_MODELS/`.
