# Open And Closed Model Source Atlas - 2026-07-07

This atlas expands the model directory with sources for closed-provider models, open-weight models, model hubs, model cards, deployment catalogs, and interoperability formats. Model names and availability change quickly, so each local model note should preserve the original source URL and check date.

## Provider Model Catalogs

| Source | What to capture | Local routing |
|---|---|---|
| [OpenAI models](https://developers.openai.com/api/docs/models) | Current model ids, modality support, tool support, lifecycle status, and API guidance. | `Text_LLMs/`, `Reasoning_Models/`, `Multimodal_Models/`, `Audio_Models/` |
| [Anthropic Claude models](https://platform.claude.com/docs/en/about-claude/models/overview) | Claude model families, capability guidance, context, and usage notes. | `Text_LLMs/`, `Reasoning_Models/`, agentic model notes. |
| [Google Gemini models](https://ai.google.dev/gemini-api/docs/models) | Gemini model list, modalities, stable/preview identifiers, and feature support. | `Multimodal_Models/`, `Video_Models/`, `Embedding_Models/` |
| [Google Cloud Model Garden](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models) | Enterprise deployment model catalog and cloud availability. | Cloud deployment and model-serving notes. |
| [Microsoft Foundry partner and community models](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-from-partners) | Model catalog availability, deployment routes, and enterprise governance. | Cloud models, MLOps, enterprise deployment. |

## Open And Community Model Hubs

| Source | What to capture | Local routing |
|---|---|---|
| [Hugging Face Models](https://huggingface.co/models) | Model cards, licenses, tasks, files, libraries, datasets, and community signals. | Model-family directories after classification. |
| [Hugging Face Model Hub docs](https://huggingface.co/docs/hub/en/models-the-hub) | Hosting, discovery, model cards, collaboration, and repository operations. | Model process and metadata documentation. |
| [Kaggle Models](https://www.kaggle.com/models) | Pretrained models, TensorFlow Hub migration records, notebooks, and model usage. | `Models/`, TensorFlow/Kaggle model notes. |
| [TensorFlow models and datasets](https://www.tensorflow.org/resources/models-datasets) | TensorFlow model resources, Kaggle Models, Model Garden, and community resources. | TensorFlow model routing. |
| [PyTorch Hub](https://pytorch.org/hub/) | Published PyTorch models, examples, and runnable load patterns. | Vision, audio, NLP, and research prototypes. |
| [ONNX Model Zoo](https://github.com/onnx/models) | ONNX model files, interoperability examples, and pretrained model references. | `Frameworks/`, model optimization, serving. |
| [NVIDIA NGC catalog](https://catalog.ngc.nvidia.com/) | Containers, pretrained models, Helm charts, and GPU-optimized assets. | GPU infrastructure, model serving, scientific models. |
| [Replicate Explore](https://replicate.com/explore) | Hosted model APIs, demos, task categories, and prototype routes. | Prototype deployment and model comparison. |
| [OpenRouter models](https://openrouter.ai/models) | Provider routing, hosted model availability, context windows, and pricing. | `LLM_Gateway_and_Routing/`, provider comparison. |

## Model Card Fields

| Field | Requirement |
|---|---|
| Identity | Provider, model id, aliases, release date, source URL, and access date. |
| Modality | Text, image, audio, video, embedding, reranking, tool use, or multimodal. |
| Availability | API, hosted inference, open weights, gated weights, region limits, and deprecation state. |
| License/terms | License, commercial permissions, redistribution limits, and model-card restrictions. |
| Evaluation | Provider evals, third-party evals, benchmark caveats, safety reports, and known limitations. |
| Deployment | Context/input/output limits, batching, streaming, hardware, quantization, serving route, and cost. |

## Routing Rules

- Put unknown or mixed-modality models in `Models/` first, then route after modality and license are verified.
- Put model-serving engines and inference performance notes under `04_MLOPS_AND_PRODUCTION_AI/`.
- Put benchmark-only claims under `03_DATASETS_TOOLS_AND_RESOURCES/Datasets/Benchmarks/`.
- Put safety reports and policy restrictions near model notes and cross-link to governance or safety directories when available.
