# Model Hub And Provider Index - 2026-07-07

This index maps current model discovery sources into AIForge's model taxonomy. It is intentionally source-first: model names change quickly, so provider docs, model cards, release notes, pricing pages, and benchmark evidence should be captured together.

## Official Provider Catalogs

| Source | What to track | Local routing |
|---|---|---|
| [OpenAI models](https://developers.openai.com/api/docs/models) | Current recommended models, modality support, API surface, and selection guidance. | `Text_LLMs/`, `Reasoning_Models/`, `Video_Models/`, `Audio_Models/` |
| [OpenAI all models](https://developers.openai.com/api/docs/models/all) | Full model list, deprecation status, legacy models, and API availability. | Use for lifecycle checks and migration notes. |
| [OpenAI pricing](https://openai.com/api/pricing/) | Input/output cost, tool cost, service tiers, and modality-specific pricing. | Cost notes in model cards and MLOps FinOps. |
| [Anthropic Claude models](https://platform.claude.com/docs/en/about-claude/models/overview) | Claude model family, capability comparison, context, and recommended use. | `Text_LLMs/`, `Reasoning_Models/`, agent notes. |
| [Anthropic transparency hub](https://www.anthropic.com/transparency) | Model reports, safety evaluations, safeguards, and deployment notes. | Model governance and safety documentation. |
| [Google Gemini models](https://ai.google.dev/gemini-api/docs/models) | Stable, preview, latest, and experimental model strings plus modalities. | `Multimodal_Models/`, `Video_Models/`, `Embedding_Models/` |
| [Gemini models API](https://ai.google.dev/api/models) | Programmatic model metadata and supported functionality. | Automation and provider inventory scripts. |
| [Gemini API changelog](https://ai.google.dev/gemini-api/docs/changelog) | Model lifecycle, shutdowns, feature launches, and migration risks. | Freshness and deprecation checks. |
| [Google Cloud Model Garden](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models) | Google and selected OSS models in enterprise/cloud deployment contexts. | Cloud and enterprise deployment notes. |

## Open And Multi-Provider Hubs

| Source | What to track | Local routing |
|---|---|---|
| [Hugging Face Model Hub](https://huggingface.co/models) | Open model cards, licenses, files, tasks, datasets, and community signals. | Family-specific model folders after classification. |
| [Hugging Face Hub docs](https://huggingface.co/docs/hub/en/models-the-hub) | Model hosting, discovery, model cards, private repos, and collaboration features. | Repository/process documentation. |
| [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index) | Provider-backed inference routes, supported providers, and deployment options. | `Frameworks/`, MLOps serving, provider comparison. |
| [Hugging Face supported inference models](https://huggingface.co/inference/models) | Models available through HF inference providers. | Availability checks and deployment notes. |
| [OpenRouter models](https://openrouter.ai/models) | Multi-provider model routing, pricing, context, and model availability. | Gateway/routing notes in MLOps and model comparison. |
| [Replicate explore](https://replicate.com/explore) | Hosted model demos/APIs across image, video, audio, and utility models. | Prototype and deployment notes. |
| [Together AI models](https://docs.together.ai/docs/serverless-models) | Serverless hosted OSS and proprietary model availability. | Open model deployment and inference comparison. |
| [Groq model docs](https://console.groq.com/docs/models) | Low-latency model hosting, supported models, and deprecation notes. | Inference optimization and provider routing. |
| [LiteLLM providers](https://docs.litellm.ai/docs/providers) | Provider abstraction and model routing across many APIs. | LLM gateway and provider compatibility. |

## Model Entry Checklist

- Provider, model id, release/deprecation status, modality, context/input limits, output limits, and tool support.
- License, commercial terms, API availability, region/compliance limits, and data-retention controls.
- Primary benchmark evidence, provider-reported evals, third-party evals, and known benchmark caveats.
- Price, latency class, throughput limits, batching support, streaming support, and fallback strategy.
- Model card or report link, safety notes, refusal behavior, known failure modes, and update date.

## Routing Rules

- Put model discovery and temporary unknown entries here, then move them into `Text_LLMs/`, `Vision_Models/`, `Audio_Models/`, `Video_Models/`, `Multimodal_Models/`, `Scientific_Models/`, or `Embedding_Models/`.
- Put benchmark-only files in `03_DATASETS_TOOLS_AND_RESOURCES/Datasets/Benchmarks/`.
- Put inference engines, gateways, monitoring, and cost controls in `04_MLOPS_AND_PRODUCTION_AI/`.
