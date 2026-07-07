# Text LLMs

This directory organizes text-first large language models, including frontier closed models, open-weight models, efficient transformer variants, reasoning models, coding models, and small language models.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Frontier_Closed_Models/` | API-served frontier models from providers such as OpenAI, Anthropic, Google, Mistral, xAI, and others. |
| `Open_Source_LLMs/` | Open-weight and open-source model families such as Llama, Qwen, DeepSeek, Mistral, Gemma, Phi, OLMo, Falcon, and GLM. |
| `Efficient_Transformers/` | Transformer variants, attention alternatives, compression-oriented architectures, and long-context efficiency methods. |

## Required Metadata For Model Pages

| Field | Why it matters |
|---|---|
| Provider or maintainer | Establishes provenance and support expectations. |
| Release date and version | Prevents stale model comparisons. |
| Modality | Separates text-only, vision-language, audio, video, and tool-use models. |
| License and access | Distinguishes API-only, open-weight, research-only, and commercial terms. |
| Context length and tokenizer | Affects RAG, coding, retrieval, and agent design. |
| Training/eval signals | Link to papers, model cards, evals, leaderboards, and known benchmark caveats. |
| Deployment routes | API, Hugging Face, Ollama, vLLM, TensorRT-LLM, Bedrock, Foundry, Model Garden, or local runtime. |

## High-Authority Source Families

- Official model docs and API docs.
- Model cards on Hugging Face or provider hubs.
- Technical reports and system cards.
- Reproducible evaluations from HELM, MLCommons, LM Evaluation Harness, SWE-bench, Open LLM Leaderboard artifacts, and provider-published evals with caveats.
