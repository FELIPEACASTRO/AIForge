# Prompting Evals And PromptOps Source Atlas - 2026-07-07

This atlas expands prompt engineering beyond prompt examples. It routes official provider guidance, prompt libraries, prompt lifecycle tooling, eval frameworks, and red-team checks into the correct AIForge prompt directories.

## Official Prompt Guidance

| Source | What to capture | Local routing |
|---|---|---|
| [OpenAI prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering) | Prompt structure, clarity, examples, decomposition, and model-specific guidance. | `Universal_Techniques/`, `Engineering_Prompts/` |
| [OpenAI prompt guidance](https://developers.openai.com/api/docs/guides/prompt-guidance) | Current model prompt patterns, migration practices, and model-family guidance. | Provider-specific prompt notes. |
| [OpenAI Cookbook evals topic](https://developers.openai.com/cookbook/topic/evals) | Prompt evaluation examples, regression loops, agent evals, and model comparison patterns. | `Evaluation_Prompts/`, PromptOps notes. |
| [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices) | Evaluation design, graders, datasets, and iterative testing. | `Evaluation_Prompts/` |
| [Anthropic prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) | Prompting workflow, success criteria, and when prompt engineering is appropriate. | `Universal_Techniques/` |
| [Anthropic prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) | Clarity, examples, XML structure, thinking, tool use, and agentic prompts. | `Universal_Techniques/`, `Engineering_Prompts/` |
| [Google Gemini prompting strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies) | Gemini prompt strategy, output control, examples, and multimodal prompting. | `Multilingual_Prompts/`, `UX_UI_Prompts/`, `Creative_Prompts/` |
| [Gemini Enterprise prompt strategies](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/prompts/prompt-design-strategies) | Enterprise prompt design, prompt gallery, optimization, and safety filters. | Business and production prompts. |
| [Microsoft Azure OpenAI prompt engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering) | Azure OpenAI prompt construction, examples, and grounding guidance. | `Business_Prompts/`, `Engineering_Prompts/` |

## PromptOps And Evaluation Tooling

| Source | What to capture | Local routing |
|---|---|---|
| [Promptfoo documentation](https://www.promptfoo.dev/docs/intro/) | Prompt tests, model comparisons, CI, red-team evals, assertions, and reports. | `Evaluation_Prompts/`, MLOps prompt management. |
| [LangSmith prompt management](https://docs.langchain.com/langsmith/manage-prompts) | Prompt versioning, prompt hub, templates, and lifecycle management. | PromptOps and LLMOps. |
| [Langfuse prompt management](https://langfuse.com/docs/prompts) | Prompt versions, labels, production rollout, and tracing integration. | PromptOps and observability. |
| [Ragas documentation](https://docs.ragas.io/) | RAG evaluation metrics, datasets, testset generation, and eval workflows. | RAG prompt evaluation. |
| [DeepEval documentation](https://docs.confident-ai.com/) | LLM unit tests, hallucination checks, red-team and agent eval support. | `Evaluation_Prompts/`, guardrails. |
| [OpenAI Evals GitHub](https://github.com/openai/evals) | Eval examples, task structure, and legacy/open-source eval patterns. | Eval harness examples. |
| [PromptBench](https://github.com/microsoft/promptbench) | Prompt robustness, attacks, prompt engineering methods, and benchmark research. | Prompt robustness and safety. |

## Prompt File Metadata

| Field | Requirement |
|---|---|
| Provenance | Source URL, source type, maintainer, license/terms, and date checked. |
| Model context | Provider, model family, model version, parameters, tools, and modality. |
| Task | User goal, audience, constraints, input schema, output schema, and examples. |
| Evaluation | Test cases, expected outputs, graders/assertions, pass threshold, and regression history. |
| Safety | Privacy, prompt injection, jailbreak exposure, unsupported professional advice, and refusal boundaries. |

## Local Routing

- Put general techniques in `Universal_Techniques/`.
- Put code, data, legal, medical, education, business, and creative examples in their domain-specific prompt folders.
- Put prompt tests and judge prompts in `Evaluation_Prompts/`.
- Put operational prompt registries and traces in `04_MLOPS_AND_PRODUCTION_AI/LLMOps_and_Prompt_Management/`.
- Quarantine copied prompt dumps unless provenance, license, target model, and eval evidence are available.
