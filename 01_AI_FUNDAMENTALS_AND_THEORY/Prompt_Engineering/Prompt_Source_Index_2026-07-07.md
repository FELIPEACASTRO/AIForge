# Prompt Source Index - 2026-07-07

This index organizes high-authority prompt sources for AIForge. It separates official provider guidance, prompt libraries, prompt operations, and evaluation sources so prompt files can be ranked by provenance instead of copied as disconnected examples.

## Official Provider Guidance

| Source | What to capture | Local routing |
|---|---|---|
| [OpenAI prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering) | General instruction design, decomposition, examples, tool use, and model-facing prompt practices. | `Universal_Techniques/`, `Engineering_Prompts/`, `Evaluation_Prompts/` |
| [OpenAI prompting guide](https://developers.openai.com/api/docs/guides/prompting) | Prompt construction for text, multimodal input, structured outputs, and API-oriented applications. | `Universal_Techniques/`, `Coding_Prompts/` |
| [OpenAI Cookbook](https://developers.openai.com/cookbook) | Runnable examples, prompt patterns, structured outputs, eval examples, and app recipes. | Keep examples with implementation notes. |
| [Anthropic prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) | Evaluation-first prompting workflow and criteria before prompt tuning. | `Evaluation_Prompts/`, `Universal_Techniques/` |
| [Anthropic prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) | Clarity, examples, XML structuring, thinking/tool patterns, and agentic-system prompts. | `Universal_Techniques/`, `Engineering_Prompts/` |
| [Claude Code prompt library](https://code.claude.com/docs/en/prompt-library) | Practical software-engineering prompts with explanations of the pattern behind each prompt. | `Coding_Prompts/`, `Engineering_Prompts/` |
| [Google Gemini prompting strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies) | Gemini-specific prompt strategy, iteration, multimodal prompting, and output control. | `Multilingual_Prompts/`, `UX_UI_Prompts/`, `Creative_Prompts/` |
| [Google Gemini prompt gallery](https://ai.google.dev/gemini-api/prompts) | Prompt examples for audio, video, JSON, multimodal extraction, and application prototypes. | Domain prompt folders by modality. |
| [Google Workspace prompt guide](https://workspace.google.com/resources/ai/writing-effective-prompts/) | Persona/task/context/format patterns for business use cases. | `Business_Prompts/` |
| [Microsoft Foundry prompt engineering](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/prompt-engineering) | Azure/OpenAI prompt techniques, few-shot examples, grounding, and current API guidance. | `Business_Prompts/`, `Engineering_Prompts/` |
| [Microsoft system message design](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/advanced-prompt-engineering) | System-message structure, uncertainty rules, testing, and limitations. | `Universal_Techniques/`, `Evaluation_Prompts/` |
| [Microsoft safety system messages](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/system-message) | Safety metaprompts and layered prompt-safety controls. | `Evaluation_Prompts/`, security prompt files |

## Prompt Libraries And PromptOps

| Source | Use in AIForge | Trust note |
|---|---|---|
| [LangSmith manage prompts](https://docs.langchain.com/langsmith/manage-prompts) | Prompt versioning, prompt hub, template management, and lifecycle tracking. | Operational source; record product dependency. |
| [LangChain prompt templates](https://python.langchain.com/docs/concepts/prompt_templates/) | Template variables, chat prompts, few-shot prompts, and reusable prompt composition. | Framework-specific; avoid treating examples as universal. |
| [Promptfoo](https://www.promptfoo.dev/) | Prompt regression tests, provider comparison, CI checks, red-team tests, and eval reports. | Useful for PromptOps and safety regression. |
| [OpenAI Evals](https://github.com/openai/evals) | Eval harness patterns for model and prompt behavior. | Prefer task-specific evals over generic prompt scores. |
| [PromptBench](https://github.com/microsoft/promptbench) | Prompt attacks, prompt engineering, and robustness evaluation research. | Research source; verify benchmark fit before reuse. |
| [Ragas](https://docs.ragas.io/) | RAG prompt and retrieval evaluation. | Use for RAG-specific prompt quality, not broad writing quality. |
| [DeepEval](https://docs.confident-ai.com/) | LLM unit tests, hallucination checks, and application-level evals. | Track metric definitions and model-as-judge prompts. |

## Source Ranking

| Rank | Source type | How to use |
|---|---|---|
| 1 | Official provider docs and API docs | Best source for model-specific prompt behavior, current capabilities, and supported APIs. |
| 2 | Academic or benchmark-backed prompt research | Use when claims include experiments, datasets, and reproducible evaluation. |
| 3 | PromptOps and eval tool docs | Use for lifecycle, CI, monitoring, and regression testing. |
| 4 | Community prompt libraries | Keep only with provenance, license, task label, risk notes, and local testing. |
| 5 | Copied system prompts or unsourced prompt dumps | Quarantine or mark unverified; never treat as best practice by default. |

## Required Metadata For Prompt Files

- Source URL, maintainer, license or terms, last checked date, and model family.
- Task category, input assumptions, output schema, examples, and expected failure modes.
- Evaluation method, test cases, pass/fail criteria, model version, and regression history.
- Safety controls for privacy, legal/medical/financial advice, prompt injection, jailbreaks, and unsupported claims.
