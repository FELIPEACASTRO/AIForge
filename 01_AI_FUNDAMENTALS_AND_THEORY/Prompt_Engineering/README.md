# Prompt Engineering

This directory covers prompt engineering, context engineering, prompt libraries, prompt operations, and domain-specific prompt patterns. It should separate official prompt guidance from community prompt collections and from lower-trust copied system prompts.

## Content Map

| Subarea | What belongs here |
|---|---|
| Official guides | OpenAI, Anthropic, Google Gemini, Microsoft Foundry, IBM, and other provider-authored prompting guidance. |
| Universal techniques | Instruction clarity, examples, structured output, tool use, decomposition, retrieval grounding, refusal and uncertainty policies, and prompt testing. |
| Prompt libraries | Curated prompt collections, tagged by provenance, intended use, freshness, and verification level. |
| PromptOps | Prompt versioning, prompt hubs, eval loops, CI checks, prompt regression tests, and production change control. |
| Domain prompts | Coding, data science, medical, legal, scientific, engineering, education, UX/UI, multilingual, translation, and business prompts. |
| Security and risk | Prompt injection, jailbreaks, system-prompt leakage, prompt attacks, adversarial prompts, and prompt red-teaming. |

## High-Authority Sources

| Source | Use |
|---|---|
| [OpenAI prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering) | Official OpenAI prompt patterns and model guidance. |
| [Anthropic prompt engineering](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) | Claude-specific prompt design and evaluation-first prompting. |
| [Google Gemini prompt design](https://ai.google.dev/gemini-api/docs/prompting-strategies) | Gemini prompt strategies and multimodal prompting. |
| [Microsoft Foundry prompt engineering](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/prompt-engineering) | Azure OpenAI grounding, system messages, and validation guidance. |
| [LangSmith prompt management](https://docs.langchain.com/langsmith/manage-prompts) | Prompt versioning, prompt hub, and template lifecycle. |
| [PromptBench](https://github.com/microsoft/promptbench) | Prompt engineering, prompt attacks, and dynamic evaluation research. |

## Local Routing

- Use `Universal_Techniques/` for general methods that work across models.
- Use domain folders such as `Coding_Prompts/`, `Medical_Prompts/`, `Scientific_Prompts/`, and `Data_Science_Prompts/` for applied prompt templates.
- Keep provider-specific guides as standalone files when the source is official.
- Mark prompt collections as `official`, `academic`, `community`, or `unverified`.

## Next Enrichment

Add a prompt provenance matrix with columns for source, maintainer, license, task family, model family, evaluation status, and risk notes.
