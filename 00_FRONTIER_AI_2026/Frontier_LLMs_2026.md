# 🛰️ Frontier Closed LLMs — 2026

> The proprietary leading edge: the newest flagship closed models from OpenAI, Google DeepMind, Anthropic and xAI as they stand in autumn 2026.

**Refreshed:** 2026-10-09 · **Scope:** releases through October 2026

> ⚠️ Fast-moving field; entries reflect public reporting at refresh time. Confirm versions, licenses and benchmarks at the source links before relying on them. Benchmark figures below are attributed to their source, not asserted as settled fact.

## OpenAI — GPT-6 family

| Item | Date | What's new |
|---|---|---|
| [GPT-6 with Intelligent UI ("GPT-6 for everyone")](https://openai.com/index/gpt-6-for-everyone/) | 2026-10-07 | ChatGPT update that generates adaptive, interactive UI (diagrams, side-by-side comparisons, in-chat tools) per question; rolled out globally Oct 7 — Plus/Pro/Business/Enterprise served by GPT-6 Sol and Free/Go by GPT-6 Luna, replacing GPT-5.6 Sol/Luna in ChatGPT. |
| [GPT-6.1 Sol](https://openai.com/index/introducing-gpt-6-1-sol/) | 2026-09-29 | Updated mid/high-tier reasoning + agentic coding model (API id `gpt-6.1-sol`) positioned between GPT-6 Sol and GPT-6 Astra; ~1.05M-token context, 128K max output, pricing reported ~$2/$10 per MTok. Primary announcement is OpenAI's research page "Introducing GPT-6.1 Sol". |
| [GPT-6 Sol](https://techcrunch.com/2026/09/22/openai-launches-gpt-6-sol-and-luna/) | 2026-09-22 | Cheaper GPT-6-generation model for complex work/coding (API id `gpt-6-sol`), ~1.05M context, reported ~$2/$10 per MTok (about half of GPT-5.6 Sol); OpenAI claims roughly Astra-level reliability with about half the errors and a far lower coding-deception rate. |
| [GPT-6 Luna](https://github.blog/changelog/2026-09-22-openais-gpt-6-sol-and-gpt-6-luna-now-available/) | 2026-09-22 | Smallest/cheapest GPT-6 tier for high-volume tasks (API id `gpt-6-luna`), ~1.05M context, reported ~$0.10/$0.50 per MTok; now the default model for ChatGPT Free/Go. Launched alongside GPT-6 Sol. |
| [GPT-6 Astra](https://openai.com/index/gpt-6-astra/) | 2026-09-03 | OpenAI's new-generation frontier flagship (API id `gpt-6-astra`), positioned as SOTA at coding, math and agentic computer/browser use; ~1.05M context, pricing reported ~$10/$50 per MTok. ARC Prize reports 62.7% on ARC-AGI-3 on a standard harness and up to 99.9% via a provider-adapter/custom harness. Successor to GPT-5.6. |

## Google DeepMind — Gemini 3.x / 4

| Item | Date | What's new |
|---|---|---|
| [Gemini 4 Argon](https://x.com/GoogleDeepMind/status/2105388084154056939) | 2026-09-30 | Google DeepMind's new frontier model for complex coding, enterprise knowledge work and cybersecurity defense; raises the output limit to 1M tokens, with intro pricing reported ~$2/$10 per MTok. Initially gated to trusted testers via the Fairwind Program (no public API model id yet); as reported, it trails Claude Opus 5.5 on the independent Artificial Analysis index. Official DeepMind announcement. |
| [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/changelog) | 2026-09-02 | Google's most intelligent Flash-tier model (`gemini-3.8-flash`), GA Sep 2, engineered for long-horizon software engineering, autonomous agents and complex enterprise workflows; confirmed on the official Gemini API changelog. |
| [Gemini 3.1 Pro](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/) | 2026-02-19 | Gemini 3 Pro-tier reasoning flagship (preview since Feb 19); Google reports 77.1% on ARC-AGI-2 (more than 2x Gemini 3 Pro). Remained the top documented Pro model until Gemini 4 Argon. Pre-dates the June refresh; official Google blog. |

## Anthropic — Claude 5.5

| Item | Date | What's new |
|---|---|---|
| [Claude Sonnet 5.5](https://platform.claude.com/docs/en/release-notes/overview) | 2026-09-28 | Mid-tier model in the new Claude 5.5 family (`claude-sonnet-5-5`), balancing cost and frontier capability for agentic coding, tool use and knowledge work; confirmed on Anthropic's official platform release notes. |
| [Claude Opus 5.5](https://www.anthropic.com/news/claude-opus-5-5) | 2026-09-22 | Current Anthropic flagship and first model in the Claude 5.5 family (`claude-opus-5-5`); priced $4/$20 per MTok (20% below Opus 5, and ~40% cheaper per task as reported). Headline benchmarks per Anthropic: Terminal-Bench 4.0 66.4%, CursorBench 4.0 57.8%, Humanity's Last Exam w/ tools 67.7%. Thinking cannot be disabled. |

## xAI — Grok

| Item | Date | What's new |
|---|---|---|
| [Grok 4.7](https://x.ai/news/grok-4-7) | 2026-09-21 | xAI's (now "SpaceXAI") most powerful coding/knowledge-work model, 500K context, knowledge cutoff May 2026, reported $2/$6 per MTok with a 2x-speed fast variant. Reports gains on Terminal-Bench 4.0 (37.6%), CursorBench 4.0 and GDPval; per independent tests it still trails Claude Opus 5.5 / GPT-6. Official x.ai post. |
| [Grok 5](https://www.mindstudio.ai/blog/grok-5-agi-xai-model-roadmap) | 2026-10 (unreleased) | Radar-watch only. xAI's next flagship is reported still in training with no committed release date or official benchmarks (xAI shipping Grok 4.7/4.8/4.9 first). Widely cited ~6T-param MoE and target benchmark tables are leaks/targets, not measured — no primary release or model card exists yet. |

## Related

- Radar index: [`00_FRONTIER_AI_2026/README.md`](./README.md)
- Newest open-weight counterparts: [`Open_Source_LLMs_2026.md`](./Open_Source_LLMs_2026.md)
- Reasoning / agentic framing of these models: [`Reasoning_and_Agents_2026.md`](./Reasoning_and_Agents_2026.md)
- Stable model catalog: [`02_LLM_AND_AI_MODELS/Text_LLMs/Frontier_Closed_Models`](../02_LLM_AND_AI_MODELS/Text_LLMs/Frontier_Closed_Models/)
- Evaluation & benchmarks: [`01_AI_FUNDAMENTALS_AND_THEORY/AI_Evaluation`](../01_AI_FUNDAMENTALS_AND_THEORY/AI_Evaluation/)

**Sources:** OpenAI (openai.com, github.blog changelog, techcrunch.com), Google DeepMind (blog.google, ai.google.dev, GoogleDeepMind on X), Anthropic (anthropic.com, platform.claude.com release notes), xAI (x.ai) — each row links directly to the primary or earliest reporting verified at refresh time; see the per-row links above.

**Keywords:** frontier closed LLMs 2026, flagship proprietary models, OpenAI GPT-6, GPT-6 Astra, GPT-6.1 Sol, GPT-6 Sol, GPT-6 Luna, intelligent UI, Anthropic Claude 5.5, Claude Opus 5.5, Claude Sonnet 5.5, Google Gemini 4 Argon, Gemini 3.8 Flash, Gemini 3.1 Pro, xAI Grok 4.7, Grok 5, agentic coding, reasoning models, context window, API pricing, state of the art, October 2026.
