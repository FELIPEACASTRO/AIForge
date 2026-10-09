# 🧠 Reasoning & Agentic AI 2026 — Radar

> The defining front of 2026: reasoning models that trade latency for accuracy, and agents that take a high-level goal and run it to completion across software, browsers and long-horizon workflows.

**Refreshed:** 2026-10-09 · **Scope:** releases through October 2026

> ⚠️ Fast-moving field: entries reflect public reporting at refresh time. Confirm versions, licenses, and benchmark figures at the source links before relying on them.

## 🧩 Reasoning & agentic models

Newest first. All capability and benchmark figures are as reported by the linked source or model card, not independently verified here.

| Item | Date | What's new |
|---|---|---|
| [Google Gemini 4 Argon](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/) | 2026-09-30 | Google's new frontier reasoning model, built for deep reasoning across long-horizon workflows. Per Google, the output limit is raised to 1M tokens (hundreds of thousands of tokens per single trajectory), with a cyber-defense focus; rolling out first via the Fairwind Program. |
| [Google Gemini Spark (delegable agent for Windows)](https://blog.google/innovation-and-ai/technology/ai/google-ai-updates-september-2026/) | 2026-09-30 | An agent in the Gemini app for Windows that you can delegate multi-step tasks to; announced in Google's September 2026 AI roundup. |
| [OpenAI GPT-6.1 Sol](https://openai.com/index/introducing-gpt-6-1-sol/) | 2026-09-29 | Near-Astra agentic coding and computer-use reasoning model at roughly 1/5 the price (API `gpt-6.1-sol`). As reported, it posts strong DeepSWE v1.1 and OSWorld 2.0 results; launched at DevDay 2026. |
| [OpenAI GPT-6 Sol and Luna](https://openai.com/index/introducing-gpt-6-sol-and-luna/) | 2026-09-22 | Mid- and low-tier members of the GPT-6 reasoning family (API `gpt-6-sol` / `gpt-6-luna`), extending GPT-6 to cheaper, faster workloads. |
| [xAI Grok 4.7 + Grok Build](https://github.blog/changelog/2026-09-21-grok-4-7-is-now-available-in-github-copilot/) | 2026-09-21 | xAI reasoning model trained for problems that take many hours (longer RL run, stronger self-verification, Low/Med/High/XHigh reasoning levels, 500K context), shipped alongside Grok Build, a terminal-native agentic coding agent. |
| [Google Gemini 3.8 Flash & 3.8 Flash Cyber](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/) | 2026-09-02 | Fast model tuned for agentic workflows (3.8 Flash), plus a 3.8 Flash Cyber variant aimed at autonomous vulnerability discovery and code patching. |
| [DeepSeek V4 family (V4-Pro GA / V4.1-Flash)](https://api-docs.deepseek.com/news/news260424/) | 2026-08-13 | Open-weight MoE reasoning line with DeepSeek Sparse Attention and token-wise compression, 1M default context and Thinking / Non-Thinking modes. Per DeepSeek, V4-Pro (1.6T / 49B active) reached GA on Aug 13 2026 and V4.1-Flash added native multimodal on Sep 10 2026 (preview-announcement URL shown). |
| [Moonshot Kimi K3](https://github.com/MoonshotAI/Kimi-K3) | 2026-07-27 | Open-weight native-multimodal agentic model: 2.8T total / 104B active MoE, 1M context, MXFP4 quantization-aware training, aimed at long-horizon coding and reasoning; released with a technical report and training infra (MoonEP, FlashKDA, AgentEnv). |
| [Google Gemini 3 Deep Think (Feb 2026 upgrade)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/) | 2026-02-12 | Parallel-hypothesis test-time-compute reasoning mode, upgraded for science/research/engineering and opened to select API users; earlier benchmarks cited ARC-AGI-2 45.1% with code execution. Older, but a landmark test-time-compute mode. |

## 🛠️ Agent platforms, frameworks & always-on agents

Newest first.

| Item | Date | What's new |
|---|---|---|
| [OpenAI Dots (always-on agents) + Agents API / ChatGPT Space](https://openai.com/index/devday-2026-recap/) | 2026-09-29 | Persistent "always-on" personal agents powered by GPT-6 Astra, each with its own cloud computer, 4,000+ app connectors, Slack/Teams channels and background long-horizon work. DevDay 2026 also shipped an Agents API (public beta) with hosted computer use, plus ChatGPT Space for human + agent collaboration. |
| [Google Agent Development Kit (ADK) 2.0](https://adk.dev/2.0/) | 2026-08-21 | Agent framework re-architected from a hierarchical executor into a graph execution engine (agents / tools / joins as nodes), with durable pause/resume, built-in human-in-the-loop and native A2A. Per the ADK docs: Python GA May 19, Go GA Jun 30, TypeScript GA Aug 21 2026. |
| [Microsoft Agent Framework 1.0](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/the-future-of-agentic-ai-inside-microsoft-agent-framework-1-0/4510698) | 2026-04-03 | GA framework unifying AutoGen and Semantic Kernel into one .NET/Python SDK with graph-based multi-agent workflows, native MCP + A2A support, a DevUI debugger, OpenTelemetry observability and LTS. Older, but the current best-in-class for the Microsoft stack. |

## 🔌 Protocols & interoperability

Newest first.

| Item | Date | What's new |
|---|---|---|
| [Model Context Protocol — 2026-07-28 specification](https://blog.modelcontextprotocol.io/posts/2026-07-28/) | 2026-07-28 | The largest MCP revision yet: a stateless core (initialize handshake and session-id removed), Multi Round-Trip Requests replacing held-open sampling/elicitation/roots, header-based routing (`Mcp-Method` / `Mcp-Name`), Tasks moved to an extension, MCP Apps, cacheable lists, and auth hardening. |
| [Agent2Agent (A2A) protocol — one-year update](https://opensource.googleblog.com/2026/04/a-year-of-open-collaboration-celebrating-the-anniversary-of-a2a.html) | 2026-04-30 | Maturation of the vendor-neutral agent-to-agent interoperability standard under the Linux Foundation. Per the anniversary post: 150+ orgs and production use in Azure AI Foundry, Bedrock AgentCore and Agentforce. Older, but the backbone agent-interop standard. |

## 📏 Long-horizon autonomy & agent research

Newest first.

| Item | Date | What's new |
|---|---|---|
| [A2ABreak — systematic security analysis of the A2A protocol](https://arxiv.org/abs/2609.10871) | 2026-09-09 | First rigorous systematic security analysis of the Agent2Agent (A2A) multi-agent protocol. Per the paper, an LLM-extracted finite-state machine plus model checking surfaces 11 spec-level vulnerabilities exploitable without any implementation flaw. |
| [OneDayAgent — long-horizon harness for autonomous agents](https://arxiv.org/abs/2608.05013) | 2026-08-04 | A harness for open-ended, multi-step autonomous agents (subtask decomposition, execution-state tracking under context pressure, self-verification). As reported: SOTA 0.821 on the 104-task AgentIF-OneDay benchmark with GLM-5.2, model-agnostic across 5 backends. |
| [METR Time Horizon 1.1 (+ May 2026 Frontier Risk Report)](https://metr.org/blog/2026-1-29-time-horizon-1-1/) | 2026-01-29 | Updated long-horizon autonomy measurement: the suite expanded ~34% with more 8h+ tasks. Per METR, the May 19 2026 Frontier Risk Report fits a ~105-day time-horizon doubling for post-2024 frontier models and flags the suite nearing a ~16h ceiling. Older, but the key autonomy metric. |

## Reading the radar

- **Reasoning is now the default, not a mode.** The newest flagships (Gemini 4 Argon, GPT-6 Sol/Luna, GPT-6.1 Sol, Grok 4.7) are trained for long, self-verifying chains and tiered reasoning effort rather than offering reasoning as an opt-in.
- **Agents are moving to "always-on."** OpenAI's Dots and Google's Gemini Spark reframe the agent as a persistent, delegable worker with its own compute and connectors, not a single-request call.
- **The plumbing is standardizing.** The stateless MCP 2026-07-28 spec and the one-year A2A update are converging the ecosystem on shared tool-context and agent-to-agent interop — and that surface is now itself a security-research target (A2ABreak).
- **Autonomy is being measured, not just claimed.** METR's time-horizon work and harnesses like OneDayAgent put numbers on how long agents can run unattended before errors compound.

## Related

- Radar index: [`00_FRONTIER_AI_2026/README.md`](./README.md) · sibling pages: [Frontier_LLMs_2026.md](./Frontier_LLMs_2026.md) · [AI_Coding_Tools_2026.md](./AI_Coding_Tools_2026.md) · [Open_Source_LLMs_2026.md](./Open_Source_LLMs_2026.md)
- Stable pillars: [`02_LLM_AND_AI_MODELS/Reasoning_Models`](../02_LLM_AND_AI_MODELS/Reasoning_Models/) · [`01_AI_FUNDAMENTALS_AND_THEORY/Agentic_AI`](../01_AI_FUNDAMENTALS_AND_THEORY/Agentic_AI/) · [`01_AI_FUNDAMENTALS_AND_THEORY/Test_Time_Compute`](../01_AI_FUNDAMENTALS_AND_THEORY/Test_Time_Compute/) · [`04_MLOPS_AND_PRODUCTION_AI/AI_Agents`](../04_MLOPS_AND_PRODUCTION_AI/AI_Agents/)

**Sources:** per-row links above — OpenAI ([GPT-6.1 Sol](https://openai.com/index/introducing-gpt-6-1-sol/), [GPT-6 Sol and Luna](https://openai.com/index/introducing-gpt-6-sol-and-luna/), [DevDay 2026 recap](https://openai.com/index/devday-2026-recap/)), Google ([Gemini 4 Argon](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/), [Gemini 3.8 Flash & Cyber](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/), [Gemini 3 Deep Think](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/), [Gemini Spark / Sept 2026 roundup](https://blog.google/innovation-and-ai/technology/ai/google-ai-updates-september-2026/)), [GitHub Changelog (Grok 4.7)](https://github.blog/changelog/2026-09-21-grok-4-7-is-now-available-in-github-copilot/), [DeepSeek API docs](https://api-docs.deepseek.com/news/news260424/), [Moonshot Kimi-K3 (GitHub)](https://github.com/MoonshotAI/Kimi-K3), [MCP blog](https://blog.modelcontextprotocol.io/posts/2026-07-28/), [Google ADK 2.0](https://adk.dev/2.0/), [Microsoft Agent Framework 1.0](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/the-future-of-agentic-ai-inside-microsoft-agent-framework-1-0/4510698), [A2A anniversary (Google Open Source)](https://opensource.googleblog.com/2026/04/a-year-of-open-collaboration-celebrating-the-anniversary-of-a2a.html), [A2ABreak (arXiv)](https://arxiv.org/abs/2609.10871), [OneDayAgent (arXiv)](https://arxiv.org/abs/2608.05013), [METR Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/).

**Keywords:** reasoning models 2026, agentic AI, test-time compute, self-verification, long-horizon autonomy, GPT-6 Sol, GPT-6 Luna, GPT-6.1 Sol, OpenAI Dots always-on agents, Agents API, Gemini 4 Argon, Gemini 3.8 Flash, Gemini 3 Deep Think, Gemini Spark, Grok 4.7, Grok Build, DeepSeek V4, Moonshot Kimi K3, Model Context Protocol, MCP 2026-07-28, Agent2Agent A2A, A2ABreak, Google ADK 2.0, Microsoft Agent Framework, multi-agent, computer-use agents, browser agents, METR time horizon, OneDayAgent, October 2026.
