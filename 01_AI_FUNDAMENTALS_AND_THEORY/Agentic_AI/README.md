# Agentic AI

Resources for **Agentic AI** — LLM-driven autonomous agents that plan, act, use tools, and complete multi-step tasks. Covers reasoning loops (ReAct, Reflection), tool use, multi-agent orchestration, browser/computer use, and production agent frameworks.

## Core Paradigms

| Paradigm | Description | Key Refs |
|---|---|---|
| **ReAct** | Reason + Act interleaved loop with tool calls | Yao et al. 2022 — https://arxiv.org/abs/2210.03629 |
| **Reflexion** | Verbal RL — agents critique own outputs to improve | Shinn et al. 2023 — https://arxiv.org/abs/2303.11366 |
| **Tree of Thoughts** | Search-based deliberate reasoning | Yao et al. 2023 — https://arxiv.org/abs/2305.10601 |
| **Tool Use / Function Calling** | Native tool invocation (OpenAI / Anthropic / Gemini) | API docs |
| **Computer Use / Browser Use** | Agents that drive UIs (Anthropic Computer Use, OpenAI Operator) | https://www.anthropic.com/news/3-5-models-and-computer-use |
| **Multi-Agent Systems** | Specialized agents collaborating (AutoGen, CrewAI, MetaGPT) | https://arxiv.org/abs/2308.10848 |
| **Memory & World Models** | Long-term memory, episodic recall, generative world models | https://arxiv.org/abs/2305.10250 |

## Production Frameworks

- **LangGraph** — https://www.langchain.com/langgraph
- **LlamaIndex Agents** — https://www.llamaindex.ai/
- **AutoGen (Microsoft)** — https://github.com/microsoft/autogen
- **CrewAI** — https://www.crewai.com/
- **OpenAI Swarm / Agents SDK** — https://github.com/openai/openai-agents-python
- **Anthropic Claude Agent SDK** — https://docs.claude.com/en/api/agent-sdk
- **Pydantic AI** — https://ai.pydantic.dev/
- **Phidata** — https://www.phidata.com/
- **DSPy** — https://github.com/stanfordnlp/dspy
- **Letta (MemGPT)** — https://github.com/letta-ai/letta
- **Smolagents (HF)** — https://github.com/huggingface/smolagents

## Protocols and Standards

- **MCP (Model Context Protocol)** — https://modelcontextprotocol.io/
- **OpenAPI for Tools / Plugins** — https://platform.openai.com/docs/plugins
- **Anthropic Tool Use** — https://docs.anthropic.com/en/docs/build-with-claude/tool-use

## Benchmarks

- **SWE-bench / SWE-bench Verified** — https://www.swebench.com/
- **WebArena** — https://webarena.dev/
- **GAIA** — https://huggingface.co/gaia-benchmark
- **TAU-Bench** — https://github.com/sierra-research/tau-bench
- **AgentBench** — https://github.com/THUDM/AgentBench
- **OSWorld** — https://os-world.github.io/

## Key Surveys

- **A Survey on LLM-based Autonomous Agents** — Wang et al. 2023 — https://arxiv.org/abs/2308.11432
- **The Rise and Potential of LLM-based Agents** — Xi et al. 2023 — https://arxiv.org/abs/2309.07864
- **Agentic Information Retrieval** — Zhang et al. 2024 — https://arxiv.org/abs/2410.09713
