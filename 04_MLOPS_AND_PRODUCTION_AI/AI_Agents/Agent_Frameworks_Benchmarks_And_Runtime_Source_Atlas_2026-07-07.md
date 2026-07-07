# Agent Frameworks Benchmarks And Runtime Source Atlas - 2026-07-07

This atlas expands the agent directory with primary sources for agent frameworks, tool protocols, computer-use environments, software-engineering agents, evaluation harnesses, and runtime operations. Agent claims should be tied to tasks, tools, execution traces, and evaluator definitions.

## Agent Frameworks And Protocols

| Source | What to capture | Local routing |
|---|---|---|
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Agents, tools, handoffs, tracing, guardrails, and Python runtime patterns. | `AI_Agents/`, `LLMOps_and_Prompt_Management/` |
| [Model Context Protocol](https://modelcontextprotocol.io/introduction) | Client/server protocol, tools, resources, prompts, transports, and integration patterns. | Agent tooling and API integration. |
| [LangGraph documentation](https://langchain-ai.github.io/langgraph/) | Graph-based agent workflows, durable execution, state, interrupts, and human-in-the-loop. | `02_OPEN_SOURCE_AGENTS/`, workflow agents. |
| [Microsoft AutoGen documentation](https://microsoft.github.io/autogen/) | Multi-agent conversations, tools, teams, and agent runtime components. | `02_OPEN_SOURCE_AGENTS/` |
| [CrewAI documentation](https://docs.crewai.com/) | Role-based multi-agent orchestration, crews, flows, tools, and deployment. | `02_OPEN_SOURCE_AGENTS/` |
| [LlamaIndex agents documentation](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/) | Data-aware agents, tools, workflows, and retrieval-connected systems. | RAG agents and retrieval agents. |
| [Pydantic AI documentation](https://ai.pydantic.dev/) | Typed agent outputs, tool calling, dependency injection, and model abstraction. | Structured-output and engineering agents. |
| [Semantic Kernel Agent Framework](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/) | Agent abstractions, plugins, orchestration, and Microsoft ecosystem integration. | Enterprise agent frameworks. |
| [Google Agent Development Kit](https://google.github.io/adk-docs/) | Agent construction, tools, orchestration, evaluation, and deployment patterns. | Google/enterprise agent routes. |
| [Hugging Face smolagents](https://huggingface.co/docs/smolagents/en/index) | Lightweight code agents, tool use, model integrations, and examples. | Open-source agent prototypes. |

## Agent Benchmarks And Environments

| Source | What to capture | Capability slice |
|---|---|---|
| [SWE-bench](https://www.swebench.com/) | Software issue resolution, verified subsets, harness details, and leaderboard context. | Coding agents. |
| [WebArena](https://webarena.dev/) | Browser-based web tasks, environments, and evaluation setup. | Web agents. |
| [OSWorld](https://os-world.github.io/) | Computer-use tasks across desktop applications and operating-system interactions. | GUI/computer-use agents. |
| [tau-bench](https://github.com/sierra-research/tau-bench) | Real-world customer-service style agent evaluation with policies and tools. | Tool-use and policy-following agents. |
| [AgentBench](https://llmbench.ai/agentbench/) | Multi-environment LLM-as-agent benchmark categories. | General agent capability. |
| [ToolBench](https://github.com/OpenBMB/ToolBench) | Tool-use datasets, evaluation, and API-call planning. | Tool-use agents. |
| [Terminal-Bench](https://www.tbench.ai/) | Terminal-oriented tasks and command-line agent evaluation. | Shell and developer agents. |
| [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard) | General-assistant questions requiring reasoning, tools, and multimodal evidence. | General assistant agents. |

## Agent Evidence Requirements

| Field | Requirement |
|---|---|
| Task model | Goal, environment, allowed tools, hidden state, interaction budget, and success criteria. |
| Trace evidence | Tool calls, observations, intermediate artifacts, final answer, and failure mode. |
| Evaluator | Benchmark version, scoring script, subset, metric, and contamination controls. |
| Runtime | Model id, framework, sandbox, permissions, browser/computer access, retries, and cost. |
| Safety | Prompt injection exposure, data exfiltration risk, destructive actions, and human approval boundaries. |

## Routing Rules

- Put frameworks and protocols in `AI_Agents/02_OPEN_SOURCE_AGENTS/` unless they are commercial managed products.
- Put agent benchmarks under `Datasets/Benchmarks/` and cross-link back here.
- Put production traces, monitoring, prompt versioning, and cost controls under `LLMOps_and_Prompt_Management/` and `AI_Observability/`.
- Treat benchmark leaderboard scores as time-sensitive; preserve source date and evaluator version.
