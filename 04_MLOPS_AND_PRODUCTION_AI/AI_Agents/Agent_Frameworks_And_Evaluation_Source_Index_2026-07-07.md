# Agent Frameworks And Evaluation Source Index - 2026-07-07

This index organizes sources for building and evaluating AI agents. It separates framework docs, multi-agent orchestration, tool standards, observability, and benchmark sources so agent projects can be evaluated as systems rather than single prompts.

## Agent Frameworks And SDKs

| Source | What to capture | Local routing |
|---|---|---|
| [OpenAI Agents SDK guide](https://developers.openai.com/api/docs/guides/agents) | Code-first agents, tools, handoffs, sessions, guardrails, tracing, and runtime patterns. | `Agent_Frameworks/`, guardrails, tracing |
| [OpenAI Agents SDK Python docs](https://openai.github.io/openai-agents-python/) | Python SDK primitives, runners, sessions, guardrails, tracing, and examples. | Implementation notes and examples |
| [OpenAI Agents SDK GitHub](https://github.com/openai/openai-agents-python) | Source code, releases, issues, examples, and license. | Framework provenance |
| [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents) | Agent loop, model/tool harness, middleware, and configurable agent runtime. | `Agent_Frameworks/` |
| [LangGraph overview](https://langchain-ai.github.io/langgraph/) | Graph-based stateful orchestration for controllable agent workflows. | `Multi_Agent_Systems/`, workflow orchestration |
| [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/) | Microsoft agent runtime combining AutoGen-style abstractions and Semantic Kernel enterprise features. | Microsoft/platform routing |
| [AutoGen stable docs](https://microsoft.github.io/autogen/stable//index.html) | Multi-agent applications, AgentChat, Core, Studio, and extensions. | Legacy and research agent notes |
| [AutoGen Microsoft Research](https://www.microsoft.com/en-us/research/project/autogen/) | Research project background and publications. | Primary-source research provenance |
| [CrewAI docs](https://docs.crewai.com/) | Crews, flows, memory, knowledge, guardrails, and observability for multi-agent systems. | `Agent_Frameworks/`, business automation |
| [Semantic Kernel overview](https://learn.microsoft.com/en-us/semantic-kernel/overview/) | Plugins, planners, memory, connectors, and enterprise app integration. | Agent framework and .NET/Python integration |
| [LlamaIndex agents](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/) | Agents over data, tools, workflow, and retrieval-centered systems. | RAG agents and data agents |
| [Pydantic AI](https://ai.pydantic.dev/) | Typed agent framework with structured outputs and validation. | Typed outputs and Python app agents |
| [Google Agent Development Kit](https://google.github.io/adk-docs/) | Google ADK for agent development and deployment patterns. | Google ecosystem agent notes |
| [Model Context Protocol](https://modelcontextprotocol.io/) | Tool/context interoperability standard for connecting AI apps to external systems. | Tool standards and integration patterns |

## Agent Evaluation And Observability

| Source | Scope | Evidence to record |
|---|---|---|
| [LangSmith evaluation](https://docs.smith.langchain.com/evaluation) | Dataset-driven evals, traces, experiments, and regressions. | Dataset, evaluator, trace ID, prompt/model version, and pass criteria. |
| [Langfuse](https://langfuse.com/docs) | LLM observability, tracing, prompt management, scores, and datasets. | Trace schema, retention, metadata, scoring, and privacy policy. |
| [Ragas](https://docs.ragas.io/) | RAG and LLM application evaluation. | Question set, retrieved context, answer, faithfulness, and model judge settings. |
| [DeepEval](https://docs.confident-ai.com/) | Unit tests and metrics for LLM apps. | Metric prompt, threshold, dataset, and CI result. |
| [OpenAI Evals](https://github.com/openai/evals) | Custom evaluation harness and graders. | Eval spec, samples, grading model, and reproducibility. |
| [Promptfoo](https://www.promptfoo.dev/) | Prompt/provider matrix testing and red-team evals. | Assertions, providers, attack plugins, and regression history. |

## Agent Benchmarks

| Source | Scope | Caveat |
|---|---|---|
| [SWE-bench](https://www.swebench.com/) | Software-engineering agents on real GitHub issues. | Scores depend heavily on scaffold, tools, budget, and retry policy. |
| [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) | General assistant tasks requiring reasoning, browsing, and tool use. | Verify allowed tools and split. |
| [AgentBench](https://github.com/THUDM/AgentBench) | Multi-environment benchmark for LLM agents. | Environment setup can dominate results. |
| [tau-bench](https://github.com/sierra-research/tau-bench) | Tool-agent benchmark for realistic user interactions and policies. | Track policy, simulator, and tool-call constraints. |
| [OSWorld](https://github.com/xlang-ai/OSWorld) | Computer-use agents in real operating-system tasks. | Requires careful environment and safety isolation. |
| [WebArena](https://github.com/web-arena-x/webarena) | Web agent benchmark with realistic websites. | Requires deployed environment and task-specific accounts. |

## Production Readiness Checklist

- Agent scope, allowed tools, permissions, identity, memory boundaries, state retention, and human approval gates.
- Tool schema, authentication, sandboxing, network access, filesystem access, and rollback behavior.
- Prompt/context version, retrieval source, model/provider, tracing, cost controls, timeout, retry policy, and fallback.
- Eval suite covering success, safety, tool misuse, prompt injection, data leakage, latency, cost, and regression.
- Incident playbook for runaway loops, unsafe tool calls, stale context, hidden state, and compromised tool outputs.
