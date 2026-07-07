# AI Agents

This directory covers production agent systems: planning, tool use, memory, retrieval, orchestration, evaluation, safety, observability, and human-in-the-loop controls.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Agent_Frameworks/` | OpenAI Agents SDK, LangChain/LangGraph, LlamaIndex, CrewAI, AutoGen, Semantic Kernel, and related frameworks. |
| `Multi_Agent_Systems/` | Agent collaboration, routing, delegation, debate, tool specialization, and role design. |

## Production Requirements

- Explicit tools, permissions, and state boundaries.
- Prompt and context versioning.
- Traceability for tool calls and decisions.
- Evals for task completion, safety, cost, latency, and regression.
- Guardrails against prompt injection, unsafe tool use, data leakage, and runaway loops.

## Source Families

Use OpenAI Agents SDK, Model Context Protocol, LangChain, LangGraph, LlamaIndex, Promptfoo, Ragas, and production observability tools as primary source families.
