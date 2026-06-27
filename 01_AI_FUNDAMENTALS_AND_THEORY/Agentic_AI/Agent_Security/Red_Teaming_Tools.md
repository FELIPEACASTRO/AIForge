# AI Agent Red-Teaming & Guardrail Tools

Open-source and vendor tooling to **test** (red-team) and **protect** (guardrail) tool-using agents. Increasingly, open-source red-teaming engines sit *inside* vendor platforms.

## Red-teaming / attack-simulation

| Tool | What it does | Notes | Link |
|---|---|---|---|
| **Promptfoo** | CLI that throws thousands of automated attacks; scans 50+ vuln types (prompt injection, PII leakage, RBAC bypass, unauthorized tool execution). | 18k+ GitHub stars; **acquired by OpenAI (Mar 2026, ~$86M), stays MIT/open-source**. | https://www.promptfoo.dev/ |
| **PyRIT** (Microsoft) | Python Risk Identification Tool for GenAI; orchestration, converters, scoring; AI Red Teaming Agent; in Azure AI Foundry. | MIT; strong for programmatic/research red-teaming, multi-turn. | https://github.com/Azure/PyRIT |
| **Garak** (NVIDIA) | AI vulnerability scanner; ~100 attack vectors, up to 20k prompts/run; inside NeMo Evaluator. | Broad probe coverage. | https://github.com/NVIDIA/garak |
| **DeepTeam** | LLM red-teaming framework mapping to OWASP LLM/Agentic Top 10. | Framework-aligned testing. | https://www.trydeepteam.com/ |

> Comparison: Promptfoo = adaptive attack generation + ease of use + compliance mapping; PyRIT = granular programmatic orchestration for research; Garak = breadth of probes. Both Promptfoo and PyRIT support **multi-turn** attacks.

## Guardrails / runtime protection

| Tool | What it does | Link |
|---|---|---|
| **LlamaFirewall** (Meta) | Open-source guardrail system for secure agents (input/output/tool scanning, alignment checks). | https://arxiv.org/pdf/2505.03574 |
| **Llama Guard** (Meta) | Safety classifier for input/output moderation. | https://huggingface.co/meta-llama |
| **NeMo Guardrails** (NVIDIA) | Programmable rails (topical, safety, tool-use constraints). | https://github.com/NVIDIA/NeMo-Guardrails |
| **Rebuff / LLM Guard** (ProtectAI) | Prompt-injection detection & I/O sanitization. | https://github.com/protectai/llm-guard |
| **Invariant** | MCP/agent security analysis & guardrails (surfaced MCP tool poisoning). | https://invariantlabs.ai/ |

## Trend

Open-source AI red-teaming tools are becoming the **engines inside vendor offerings**: Promptfoo → OpenAI, PyRIT → Microsoft Foundry, Garak → NVIDIA NeMo Evaluator. Adopt an open engine for portability and pair it with a runtime guardrail.

## Sources
- [Promptfoo — Top open-source AI red-teaming tools 2025](https://www.promptfoo.dev/blog/top-5-open-source-ai-red-teaming-tools-2025/)
- [netguardia — Best AI red-teaming tools 2026 (Garak→Promptfoo)](https://netguardia.com/security-operations/software-tools/the-best-ai-red-teaming-tools-of-2026-from-garak-to-promptfoo/)
- [Vectra — AI red teaming tools & frameworks](https://www.vectra.ai/topics/ai-red-teaming)
- [LlamaFirewall (arXiv 2505.03574)](https://arxiv.org/pdf/2505.03574)
