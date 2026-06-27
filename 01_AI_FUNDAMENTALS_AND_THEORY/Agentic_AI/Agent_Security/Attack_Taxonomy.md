# AI Agent Attack Taxonomy

A layered view of the agentic attack surface. Attackers can compromise an agent at **every transition** of its loop: input/retrieval, memory lookup, observed web/API content, stored memory, tool definitions, and inter-agent messages.

## Privilege-escalation vectors (5 canonical)

Research on privilege escalation in LLM-agent systems identifies five distinct vectors ([arXiv 2601.11893](https://arxiv.org/html/2601.11893v1)):

1. **Direct prompt injection** — adversarial instructions in the user-facing input.
2. **Indirect prompt injection (IPI)** — malicious instructions embedded in content the agent *reads* (web pages, documents, tool outputs, emails).
3. **RAG poisoning** — corrupting the retrieval corpus so the agent retrieves attacker-controlled context.
4. **Untrusted agents** — a malicious agent in a multi-agent system.
5. **Confused deputy** — a low-privilege source manipulates a higher-privilege agent into misusing its legitimate authority.

> **Privilege escalation (definition):** any agent action beyond what is minimally required to fulfill the user's intent.

## Core attack categories

| Category | Description | Reference |
|---|---|---|
| **Direct Prompt Injection** | Override system/developer instructions via user input. | OWASP LLM01 |
| **Indirect Prompt Injection** | Instructions hidden in retrieved/observed data; the highest-impact agent threat. | [InjecAgent](https://arxiv.org/abs/2403.02691) |
| **Tool Misuse / Exploitation** | Coaxing the agent to call tools in unintended, harmful ways — acute under unconstrained autonomy and in multi-agent setups. | OWASP Agentic |
| **Memory / Context Poisoning** | Plant malicious text into memory, trigger consumption later (time-separated). | OWASP Agentic |
| **Confused Deputy** | Agent holds legitimate authority (network/tool access) but is driven by a less-privileged source. | [arXiv 2601.11893](https://arxiv.org/html/2601.11893v1) |
| **Goal/Intent Hijacking** | Shift the agent's objective away from the user's intent. | OWASP Agentic |
| **Over-Privileged Tool Selection** | Agent picks a more powerful tool than the task needs, widening blast radius. | [arXiv 2606.20023](https://arxiv.org/html/2606.20023v1) |
| **Agent-to-Agent (A2A) Threats** | Manipulating inter-agent messages/protocols in multi-agent systems. | [arXiv 2602.05877](https://arxiv.org/html/2602.05877) |
| **Implicit / Silent Exfiltration** | Data leaves the system with no obvious trace. | [arXiv 2602.22450](https://arxiv.org/pdf/2602.22450) |
| **Supply-chain (skills/tools)** | Malicious skills/tools in registries; tool squatting & rug pulls. | [arXiv 2506.01333](https://arxiv.org/pdf/2506.01333) |

## Layered attack-surface framework

A systematic survey organizes threats by **layer** — perception/input, memory, planning/reasoning, tool/action, and inter-agent communication — mapping where each attack injects and what it compromises ([arXiv 2604.23338](https://arxiv.org/html/2604.23338v2)).

## Intent classes (what the attacker wants)

- **Direct harm to the user** (destructive or unauthorized actions).
- **Private-data exfiltration** (the two primary intents formalized by InjecAgent).
- **Persistence / lateral movement** (especially multi-agent and memory-based).

## Sources
- [arXiv 2601.11893 — Taming Privilege Escalation in LLM-Based Agent Systems](https://arxiv.org/html/2601.11893v1)
- [arXiv 2604.23338 — Systematic Survey: Layered Attack-Surface Framework](https://arxiv.org/html/2604.23338v2)
- [arXiv 2606.20023 — Over-Privileged Tool Selection](https://arxiv.org/html/2606.20023v1)
- [arXiv 2602.05877 — Agent2Agent Threats: A Human-Centric Taxonomy](https://arxiv.org/html/2602.05877)
- [MDPI 2026 — Prompt Injection in LLMs and AI Agent Systems: Comprehensive Review](https://www.mdpi.com/2078-2489/17/1/54)
- [OWASP — Agentic AI: Threats and Mitigations (v1.0)](https://www.aigl.blog/content/files/2025/04/Agentic-AI---Threats-and-Mitigations.pdf)
