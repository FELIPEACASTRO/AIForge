# AI Agent Security — Multi-Step Tool Attacks

> **Defensive security & research resource.** This section catalogs the public research, benchmarks, frameworks, and defenses around the security of **tool-using AI agents** — with a focus on **multi-step tool attacks**. It is intended for builders, red/blue teams, and researchers hardening agentic systems. It describes attack *classes* and *threat models* for defense; it does not provide operational exploit payloads.

As of 2026, agentic AI (LLMs that plan, call tools, and act autonomously over multiple steps) is the fastest-growing AI deployment pattern — and its fastest-growing attack surface. Security research consistently lists **prompt injection**, **MCP/tool vulnerabilities**, and **data exfiltration through AI assistants** as the three fastest-expanding agentic attack classes, while Gartner projects 40% of enterprise apps will embed task-specific agents by 2026 (up from <5% in 2025).

## Why "multi-step" is the hard part

A single tool call is easy to reason about. An agent that chains **retrieval → reasoning → tool call → observation → next tool call** over many turns can be compromised at *every* transition: injection at retrieval or memory lookup, manipulation of observed web/API content, poisoning of stored memories, and interception of inter-agent messages. Errors and adversarial drift **compound** across steps — which is exactly what multi-step attacks exploit.

## Contents

| File | Topic |
|---|---|
| [Multi_Step_Tool_Attacks.md](./Multi_Step_Tool_Attacks.md) | **Core topic** — salami-slicing/goal-drift, tool-call chaining, cascading attacks, real incidents, the Kaggle "Multi-Step Tool Attacks" competition. |
| [Attack_Taxonomy.md](./Attack_Taxonomy.md) | Threat taxonomy: prompt injection (direct/indirect), tool misuse, memory/context poisoning, confused deputy, privilege escalation, RAG poisoning, agent-to-agent. |
| [Prompt_Injection_and_Tool_Poisoning.md](./Prompt_Injection_and_Tool_Poisoning.md) | Direct vs indirect prompt injection; MCP tool poisoning, tool squatting, rug pulls. |
| [Defenses_and_Mitigations.md](./Defenses_and_Mitigations.md) | CaMeL, Spotlighting, Progent, MELON, Meta's Rule of Two, guardrails, design patterns. |
| [Benchmarks_and_Evaluation.md](./Benchmarks_and_Evaluation.md) | InjecAgent, AgentDojo, Agent Security Bench (ASB), and how to evaluate. |
| [Frameworks_and_Standards.md](./Frameworks_and_Standards.md) | OWASP Top 10 (LLM + Agentic 2026), MITRE ATLAS, NIST AI RMF, ISO/IEC 42001. |
| [Red_Teaming_Tools.md](./Red_Teaming_Tools.md) | Promptfoo, PyRIT, Garak, DeepTeam, LlamaFirewall, NeMo Guardrails, Llama Guard. |
| [Key_Papers_and_Surveys.md](./Key_Papers_and_Surveys.md) | The essential survey + primary-source reading list. |
| [Incidents_and_CVEs.md](./Incidents_and_CVEs.md) | **Real-world** disclosed CVEs & exploits (EchoLeak, MCP RCEs, malicious MCP servers, CamoLeak, AI-orchestrated intrusion). |
| [Verified_Sources_2026.md](./Verified_Sources_2026.md) | Expanded, adversarially-verified source list (67+ benchmarks/papers/standards/tools/vendor docs). |

## Cross-references in AIForge
- Agentic AI fundamentals: [`../`](../)
- AI Safety & Alignment: [`../../AI_Safety_and_Alignment`](../../AI_Safety_and_Alignment/)
- Applied cybersecurity: [`../../../05_VERTICAL_APPLICATIONS/14_Cybersecurity_AI`](../../../05_VERTICAL_APPLICATIONS/14_Cybersecurity_AI/)
- Privacy & robustness: [`../../Privacy_and_Security`](../../Privacy_and_Security/)

## Responsible-use note
All material here is public, defensive, and research-oriented (benchmarks, surveys, frameworks, mitigations). Use it to **evaluate and harden** systems you own or are authorized to test. Production guardrails should combine architecture (least privilege, isolation) with runtime defenses and continuous red-teaming.
