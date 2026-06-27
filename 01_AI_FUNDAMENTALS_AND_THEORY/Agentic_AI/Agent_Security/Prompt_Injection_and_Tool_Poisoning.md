# Prompt Injection & Tool Poisoning

The dominant agent threat. **Indirect prompt injection (IPI)** — malicious instructions embedded in content the agent processes — is uniquely dangerous for tool-using agents because the "data" an agent reads can become "instructions" it follows.

## Direct vs Indirect

- **Direct prompt injection** — the attacker controls the user-facing prompt and tries to override system/developer instructions.
- **Indirect prompt injection (IPI)** — the attacker plants instructions in *third-party content* the agent later reads: a web page, a PDF, a tool's output, an email, a calendar invite, a code comment. The user never typed anything malicious. This is the highest-impact class for agents that browse, retrieve, or read tool outputs.

> Benchmarks quantify the risk: ReAct-prompted GPT-4 was vulnerable to IPI **~24%** of the time on InjecAgent's 1,054 cases. Adaptive attacks have been shown to **break many proposed defenses**, so layered defense is essential ([arXiv 2503.00061](https://arxiv.org/pdf/2503.00061)).

## MCP & Tool Poisoning

The **Model Context Protocol (MCP)** standardizes how LLMs connect to external tools/data — and expands the attack surface, especially client-side.

| Attack | What it is | Reference |
|---|---|---|
| **Tool poisoning** | Malicious instructions hidden in **tool *metadata*/descriptions** (not visible to the user) that the model reads and obeys. | [Invariant Labs](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks) |
| **Cross-server hijack** | A malicious MCP server overrides instructions from other *trusted* servers, fully compromising agent behavior. | [arXiv 2603.22489](https://arxiv.org/html/2603.22489v1) |
| **Tool squatting** | Registering tools with names/identities impersonating legitimate ones. | [ETDI, arXiv 2506.01333](https://arxiv.org/pdf/2506.01333) |
| **Rug pull** | A tool behaves benignly during review, then changes behavior after being trusted/installed. | [ETDI, arXiv 2506.01333](https://arxiv.org/pdf/2506.01333) |
| **Credential aggregation** | MCP servers hold OAuth tokens for many services — a single compromise grants cross-service access. | [SentinelOne](https://www.sentinelone.com/cybersecurity-101/cybersecurity/mcp-security/) |

Tool-poisoning attacks target the **cognitive process** of the model rather than code-level flaws, making them hard to catch with conventional scanners; research reports high success rates with safety alignment providing minimal protection.

## Why agentic coding assistants are a hot target

Coding agents read repos, issues, and dependencies — all attacker-influenceable. Systematic analyses show injection via **skills, tools, and protocol ecosystems** ([arXiv 2601.17548](https://arxiv.org/html/2601.17548v1)), and **query-agnostic IPI** on coding agents ([QueryIPI, arXiv 2510.23675](https://arxiv.org/pdf/2510.23675)).

## Defensive levers (see Defenses_and_Mitigations.md)
- Treat **all** tool output / retrieved content as untrusted data (Spotlighting / data-marking).
- Verify tool identity & integrity (signed tool definitions, OAuth-enhanced — ETDI).
- Isolate privileged actions from untrusted-data processing (CaMeL dual-LLM, Meta's Rule of Two).

## Sources
- [InjecAgent (arXiv 2403.02691)](https://arxiv.org/abs/2403.02691) · [Adaptive Attacks Break Defenses (arXiv 2503.00061)](https://arxiv.org/pdf/2503.00061)
- [Invariant Labs — MCP Tool Poisoning](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks)
- [arXiv 2603.22489 — MCP Threat Modeling & Tool Poisoning](https://arxiv.org/html/2603.22489v1) · [JCP 2026 DOI](https://doi.org/10.3390/jcp6030084)
- [ETDI (arXiv 2506.01333) — Tool Squatting & Rug Pulls](https://arxiv.org/pdf/2506.01333)
- [SentinelOne — MCP Security Guide](https://www.sentinelone.com/cybersecurity-101/cybersecurity/mcp-security/)
- [arXiv 2601.17548 — Prompt Injection on Agentic Coding Assistants](https://arxiv.org/html/2601.17548v1)
