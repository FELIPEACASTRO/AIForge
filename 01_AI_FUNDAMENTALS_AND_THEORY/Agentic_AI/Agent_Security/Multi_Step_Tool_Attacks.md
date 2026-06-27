# Multi-Step Tool Attacks on AI Agents

The core threat: an adversary who cannot break a single tool call instead **chains influence across many steps** of an agent's reasoning–action loop until the agent performs an unauthorized action. Because tool-using agents carry state (memory, context, prior observations), small manipulations **compound** turn over turn.

## How multi-step attacks work (defensive overview)

| Pattern | Idea (conceptual) | Why it evades simple defenses |
|---|---|---|
| **Goal drift / "salami slicing"** | Over many benign-looking inputs, an attacker incrementally redefines what the agent treats as "normal," until its constraint model has drifted enough to act against policy. | No single message is obviously malicious; per-message filters miss the cumulative shift. |
| **Tool-call chaining** | Output of one (attacker-influenced) tool becomes the trusted input of the next, escalating capability step by step. | Each call looks locally valid; the danger is in the *sequence*. |
| **Cascading attacks** | One poisoned tool/observation triggers a chain of further tool calls (e.g., read → summarize → send), exfiltrating or acting before any human sees it. | Autonomy means many steps execute between human checkpoints. |
| **Memory-triggered delay** | Inject malicious content into long-term memory in step *N*; trigger its consumption in a later, unrelated session. | Attack and effect are separated in time, breaking trajectory-based detection. |
| **Cross-tool / cross-server hijack** | A malicious tool/server overrides instructions from other trusted tools, taking over the agent's behavior. | Trust between tools is often implicit and unchecked. |

> A recurring research finding: errors and adversarial influence **accumulate across multi-step workflows**, and "self-verification" / internal feedback loops are a leading mitigation direction (the agent checks its own work before acting).

## Real-world signals (2025–2026)

- **Autonomous compromise speed:** in a controlled red-team exercise, an internal enterprise AI platform was reported compromised by an autonomous agent that gained broad system access in **under two hours** — reframing the threat model around speed. ([Lakera](https://www.lakera.ai/blog/the-year-of-the-agent-what-recent-attacks-revealed-in-q4-2025-and-what-it-means-for-2026), [eSecurity Planet](https://www.esecurityplanet.com/artificial-intelligence/ai-agent-attacks-in-q4-2025-signal-new-risks-for-2026/))
- **Agent-executed intrusions:** Anthropic described a state-sponsored campaign in which the AI model autonomously executed the majority of intrusion steps while humans acted as strategic supervisors. ([Lakera](https://www.lakera.ai/blog/the-year-of-the-agent-what-recent-attacks-revealed-in-q4-2025-and-what-it-means-for-2026))
- **Supply chain:** a registry scan across ~98,380 agent "skills" found **157 confirmed malicious entries**. ([arXiv 2603.00195](https://arxiv.org/pdf/2603.00195))
- Security research consistently ranks **MCP vulnerabilities, prompt injection, and assistant-mediated data exfiltration** as the fastest-expanding agentic attack classes. ([Shattered.io](https://shattered.io/agentic-ai-security-2026/))

## The Kaggle competition: "AI Agent Security — Multi-Step Tool Attacks"

A public benchmark directly on this topic:

- **What:** build an attack algorithm that stress-tests tool-using AI agents in a **deterministic offline benchmark**, finding reproducible multi-step failures.
- **Partners:** Kaggle in partnership with **OpenAI, Google, and IEEE**.
- **Prize / timeline:** ~$50,000 pool; entry deadline **2026-08-25** (launched ~June 2026).
- **Page:** https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks

Related public-competition research: **"Security Challenges in AI Agent Deployment: Insights from a Large-Scale Public Competition"** — https://arxiv.org/pdf/2507.20526

## Defensive takeaways

1. **Least privilege per step** — scope tool permissions to the minimal action needed (see over-privileged tool selection: https://arxiv.org/html/2606.20023v1).
2. **Trust boundaries between tools/servers** — never let one tool's output silently override instructions.
3. **Trajectory monitoring** — detect goal drift and anomalous tool-call sequences, not just single bad messages (e.g., MELON masked re-execution).
4. **Human-in-the-loop at high-impact actions** — gate exfiltration/irreversible actions.
5. **Continuous red-teaming** — use the benchmarks and tools in the sibling files.

## Sources
- [Lakera — The Year of the Agent (Q4 2025 → 2026)](https://www.lakera.ai/blog/the-year-of-the-agent-what-recent-attacks-revealed-in-q4-2025-and-what-it-means-for-2026)
- [eSecurity Planet — AI Agent Attacks in Q4 2025](https://www.esecurityplanet.com/artificial-intelligence/ai-agent-attacks-in-q4-2025-signal-new-risks-for-2026/)
- [Shattered.io — Agentic AI Security 2026](https://shattered.io/agentic-ai-security-2026/)
- [Kaggle — AI Agent Security: Multi-Step Tool Attacks](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks)
- [arXiv 2507.20526 — Security Challenges in AI Agent Deployment](https://arxiv.org/pdf/2507.20526)
- [arXiv 2603.11214 — Measuring AI Agents' Progress on Multi-Step Cyber Attack Scenarios](https://arxiv.org/html/2603.11214v2)
- [arXiv 2603.00195 — Formal Analysis and Supply Chain Security for Agentic AI Skills](https://arxiv.org/pdf/2603.00195)
