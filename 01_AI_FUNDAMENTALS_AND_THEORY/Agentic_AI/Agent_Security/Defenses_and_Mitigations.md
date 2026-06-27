# Defenses & Mitigations for Agent Security

No single defense is sufficient — adaptive attacks break individual defenses ([arXiv 2503.00061](https://arxiv.org/pdf/2503.00061)). Combine **architecture** (least privilege, isolation), **runtime** (filtering, monitoring), and **process** (continuous red-teaming).

## Architectural / design-pattern defenses

| Defense | Idea | Reference |
|---|---|---|
| **CaMeL** (Google DeepMind, Apr 2025) | Treat the LLM as untrusted: a **Privileged LLM** handles trusted commands and a **Quarantined LLM** processes untrusted data with no memory and no ability to act. | [arXiv 2503.18813](https://arxiv.org/abs/2503.18813) |
| **Spotlighting** (Microsoft) | Mark untrusted data via delimiting / datamarking / encoding so the model treats it as *data, not instructions*. | [Microsoft research](https://arxiv.org/abs/2403.14720) |
| **Meta's "Rule of Two"** | Don't let an agent simultaneously have: untrusted input + sensitive data + external communication. Drop one leg. | [Meta AI](https://ai.meta.com/blog/) |
| **Progent** | Programmable privilege control for tool use; reported attack success **41.2% → 2.2%**. | [arXiv 2504.11703](https://arxiv.org/abs/2504.11703) |
| **Least privilege / MAC** | Mandatory Access Control for agent tool permissions; minimize over-privileged tool selection. | [arXiv 2601.11893](https://arxiv.org/html/2601.11893v1) |
| **ETDI** | OAuth-enhanced, signed tool definitions + policy-based access control to stop tool squatting & rug pulls. | [arXiv 2506.01333](https://arxiv.org/pdf/2506.01333) |

## Runtime / detection defenses

| Defense | Idea | Reference |
|---|---|---|
| **MELON** | Masked re-execution to detect trajectory manipulation. | [arXiv (MELON)](https://arxiv.org/abs/2502.05174) |
| **AgentSentry** | Temporal causal diagnostics + context purification against IPI. | [arXiv 2602.22724](https://arxiv.org/pdf/2602.22724) |
| **Tool-result parsing defense** | Structurally parse tool results to strip injected instructions. | [arXiv 2601.04795](https://arxiv.org/html/2601.04795) |
| **LlamaFirewall** | Open-source guardrail system for secure agents (input/output/tool scanning). | [arXiv 2505.03574](https://arxiv.org/pdf/2505.03574) |
| **Llama Guard / NeMo Guardrails** | Runtime content moderation & programmable rails. | Meta / NVIDIA |
| **Self-verification loops** | Agent checks its own work before high-impact actions (a leading 2026 mitigation direction). | survey |

## Lessons from defending production agents

Google's writeup on **defending Gemini against indirect prompt injection** stresses **layered defense + adaptive evaluation** (static defenses degrade under adaptive attackers) ([arXiv 2505.14534](https://arxiv.org/pdf/2505.14534)).

## A practical defense checklist

1. **Isolate** untrusted-data processing from privileged action (CaMeL / Rule of Two).
2. **Least privilege** per tool, per step; deny over-privileged tool selection.
3. **Verify tool identity/integrity** (signed definitions, OAuth — ETDI); pin trusted MCP servers.
4. **Treat all retrieved/tool content as data** (Spotlighting / data-marking).
5. **Monitor trajectories** for goal drift & anomalous tool-call sequences (MELON-style).
6. **Gate irreversible/exfiltration actions** behind human approval.
7. **Red-team continuously** with the benchmarks & tools in this section.

## Sources
- [CaMeL (arXiv 2503.18813)](https://arxiv.org/abs/2503.18813) · [Spotlighting (arXiv 2403.14720)](https://arxiv.org/abs/2403.14720)
- [Progent (arXiv 2504.11703)](https://arxiv.org/abs/2504.11703) · [LlamaFirewall (arXiv 2505.03574)](https://arxiv.org/pdf/2505.03574)
- [Defending Gemini against IPI (arXiv 2505.14534)](https://arxiv.org/pdf/2505.14534)
- [AgentSentry (arXiv 2602.22724)](https://arxiv.org/pdf/2602.22724) · [Tool-Result Parsing (arXiv 2601.04795)](https://arxiv.org/html/2601.04795)
- [awesome-ipi-defense list](https://awesome.ecosyste.ms/lists/rem1l/awesome-ipi-defense)
