# Frameworks & Standards for Agent Security

Use these together — they are complementary, not competing. OWASP gives a developer-centric vulnerability list, MITRE ATLAS gives an adversary-centric TTP framework, and NIST/ISO give the risk-management system.

## OWASP

- **OWASP Top 10 for LLM Applications (GenAI)** — high-level critical risks: prompt injection (LLM01), insecure output handling, supply-chain, sensitive-info disclosure, excessive agency, etc. — https://genai.owasp.org/
- **OWASP Top 10 for Agentic Applications (2026)** — extends GenAI risks to autonomous/multi-agent systems, naming **agent goal hijacking, tool misuse & exploitation, memory & context poisoning**, and risks from delegation and tool integration. — [Practical DevSecOps](https://www.practical-devsecops.com/owasp-top-10-agentic-applications/) · [DeepTeam](https://www.trydeepteam.com/docs/frameworks-owasp-top-10-for-agentic-applications)
- **OWASP "Agentic AI — Threats and Mitigations" (v1.0)** — https://www.aigl.blog/content/files/2025/04/Agentic-AI---Threats-and-Mitigations.pdf

## MITRE ATLAS

Adversarial Threat Landscape for AI Systems — an ATT&CK-style knowledge base for AI:
- **16 tactics, 84 techniques, 32 mitigations, 42 case studies** (32 sub-techniques), with **agentic-AI techniques** added through Feb 2026.
- https://atlas.mitre.org/ · overview: [Vectra](https://www.vectra.ai/topics/mitre-atlas)

## NIST & ISO

- **NIST AI Risk Management Framework (AI RMF)** — govern/map/measure/manage AI risk. https://www.nist.gov/itl/ai-risk-management-framework
- **ISO/IEC 42001** — AI management system standard (the system that manages AI risk).
- **NIST AI 100-2** — Adversarial ML taxonomy & terminology.

## How they fit together

| Framework | Lens | Use it to… |
|---|---|---|
| OWASP LLM / Agentic Top 10 | Developer / vulnerability | Build securely; checklist of what to prevent. |
| MITRE ATLAS | Adversary / TTP | Threat-model, detect, and map real attacker behavior. |
| NIST AI RMF / ISO 42001 | Governance / risk | Operate a program that evaluates & treats the above. |

> Guidance for 2026: ATLAS, OWASP LLM Top 10, and NIST AI RMF are **complementary — use all three** for comprehensive coverage ([RedFox](https://www.redfoxsec.com/blog/mitre-atlas-vs-owasp-llm-top-10-which-framework-should-you-use-in-2026)).

## Sources
- [OWASP GenAI Security Project](https://genai.owasp.org/) · [OWASP Top 10 for Agentic Apps (Practical DevSecOps)](https://www.practical-devsecops.com/owasp-top-10-agentic-applications/)
- [MITRE ATLAS](https://atlas.mitre.org/) · [Vectra — MITRE ATLAS](https://www.vectra.ai/topics/mitre-atlas)
- [RedFox — ATLAS vs OWASP LLM Top 10 (2026)](https://www.redfoxsec.com/blog/mitre-atlas-vs-owasp-llm-top-10-which-framework-should-you-use-in-2026)
- [Alice — NIST/OWASP/MITRE/ISO 42001 governance](https://alice.io/blog/ai-risk-management-frameworks-nist-owasp-mitre-maestro-iso)
