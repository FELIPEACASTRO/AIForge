# Benchmarks & Evaluation for Agent Security

How to measure whether a tool-using agent resists multi-step attacks. Use a **dynamic** environment (adaptive attacks) plus **static** test suites for regression.

## Core benchmarks

| Benchmark | What it measures | Scale / notes | Link |
|---|---|---|---|
| **InjecAgent** | Indirect prompt injection on tool-integrated agents; intents = direct harm + data exfiltration. | 1,054 test cases · 17 user tools · 62 attacker tools. ReAct GPT-4 ~24% vulnerable. | [arXiv 2403.02691](https://arxiv.org/abs/2403.02691) |
| **AgentDojo** | Dynamic environment to design/evaluate tasks, attacks, and defenses over untrusted data. | 4 suites: workspace, Slack, travel, banking. Extensible, not static. | [arXiv 2406.13352](https://arxiv.org/html/2406.13352v3) |
| **Agent Security Bench (ASB)** | Formalize & benchmark attacks AND defenses in LLM agents. | Broad attack/defense matrix. | [ASB](https://arxiv.org/abs/2410.02644) |
| **Kaggle: Multi-Step Tool Attacks** | Build attack algorithms vs tool-using agents in a deterministic offline benchmark. | OpenAI + Google + IEEE; ~$50k; deadline 2026-08-25. | [Kaggle](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks) |
| **IPI w/ externally stored personal data** | Prompt-injection on tool-integrated agents using stored personal data. | OpenReview benchmark. | [OpenReview](https://openreview.net/forum?id=APaE1JUje1) |
| **QueryIPI** | Query-agnostic indirect prompt injection on coding agents. | Coding-agent focus. | [arXiv 2510.23675](https://arxiv.org/pdf/2510.23675) |

## Evaluation principles

- **Static vs dynamic:** InjecAgent = single-turn static cases (good for regression); AgentDojo = dynamic/extensible (good for adaptive attacks & new defenses). Use both.
- **Adaptive evaluation is mandatory:** defenses that pass static tests often **fall to adaptive attacks** ([arXiv 2503.00061](https://arxiv.org/pdf/2503.00061)); always re-evaluate defenses against an adaptive attacker.
- **Multi-step / trajectory metrics:** measure failures across the *sequence* of tool calls, not just per-message — e.g., goal-drift detection, cascading-action containment.
- **Report attack success rate (ASR)** before/after each defense, and the residual ASR under adaptive attack.
- **Firewalls vs benchmarks:** some "firewall" defenses look strong until stronger benchmarks are applied ([arXiv 2510.05244](https://arxiv.org/html/2510.05244v1)).

## Public-competition evidence

**"Security Challenges in AI Agent Deployment: Insights from a Large-Scale Public Competition"** synthesizes what real attackers found at scale — useful for prioritizing defenses ([arXiv 2507.20526](https://arxiv.org/pdf/2507.20526)).

## Sources
- [InjecAgent](https://arxiv.org/abs/2403.02691) · [AgentDojo](https://arxiv.org/html/2406.13352v3) · [ASB](https://arxiv.org/abs/2410.02644)
- [Adaptive Attacks Break Defenses](https://arxiv.org/pdf/2503.00061) · [Firewalls or Stronger Benchmarks?](https://arxiv.org/html/2510.05244v1)
- [Kaggle Multi-Step Tool Attacks](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks) · [Large-Scale Public Competition](https://arxiv.org/pdf/2507.20526)
