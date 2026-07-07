# A/B Testing And Canary Releases

This directory covers safe release and experimentation patterns for ML and AI systems.

## Content Map

| Subdirectory | Scope |
|---|---|
| `A_B_Testing/` | Randomized experiments, metrics, sample size, guardrails, segmentation, and analysis. |
| `Canary_Releases/` | Progressive rollout, shadow mode, traffic splitting, rollback, and production safety gates. |

## AI-Specific Risks

- Model outputs can affect user behavior and contaminate future training data.
- LLM quality may require human review, rubric scoring, or task-specific evals.
- Canary metrics should include safety, latency, cost, refusal behavior, hallucination, and user impact.
