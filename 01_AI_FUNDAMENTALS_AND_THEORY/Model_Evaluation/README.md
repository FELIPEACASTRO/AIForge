# Model Evaluation

This directory covers evaluation principles for ML models, LLM systems, agents, retrieval systems, and domain-specific AI.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Evaluation_Metrics/` | Accuracy, F1, AUROC, calibration, ranking, regression, generative, RAG, and domain metrics. |
| `Model_Selection_and_Validation/` | Cross-validation, holdout design, leakage control, hyperparameter selection, and model comparison. |

## Evaluation Layers

| Layer | Examples |
|---|---|
| Dataset split quality | Train/test leakage, time split, stratification, group split, benchmark contamination. |
| Metric fit | Whether the metric matches the actual decision, cost, risk, or user outcome. |
| Robustness | Domain shift, adversarial examples, missing data, rare classes, long-tail behavior. |
| Human and system evaluation | LLM judges, human preference, task completion, latency, cost, safety, and reliability. |
| Production monitoring | Drift, calibration decay, feedback loops, canaries, and rollback criteria. |

## Source Families

Use MLCommons, HELM, SWE-bench, OpenML benchmark suites, PromptBench, Ragas, Promptfoo, OpenAI Evals, and EleutherAI LM Evaluation Harness as source families for deeper pages.
