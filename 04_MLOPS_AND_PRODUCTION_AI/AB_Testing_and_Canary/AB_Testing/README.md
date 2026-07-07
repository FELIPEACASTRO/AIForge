# A/B Testing for AI Systems

This directory covers A/B tests, online experiments, and controlled rollouts for AI products.

## Scope

- Experiment design, assignment, guardrails, metrics, sequential testing, interference, safety monitoring, and business/clinical/regulatory constraints.
- Track hypothesis, population, randomization unit, sample size, primary metric, guardrails, duration, and decision rule.

## Reference Links

- Microsoft Experimentation Platform: https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/
- DoWhy causal inference: https://www.pywhy.org/dowhy/
- CausalML: https://causalml.readthedocs.io/
- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

## Routing Rules

- Put canary deployment in sibling `Canary_Deployment/`.
- Put offline model evaluation in model evaluation directories.
