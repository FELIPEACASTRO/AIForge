# AutoML

This directory covers automated machine learning: search over models, features, hyperparameters, architectures, pipelines, and deployment choices.

## Content Map

| Subdirectory | Scope |
|---|---|
| `AutoML_Frameworks/` | Auto-sklearn, AutoGluon, FLAML, TPOT, H2O AutoML, Vertex AI AutoML, and related systems. |
| `Automated_Feature_Engineering/` | Featuretools, Deep Feature Synthesis, learned transformations, and feature search. |
| `Hyperparameter_Optimization/` | Bayesian optimization, random search, grid search, Hyperband, BOHB, Optuna, Ray Tune. |

## Evaluation Requirements

- Compare against strong manual baselines.
- Record search budget, hardware, data split, metric, and leakage controls.
- Separate tabular AutoML, neural architecture search, LLM prompt/model selection, and production tuning.
