# Loss Modules

This directory stores custom loss functions and objective helpers for biomass modeling.

## Scope

- Regression losses, robust losses, quantile losses, ranking losses, multi-task objectives, uncertainty losses, and metric-aligned objectives.
- Loss modules should document expected target scale, reduction mode, masking behavior, and relationship to the evaluation metric.

## Reference Links

- PyTorch loss functions: https://pytorch.org/docs/stable/nn.html#loss-functions
- scikit-learn regression metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

## Routing Rules

- Put optimizer code in `../optimizers/`.
- Put metric summaries in evaluation or project-report files.
