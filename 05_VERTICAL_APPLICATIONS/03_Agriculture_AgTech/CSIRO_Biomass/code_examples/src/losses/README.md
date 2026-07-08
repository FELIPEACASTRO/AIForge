# Losses

> Custom loss functions for CSIRO biomass prediction, providing robust regression objectives tolerant to outliers and support for multi-task and domain-adaptation training.

## Contents

| Item | Description |
| --- | --- |
| [Custom Losses](custom_losses.py) | Implements Huber, Quantile, R2, Multi-Task (uncertainty-weighted), and MMD losses plus a `get_loss_function` factory for robust biomass regression. |
| [\_\_init\_\_.py](__init__.py) | Package exports for the custom loss classes and the loss-selection helper. |

## Related

- Parent: [`../`](../)

**Keywords:** loss functions, Huber loss, quantile loss, R2 loss, multi-task loss, MMD loss, robust regression, outlier handling, uncertainty weighting, PyTorch, biomass prediction, deep learning
