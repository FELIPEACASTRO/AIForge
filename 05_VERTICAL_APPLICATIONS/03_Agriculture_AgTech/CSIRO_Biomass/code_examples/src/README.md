# Source (src)

> Core Python package for CSIRO biomass prediction, organizing the training pipeline, custom loss functions, optimizers, and inference/ensemble code for a DINOv2-based regression model.

## Contents

| Item | Description |
| --- | --- |
| [inference/](inference/) | Kaggle inference, test-time augmentation, and ensemble/stacking scripts for generating biomass predictions. |
| [losses/](losses/) | Custom robust loss functions (Huber, Quantile, R2, Multi-Task, MMD) tuned for biomass regression. |
| [optimizers/](optimizers/) | Optimizer implementations and schedules (e.g., RAdam with Lookahead) used in training. |
| [training/](training/) | Advanced DINOv2 training pipeline and domain adaptation techniques for cross-location generalization. |
| [\_\_init\_\_.py](__init__.py) | Package initializer defining version and metadata for the CSIRO biomass source code. |

## Related

- Parent: [`../`](../)

**Keywords:** biomass prediction, source package, DINOv2, PyTorch, custom losses, optimizers, domain adaptation, ensemble inference, deep learning, regression, agriculture AI, CSIRO
