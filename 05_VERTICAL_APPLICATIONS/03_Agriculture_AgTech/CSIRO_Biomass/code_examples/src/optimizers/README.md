# Optimizers

> PyTorch implementations of advanced deep learning optimizers used to train the CSIRO Biomass regression models, including Lookahead wrapping and rectified adaptive learning rates for stable convergence on remote-sensing imagery.

## Contents

| Item | Description |
| --- | --- |
| [\_\_init\_\_.py](__init__.py) | Package initializer exporting `Lookahead`, `RAdam`, `get_optimizer`, and `get_scheduler` for easy import. |
| [advanced_optimizers.py](advanced_optimizers.py) | Custom optimizer implementations: `Lookahead` (a wrapper improving convergence of any base optimizer), `RAdam` (Rectified Adam), plus `get_optimizer` and `get_scheduler` factory helpers for learning-rate scheduling. |

## Related

- Parent: [`../`](../)

**Keywords:** PyTorch optimizers, Lookahead, RAdam, Rectified Adam, Ranger21, learning rate scheduler, deep learning training, gradient descent, cosine annealing, CSIRO Biomass, model optimization, convergence
