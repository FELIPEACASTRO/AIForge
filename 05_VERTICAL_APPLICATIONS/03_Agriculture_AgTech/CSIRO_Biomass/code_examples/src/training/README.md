# Training Modules

This directory stores training code for biomass models.

## Scope

- Epoch loops, folds, validation, checkpointing, callbacks, logging, reproducibility, and training configuration.
- Training code should record random seeds, split strategy, metric, hardware, input shape, output target, and artifact path.

## Reference Links

- PyTorch training recipes: https://pytorch.org/tutorials/recipes/recipes_index.html
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- scikit-learn model selection: https://scikit-learn.org/stable/model_selection.html

## Routing Rules

- Put inference-only code in `../inference/`.
- Put losses in `../losses/`.
- Put optimizer utilities in `../optimizers/`.
