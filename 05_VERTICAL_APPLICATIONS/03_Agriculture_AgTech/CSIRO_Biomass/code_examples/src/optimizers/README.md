# Optimizer Modules

This directory stores optimizer and schedule utilities for biomass model training.

## Scope

- Optimizers, parameter groups, weight decay rules, learning-rate schedules, warmup, clipping, mixed precision, and restart policies.
- Optimizer notes should record model family, batch size, hardware, metric impact, and failure modes.

## Reference Links

- PyTorch optimizers: https://pytorch.org/docs/stable/optim.html
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- timm optimizers and schedulers: https://huggingface.co/docs/timm/index

## Routing Rules

- Put training-loop integration in `../training/`.
- Put loss functions in `../losses/`.
