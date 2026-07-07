# Biomass Source Modules

This directory stores source modules used by the CSIRO biomass code examples.

## Scope

- Training loops, inference helpers, model utilities, losses, optimizers, data transforms, and evaluation code.
- Every module should be importable, testable, and clear about expected tensor/dataframe shapes.
- Keep notebooks and narrative reports outside this source tree.

## Content Map

| Subdirectory | Scope |
|---|---|
| `training/` | Training loops, schedulers, validation hooks, and checkpoint logic. |
| `inference/` | Prediction, batching, tiling, export, and post-processing. |
| `losses/` | Custom biomass losses, robust objectives, and metric-aligned losses. |
| `optimizers/` | Optimizer choices, learning-rate policies, and optimizer wrappers. |

## Routing Rules

- Put dataset documentation in the dataset folders.
- Put model-selection notes in the AgTech biomass or project-showcase folders.
