# Generative Models

This directory covers model families that learn to generate data: text, images, audio, video, molecules, simulations, tabular data, and latent representations.

## Content Map

| Subdirectory | Scope |
|---|---|
| `GANs/` | Adversarial training, image synthesis, stabilization, conditional GANs, and StyleGAN-style models. |
| `Variational_Autoencoders/` | Latent-variable models, ELBO, VQ-VAE, beta-VAE, and representation learning. |
| `diffusion/` | DDPMs, score-based models, latent diffusion, rectified flow, video diffusion, and controlled generation. |
| `flow_based/` | Normalizing flows, invertible networks, density estimation, and exact likelihood models. |
| `ebm_training_methods/` | Energy-based models, contrastive divergence, score matching, and sampling. |

## Routing Rules

- Put named foundation models in `../../02_LLM_AND_AI_MODELS/`.
- Put prompt templates for generation in `../Prompt_Engineering/`.
- Put production serving and optimization in `../../04_MLOPS_AND_PRODUCTION_AI/`.

## Evaluation Notes

Record task, dataset, likelihood or perceptual metric, safety constraints, memorization risk, and whether the model is conditional, controllable, or open-ended.
