# Deep Learning

This directory covers neural-network theory, architectures, optimization, training strategies, and reusable deep-learning methods that support model families across AIForge.

## Content Map

| Subarea | What belongs here |
|---|---|
| Architectures | MLPs, CNNs, RNNs, LSTMs, GRUs, transformers, state-space models, graph neural networks, and hybrid models. |
| Training methods | Backpropagation, optimizers, schedulers, normalization, initialization, regularization, curriculum learning, and mixed precision. |
| Representation learning | Embeddings, contrastive learning, self-supervised learning, masked modeling, metric learning, and transfer learning. |
| Scaling and efficiency | Distillation, pruning, quantization, sparsity, MoE, activation checkpointing, and efficient attention. |
| Robustness and generalization | Distribution shift, adversarial robustness, calibration, uncertainty, and out-of-distribution detection. |

## Source Priorities

- Primary papers and benchmark papers from NeurIPS, ICML, ICLR, PMLR, JMLR, ACL, CVPR, ICCV, ECCV, and arXiv.
- Official framework docs from PyTorch, TensorFlow, JAX, Keras, Hugging Face, NVIDIA, and MLCommons.
- Reproducible code and benchmark implementations before blog summaries.

## Routing Rules

- Put individual model families in `../../02_LLM_AND_AI_MODELS/` when the page is mainly about a named model.
- Put deployment and inference optimization in `../../04_MLOPS_AND_PRODUCTION_AI/`.
- Put applied domain examples in `../../05_VERTICAL_APPLICATIONS/`.

## Next Enrichment

Create subtopic indexes for architectures, optimization, self-supervised learning, robustness, and efficient training.
