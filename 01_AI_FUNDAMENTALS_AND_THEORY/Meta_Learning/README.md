# Meta Learning

> Meta-learning ("learning to learn") trains models that rapidly adapt to new tasks from few examples by learning transferable inductive biases — initializations, optimizers, or embeddings — across a distribution of tasks.

## Why it matters

Standard supervised deep learning needs many labeled examples per task and re-trains from scratch for every new problem. Meta-learning instead optimizes for fast adaptation: after seeing a handful of examples (the *support set*), the model generalizes to unseen classes or tasks. It underpins few-shot classification, fast RL adaptation, hyperparameter and optimizer learning, and is a conceptual ancestor of in-context learning in modern LLMs.

## Taxonomy

| Family | Idea | Representative methods |
|---|---|---|
| **Optimization-based** | Learn an initialization / update rule so a few gradient steps adapt to a new task | MAML, Reptile, Meta-SGD, iMAML, Meta-LSTM |
| **Metric-based** | Learn an embedding space where simple distance/attention solves new tasks without inner-loop updates | Matching Networks, Prototypical Networks, Relation Networks, Siamese Nets |
| **Model-based (black-box)** | A memory-augmented / recurrent model that internalizes adaptation in its activations | MANN, SNAIL, Meta Networks |
| **Learned optimizers** | Replace hand-designed optimizers (SGD/Adam) with a learned update function | L2L by gradient descent, learned optimizers that scale |
| **Metalearning the loss/hyperparameters** | Learn loss functions, learning rates, augmentations, or architectures across tasks | Meta-SGD (per-param LR), Bilevel/AutoML hybrids |

## Key methods

| Method | Family | Year | Link |
|---|---|---|---|
| MAML — Model-Agnostic Meta-Learning (Finn et al.) | Optimization | 2017 | https://arxiv.org/abs/1703.03400 |
| Reptile — On First-Order Meta-Learning (OpenAI) | Optimization | 2018 | https://arxiv.org/abs/1803.02999 |
| Meta-SGD: Learning to Learn Quickly (Li et al.) | Optimization | 2017 | https://arxiv.org/abs/1707.09835 |
| Optimization as a Model for Few-Shot Learning (Ravi & Larochelle) | Optimization (LSTM meta-learner) | 2017 | https://openreview.net/pdf?id=rJY0-Kcll |
| Matching Networks for One Shot Learning (Vinyals et al.) | Metric | 2016 | https://arxiv.org/abs/1606.04080 |
| Prototypical Networks (Snell et al.) | Metric | 2017 | https://arxiv.org/abs/1703.05175 |
| Learning to learn by gradient descent by gradient descent (Andrychowicz et al.) | Learned optimizer | 2016 | https://arxiv.org/abs/1606.04474 |

## Key frameworks & tools

| Tool | What it provides | Link |
|---|---|---|
| learn2learn | PyTorch library: MAML, Meta-SGD, ProtoNets, task samplers, vision benchmarks | https://github.com/learnables/learn2learn |
| higher | Differentiable inner-loop optimizers (unrolled SGD) for meta-gradients in PyTorch | https://github.com/facebookresearch/higher |
| Torchmeta | Datasets + dataloaders for few-shot / meta-learning in PyTorch | https://github.com/tristandeleu/pytorch-meta |
| Meta-Dataset (code) | Reference pipeline for the multi-domain few-shot benchmark | https://github.com/google-research/meta-dataset |
| learn2learn docs | API reference for algorithms (MAML, MetaSGD, etc.) | https://learn2learn.net/docs/learn2learn.algorithms/ |

## Benchmarks & datasets

| Benchmark | Scope | Link |
|---|---|---|
| Omniglot | 1,623 handwritten characters; classic N-way K-shot one-shot benchmark | https://github.com/brendenlake/omniglot |
| miniImageNet | 100-class ImageNet subset for 5-way 1/5-shot evaluation (introduced in Matching Networks / Ravi & Larochelle) | https://arxiv.org/abs/1606.04080 |
| tieredImageNet | Larger ImageNet split with class hierarchy to limit train/test leakage | https://arxiv.org/abs/1803.00676 |
| Meta-Dataset | 10 datasets (ImageNet, Omniglot, Aircraft, CUB, DTD, QuickDraw, Fungi, Flowers, Traffic Signs, MSCOCO) for cross-domain few-shot | https://arxiv.org/abs/1903.03096 |
| Meta-World | 50 robotic manipulation tasks for meta-RL and multi-task RL | https://github.com/Farama-Foundation/Metaworld |

## Key papers

- **Model-Agnostic Meta-Learning (MAML)** — Finn, Abbeel, Levine, 2017. https://arxiv.org/abs/1703.03400
- **On First-Order Meta-Learning Algorithms (Reptile)** — Nichol, Achiam, Schulman, 2018. https://arxiv.org/abs/1803.02999
- **Meta-SGD: Learning to Learn Quickly for Few-Shot Learning** — Li, Zhou, Chen, Li, 2017. https://arxiv.org/abs/1707.09835
- **Matching Networks for One Shot Learning** — Vinyals et al., 2016. https://arxiv.org/abs/1606.04080
- **Prototypical Networks for Few-shot Learning** — Snell, Swersky, Zemel, 2017. https://arxiv.org/abs/1703.05175
- **Learning to learn by gradient descent by gradient descent** — Andrychowicz et al., 2016. https://arxiv.org/abs/1606.04474
- **Meta-Learning in Neural Networks: A Survey** — Hospedales, Antoniou, Micaelli, Storkey, 2020. https://arxiv.org/abs/2004.05439
- **Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples** — Triantafillou et al., 2020 (ICLR). https://arxiv.org/abs/1903.03096

## Cross-references in AIForge

- [Few-Shot Learning](../Few_Shot_Learning/) — metric-based meta-learning and N-way K-shot evaluation
- [Transfer Learning](../Transfer_Learning/) — related paradigm of reusing knowledge across tasks/domains
- [Optimization Algorithms](../Optimization_Algorithms/) — base optimizers and the bilevel optimization behind meta-gradients
- [Reinforcement Learning](../Reinforcement_Learning/) — meta-RL for fast policy adaptation (e.g. Meta-World)

## Sources

- https://arxiv.org/abs/1703.03400
- https://arxiv.org/abs/1803.02999
- https://arxiv.org/abs/1707.09835
- https://arxiv.org/abs/1606.04080
- https://arxiv.org/abs/1703.05175
- https://arxiv.org/abs/1606.04474
- https://arxiv.org/abs/2004.05439
- https://arxiv.org/abs/1903.03096
- https://openreview.net/pdf?id=rJY0-Kcll
- https://github.com/learnables/learn2learn
- https://github.com/facebookresearch/higher
- https://github.com/tristandeleu/pytorch-meta
- https://github.com/google-research/meta-dataset

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
