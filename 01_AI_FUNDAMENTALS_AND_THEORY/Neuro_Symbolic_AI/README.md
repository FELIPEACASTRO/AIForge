# Neuro Symbolic AI

> Neuro-symbolic AI integrates neural (sub-symbolic) learning with symbolic reasoning — logic, knowledge representation, and program execution — to build systems that are simultaneously data-driven, interpretable, and capable of compositional, rule-based inference.

## Why it matters

Pure deep learning excels at perception but struggles with systematic generalization, data efficiency, verifiable reasoning, and incorporating prior knowledge or hard constraints. Pure symbolic AI reasons transparently and provably but is brittle to noisy, high-dimensional input. Neuro-symbolic methods aim to capture the strengths of both: neural networks handle perception and produce differentiable scores, while symbolic engines (logic programs, ASP, theorem provers) perform structured inference. This is widely regarded as a path toward more trustworthy, sample-efficient, and explainable AI — Garcez and Lamb call it the field's "3rd wave."

## Taxonomy (Kautz)

Henry Kautz's taxonomy (AAAI 2020 keynote, popularized in Garcez & Lamb's *3rd Wave*) classifies systems roughly by depth of integration:

| Type | Pattern | Idea | Example |
|---|---|---|---|
| 1 | `Symbolic → Neuro → Symbolic` | Symbolic in, neural processing, symbolic out (standard DL-for-NLP pipelines) | LLMs over tokenized text |
| 2 | `Symbolic[Neuro]` | A neural pattern-recognition subroutine inside a symbolic solver | AlphaGo (MCTS + nets) |
| 3 | `Neuro → Symbolic` | Neural net maps raw input to symbolic structure, then a symbolic reasoner runs | NS-CL, Neuro→Symbolic VQA |
| 4 | `Neuro : Symbolic → Neuro` | Symbolic knowledge compiled into rules/templates that supervise or regularize training | Logical constraints as loss |
| 5 | `Neuro_Symbolic` | Symbolic rules transformed into templates for neural structure; tightly fused | Logic Tensor Networks, KBANN |
| 6 | `Neuro[Symbolic]` | A symbolic reasoning engine embedded inside the neural engine for combinatorial reasoning | DeepProbLog, Logical Neural Networks |

Complementary axes: **logic semantics** (probabilistic / fuzzy / classical), **differentiability** (end-to-end trainable vs. relaxed), and **direction of correction** (perception → reasoning vs. reasoning → perception, as in abductive learning).

## Key frameworks & tools

| Framework | Symbolic substrate | Semantics | Link |
|---|---|---|---|
| DeepProbLog | ProbLog (probabilistic logic programming) | Probabilistic, differentiable | https://github.com/ML-KULeuven/deepproblog |
| Scallop | Datalog (recursion + aggregation) | Provenance / differentiable | https://github.com/scallop-lang/scallop |
| Logic Tensor Networks (LTN) | First-order fuzzy logic → tensors | Fuzzy / real-valued | https://github.com/logictensornetworks/logictensornetworks |
| NeurASP | Answer Set Programming (ASP) | Probabilistic ASP | https://github.com/azreasoners/NeurASP |
| Logical Neural Networks (LNN) | Weighted real-valued FOL | Bounded truth values, omnidirectional | https://github.com/IBM/LNN |
| Abductive Learning (ABL) | Logic programming + abduction | Symbol-correction loop | https://github.com/AbductiveLearning/ABL-Package |
| Dolphin | GPU-scalable neurosymbolic programs | Differentiable, vectorized | https://arxiv.org/abs/2410.03348 |
| pyReason | Annotated logic / graph reasoning | Temporal, open-world | https://github.com/lab-v2/pyreason |

## Benchmarks & datasets

| Benchmark | Task | Why neuro-symbolic | Link |
|---|---|---|---|
| CLEVR | Compositional visual question answering over synthetic scenes | Tests program-style reasoning over perception | https://cs.stanford.edu/people/jcjohns/clevr/ |
| CLEVRER | Causal/temporal reasoning over collision videos | Perception + dynamics + logic | http://clevrer.csail.mit.edu/ |
| MNIST-Addition | Add digits from raw image pairs given only the sum | Canonical DeepProbLog weak-supervision task | https://github.com/ML-KULeuven/deepproblog |
| bAbI | 20 synthetic reasoning/QA tasks | Symbolic inference over language | https://github.com/facebookarchive/bAbI-tasks |
| HANS | Heuristic-vs-rule NLI diagnostic | Systematic generalization | https://github.com/tommccoy1/hans |
| ProofWriter | Multi-step deductive reasoning over rules in NL | Neural theorem proving | https://allenai.org/data/proofwriter |

## Key papers

| Paper | Authors | Link |
|---|---|---|
| DeepProbLog: Neural Probabilistic Logic Programming | Manhaeve et al., 2018 | https://arxiv.org/abs/1805.10872 |
| Logic Tensor Networks | Badreddine et al., 2020 | https://arxiv.org/abs/2012.13635 |
| The Neuro-Symbolic Concept Learner (NS-CL) | Mao et al., 2019 | https://arxiv.org/abs/1904.12584 |
| Logical Neural Networks | Riegel et al., 2020 | https://arxiv.org/abs/2006.13155 |
| Bridging Machine Learning and Logical Reasoning by Abductive Learning | Dai, Xu, Yu, Zhou, NeurIPS 2019 | https://papers.nips.cc/paper/8548-bridging-machine-learning-and-logical-reasoning-by-abductive-learning |
| Neurosymbolic AI: The 3rd Wave | d'Avila Garcez & Lamb, 2020 | https://arxiv.org/abs/2012.05876 |
| Neural-Symbolic Learning and Reasoning: A Survey and Interpretation | Besold et al., 2017 | https://arxiv.org/abs/1711.03902 |
| Neuro-Symbolic AI Survey | Hamilton et al., 2021 | https://arxiv.org/abs/2105.05330 |
| Neuro-Symbolic AI in 2024: A Systematic Review | 2025 | https://arxiv.org/abs/2501.05435 |
| Dolphin: A Programmable Framework for Scalable Neurosymbolic Learning | 2024 | https://arxiv.org/abs/2410.03348 |

## Cross-references in AIForge

- [Knowledge Graphs](../Knowledge_Graphs/) — symbolic knowledge stores often paired with neural reasoners
- [Explainable AI](../Explainable_AI/) — interpretability is a primary motivation for neuro-symbolic design
- [Causal Inference](../Causal_Inference/) — structured, rule-based reasoning over interventions
- [Graph Neural Networks](../Graph_Neural_Networks/) — relational learning that bridges sub-symbolic and structured representations

## Sources

- https://arxiv.org/abs/1805.10872 — DeepProbLog
- https://arxiv.org/abs/2012.13635 — Logic Tensor Networks
- https://arxiv.org/abs/1904.12584 — Neuro-Symbolic Concept Learner
- https://arxiv.org/abs/2006.13155 — Logical Neural Networks
- https://arxiv.org/abs/2012.05876 — Neurosymbolic AI: The 3rd Wave (Kautz taxonomy)
- https://papers.nips.cc/paper/8548-bridging-machine-learning-and-logical-reasoning-by-abductive-learning — Abductive Learning
- https://arxiv.org/abs/2501.05435 — Neuro-Symbolic AI in 2024: A Systematic Review
- https://github.com/ML-KULeuven/deepproblog — DeepProbLog code
- https://github.com/scallop-lang/scallop — Scallop
- https://github.com/IBM/LNN — IBM Logical Neural Networks
- https://cs.stanford.edu/people/jcjohns/clevr/ — CLEVR benchmark

_Seed section expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
