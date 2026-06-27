# Causal Inference for ML

Resources for **causal inference**, **causal discovery**, **counterfactual reasoning**, and applying causality to ML for robustness, fairness, and decision-making.

## Foundations

- **Causality** (Pearl, 2009) — canonical textbook
- **Elements of Causal Inference** (Peters, Janzing, Schölkopf — MIT Press, free PDF) — https://mitpress.mit.edu/9780262037310/
- **The Book of Why** (Pearl & Mackenzie, 2018) — popular intro
- **Causal Inference: The Mixtape** (Cunningham) — https://mixtape.scunning.com/
- **Causal Inference for The Brave and True** — https://matheusfacure.github.io/python-causality-handbook/

## Frameworks

| Library | Focus | Link |
|---|---|---|
| **DoWhy (PyWhy)** | End-to-end causal inference | https://github.com/py-why/dowhy |
| **EconML (Microsoft)** | ML-based heterogeneous treatment effects | https://github.com/py-why/EconML |
| **CausalML (Uber)** | Uplift modeling, ITE | https://github.com/uber/causalml |
| **Pyro / NumPyro** | Probabilistic programming | https://pyro.ai/ |
| **CausalNex** | Bayesian network causal reasoning | https://github.com/mckinsey/causalnex |
| **gCastle (Huawei)** | Causal discovery toolbox | https://github.com/huawei-noah/trustworthyAI |
| **CDT (Causal Discovery Toolbox)** | Discovery algorithms | https://github.com/FenTechSolutions/CausalDiscoveryToolbox |
| **causal-learn** | CMU causal discovery suite | https://github.com/py-why/causal-learn |

## Topics

- **Potential Outcomes Framework** (Rubin)
- **Structural Causal Models / DAGs** (Pearl)
- **Causal Discovery** — PC, GES, NOTEARS, LiNGAM
- **Instrumental Variables** — IV estimation
- **Difference-in-Differences** — DiD with neural nets
- **Synthetic Control Methods**
- **Heterogeneous Treatment Effects** — T-Learner, S-Learner, X-Learner, DR-Learner
- **Causal Representation Learning** — Schölkopf et al. https://arxiv.org/abs/2102.11107
- **Causal LLMs** — counterfactual prompting and reasoning evals

## Key Papers

- **Causality for Machine Learning** — Schölkopf 2019 — https://arxiv.org/abs/1911.10500
- **NOTEARS (continuous DAG learning)** — Zheng 2018 — https://arxiv.org/abs/1803.01422
- **Causal Effect Inference with Deep Latent-Variable Models** — Louizos 2017 — https://arxiv.org/abs/1705.08821
- **Towards Causal Foundation Model** — https://arxiv.org/abs/2402.02858
