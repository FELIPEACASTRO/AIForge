# 23 Insurance AI

> AI/ML applied across the insurance value chain: automated underwriting, claims triage and document AI, actuarial pricing and reserving, fraud detection, catastrophe and parametric modeling, and telematics-based usage pricing.

## Why it matters

Insurance is a trillion-dollar vertical whose core operations — pricing risk, settling claims, reserving capital — are fundamentally statistical, making it one of the most natural and highest-value targets for machine learning. The stack spans interpretable actuarial GLMs (mandated by regulators), gradient-boosted and neural pricing models, computer-vision claims assessment, NLP/LLM document extraction for underwriting submissions, graph-based fraud detection, and physical catastrophe simulation. Stringent regulatory, explainability, and fairness requirements make this a domain where "black-box vs. interpretable" tradeoffs are first-order, not academic. This page consolidates the open-source tooling, public datasets, and foundational research for the area.

## Taxonomy

| Sub-area | What it does | Typical methods |
|---|---|---|
| Actuarial pricing | Frequency × severity → pure premium / technical tariff | GLM (Poisson/Gamma/Tweedie), GBM, CANN, LocalGLMnet |
| Reserving | Estimate outstanding/IBNR claim liabilities | Chain-ladder, Mack, bootstrap, ML reserving |
| Underwriting | Risk selection, submission triage, COPE/data extraction | LLMs, document AI, agentic pipelines |
| Claims triage & assessment | Routing, severity prediction, damage estimation | NLP, computer vision, GBM |
| Fraud detection | Flag anomalous / colluding claims | Graph/social-network analytics, anomaly detection, GBM |
| Catastrophe (cat) modeling | Simulate hazard → exposure → loss | Stochastic simulation, physical models, Oasis kernel |
| Parametric insurance | Index-triggered automatic payouts | Trigger functions over hazard indices |
| Telematics / UBI | Usage-based / behavior pricing | Driving-behavior features, interpretable ML |

## Key tools & frameworks

| Tool | Area | Link |
|---|---|---|
| chainladder-python (CAS) | Reserving & claims development | https://github.com/casact/chainladder-python |
| glum | High-performance GLMs (Poisson/Gamma/Tweedie) | https://github.com/Quantco/glum |
| scikit-learn Tweedie pricing example | Frequency/severity/pure-premium GLM | https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html |
| Oasis LMF | Open-source catastrophe modeling platform | https://github.com/OasisLMF |
| CASdatasets (R) | Canonical actuarial datasets (freMTPL etc.) | https://dutangc.github.io/CASdatasets/ |
| InterpretML (EBM) | Explainable Boosting Machine for claims | https://github.com/interpretml/interpret |
| XGBoost / LightGBM / CatBoost | GBM pricing & severity models | https://github.com/dmlc/xgboost |
| Tractable | CV-based accident/disaster claims (commercial) | https://tractable.ai/ |
| Lemonade engineering blog | AI-first claims automation (case material) | https://www.lemonade.com/blog/lemonade-metromile-acquisition/ |

## Datasets & benchmarks

| Dataset | Use | Link |
|---|---|---|
| freMTPL / freMTPL2 (French MTPL) | Claim frequency & severity, pricing benchmark | https://dutangc.github.io/CASdatasets/reference/freMTPL.html |
| Allstate Claims Severity (Kaggle) | Claim severity regression benchmark | https://www.kaggle.com/c/allstate-claims-severity |
| Porto Seguro Safe Driver (Kaggle) | Auto claim probability prediction | https://www.kaggle.com/c/porto-seguro-safe-driver-prediction |
| Vehicle Insurance Fraud (Kaggle) | Fraud classification | https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection |
| Healthcare Provider Fraud (Kaggle) | Provider-level fraud detection | https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis |
| glum French motor GLM tutorial | Reproducible pricing walkthrough | https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html |

## Key papers

| Paper | Topic | Link |
|---|---|---|
| Richman & Wüthrich (2021), *LocalGLMnet: interpretable deep learning for tabular data* | Interpretable NN matching GLM structure | https://arxiv.org/abs/2107.11059 |
| Henckaerts et al. (2023), *Neural networks for insurance pricing with frequency and severity data* | GLM vs GBM vs NN vs CANN benchmark | https://arxiv.org/abs/2310.12671 |
| Óskarsdóttir et al. (2021), *Social network analytics for supervised fraud detection in insurance* | Network/BiRank fraud detection | https://arxiv.org/abs/2009.08313 |
| *Enhanced Gradient Boosting for Zero-Inflated Insurance Claims* (CatBoost/XGBoost/LightGBM) | Zero-inflated claim modeling | https://arxiv.org/abs/2307.07771 |
| *Explainable Boosting Machine for Predicting Claim Severity and Frequency in Car Insurance* | EBM interpretable pricing | https://arxiv.org/abs/2503.21321 |
| *An Interpretable Deep Learning Model for General Insurance Pricing* | GLM/GAM vs LocalGLMnet/EBM vs NN/GBM | https://arxiv.org/abs/2509.08467 |
| *Automated Machine Learning in Insurance* | AutoML adapted to insurance data pathologies | https://arxiv.org/abs/2408.14331 |
| *Enhancing actuarial non-life pricing models via transformers* | Transformer architectures for pricing | https://arxiv.org/abs/2311.07597 |

## Cross-references in AIForge

- [02 Finance and Fintech AI](../02_Finance_and_Fintech_AI/README.md) — fraud detection, risk scoring, and time-series methods that overlap with insurance.
- [01 Healthcare and Medical AI — health-insurance fraud, claims coding, and medical document AI.
- [04 Climate and Sustainability — hazard models feeding catastrophe and parametric pricing.

## Sources

- https://github.com/casact/chainladder-python
- https://github.com/Quantco/glum
- https://github.com/OasisLMF
- https://dutangc.github.io/CASdatasets/reference/freMTPL.html
- https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
- https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html
- https://arxiv.org/abs/2107.11059
- https://arxiv.org/abs/2310.12671
- https://arxiv.org/abs/2009.08313
- https://arxiv.org/abs/2509.08467
- https://www.mdpi.com/2227-9091/9/1/4

_Expanded from a seed gap-fill sweep; all links verified via web research. Contributions welcome (see CONTRIBUTING.md)._
