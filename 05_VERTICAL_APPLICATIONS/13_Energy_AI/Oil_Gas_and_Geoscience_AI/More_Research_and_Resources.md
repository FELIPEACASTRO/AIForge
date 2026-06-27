# More Research & Resources — Wellbore Geology & Geosteering AI

> Additional verified, **not-previously-listed** sources for ML on wellbore geology, lithology, formation tops, and geosteering.

_37 sources, each WebFetch-verified and de-duplicated against the existing section (multi-agent double-check)._

## Papers & Surveys (21)

| Resource | What it is |
|---|---|
| [WLFM: A Well-Logs Foundation Model for Multi-Task and Cross-Well Geological Interpretation](https://arxiv.org/abs/2509.18152) | Transformer well-log foundation model pretrained on 1200 wells via masked-token modeling + stratigraphy-aware contrastive learning; multi-task fine-tuning for porosity estimation (0.0041 MSE) and lithology classification (74-78% accuracy). |
| [DISTINGUISH Workflow: A New Paradigm of Dynamic Well Placement Using Generative Machine Learning](https://arxiv.org/abs/2503.08509) | AI geosteering workflow combining GANs for geological parameterization, ensemble updating, and dynamic programming optimization, ingesting LWD data to adjust well trajectories in real time. |
| [High-Precision Geosteering via Reinforcement Learning and Particle Filters](https://arxiv.org/abs/2402.06377) | Integrates RL-based geosteering decision-making with particle filters that process real-time well-log data to localize the bit relative to stratigraphic layers. |
| [Optimal Sequential Decision-Making in Geosteering: A Reinforcement Learning Approach](https://arxiv.org/abs/2310.04772) | Applies Deep Q-Network (model-free RL) to optimize sequential drilling-trajectory decisions in geosteering, matching quasi-optimal ADP with greater flexibility; v2 updated Jan 2025. |
| [GIAT: A Geologically-Informed Attention Transformer for Lithology Identification](https://arxiv.org/abs/2603.09165) | Transformer that fuses data-driven geological priors with self-attention for well-log lithology identification, reporting up to 95.4% accuracy. |
| [GeoMind: An Agentic Workflow for Lithology Classification with Reasoned Tool Invocation](https://arxiv.org/abs/2604.21501) | Agentic LLM-driven workflow that models lithology classification as sequential reasoning over perception/reasoning/analysis tool modules for evidence-grounded decisions. |
| [The NCS-Model: A Seismic Foundation Model Trained on the Norwegian Repository of Public Data](https://arxiv.org/abs/2603.23211) | Open-sourced seismic interpretation foundation model trained on public Norwegian Continental Shelf (DISKOS) data; relevant to subsurface/wellbore geoscience foundation modeling. |
| [STNet: Advancing Lithology Identification with a Spatiotemporal Deep Learning Framework for Well Logging Data](https://link.springer.com/article/10.1007/s11053-024-10413-6) | Spatiotemporal deep-learning framework (Natural Resources Research) for lithology identification from well-logging curves. |
| [Enhanced Lithology Classification Using an Interpretable SHAP Model Integrating Semi-Supervised Contrastive Learning and Transformer with Well Logging Data](https://link.springer.com/article/10.1007/s11053-024-10452-z) | Interpretable (SHAP) lithology classifier combining semi-supervised contrastive learning with a transformer backbone on well-log data. |
| [Stratigraphic Correlation of Well Logs Using Geology-Informed Deep Learning Networks](https://www.mdpi.com/2227-9717/13/5/1288) | Geology-informed deep-learning approach (MDPI Processes) for automatic stratigraphic correlation / formation-top alignment across well logs. |
| [Well Logging Stratigraphic Correlation Algorithm Based on Semantic Segmentation](https://link.springer.com/article/10.1007/s11770-024-1085-8) | Treats well-log stratigraphic correlation as a semantic-segmentation task (Applied Geophysics) for automated formation-boundary picking. |
| [Leveraging Time-Series Foundation Model (TimeGPT) for Subsurface Well Logs Prediction and Anomaly Detection](https://arxiv.org/abs/2412.05681) | Fine-tunes the TimeGPT transformer foundation model on borehole well logs for curve prediction (R2 up to 87%, MAPE 1.95%) and drilling-hazard/geology anomaly detection (93% accuracy). |
| [Geological Everything Model 3D (GEM): A Promptable Foundation Model for Unified, Zero-Shot Subsurface Understanding](https://arxiv.org/abs/2507.00419) | Promptable generative foundation model that reformulates structural interpretation, stratigraphy, and property modeling as prompt-conditioned inference over latent structural frameworks from subsurface imaging; self-supervised + adversarial fine-tuning. |
| [Direct multi-modal inversion of geophysical logs using deep learning](https://arxiv.org/abs/2201.01871) | Mixture density network with multiple-trajectory-prediction loss giving fast probabilistic, multi-modal inversion of logging-while-drilling logs for geosteering decisions under uncertainty. |
| [Research on Wellbore Trajectory Optimization and Drilling Control Based on the TD3 Algorithm](https://www.mdpi.com/2076-3417/15/13/7258) | Applies the TD3 deep reinforcement learning algorithm to wellbore trajectory optimization and drilling control, with target-line alignment and dog-leg-severity constraints. |
| [Latent diffusion models for parameterization and data assimilation of facies-based geomodels](https://arxiv.org/abs/2406.14815) | Latent diffusion model (VAE + U-Net denoiser) to parameterize channel-levee-mud facies geomodels and perform ensemble-based history matching with uncertainty reduction. |
| [3D latent diffusion models for parameterizing and history matching multiscenario facies systems](https://arxiv.org/abs/2508.16621) | Extends latent diffusion to 3D facies systems for low-dimensional parameterization and multiscenario ensemble history matching with preserved geological realism. |
| [Diffusion models for multivariate subsurface generation and efficient probabilistic inversion](https://arxiv.org/abs/2507.15809) | Diffusion models generating jointly facies and correlated acoustic impedance, conditioned on well logs (hard data) and fullstack seismic, with diffusion-posterior-sampling inversion. |
| [Probabilistic forecasting for geosteering in fluvial successions using a generative adversarial network](https://arxiv.org/abs/2207.01374) | GAN trained to generate geologically realistic 2D fluvial sections, used with ensemble updating to forecast geology and reduce uncertainty up to 500 m ahead of the drill bit. |
| [Robust representations of oil wells' intervals via sparse attention mechanism (Reguformers)](https://arxiv.org/abs/2212.14246) | Regularized sparse-attention Transformers for representation/similarity learning on well-log intervals, robust to missing/noisy curves and cheaper than full Transformers. |
| [Machine learning for sustainable geoenergy: uncertainty, physics and decision-ready inference](https://arxiv.org/abs/2603.14907) | Survey/position paper on physics-ML and probabilistic methods for subsurface geoenergy (CO2 storage, geothermal, H2), covering scarce data, UQ, multi-scale modeling, and decision-ready inference. |

## Survey (1)

| Resource | What it is |
|---|---|
| [On the workflow, opportunities and challenges of developing foundation models in geophysics](https://arxiv.org/abs/2504.17384) | Systematic framework/survey for building geophysical foundation models across data collection, preprocessing, architecture, pretraining strategy and deployment, addressing physical-consistency constraints. |

## Benchmarks & Datasets (1)

| Resource | What it is |
|---|---|
| [FORCE-2020 Machine Learning Competition (official results/code/data repo)](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition) | Official post-competition repository for FORCE 2020 lithology prediction: training/test/blind LAS data, the custom penalty matrix, and winning code — the canonical code companion to the Zenodo dataset already cited. |

## Datasets (1)

| Resource | What it is |
|---|---|
| [OpenSeisML: Open Large-Scale Real Seismic and well-log Dataset for Generative AI](https://arxiv.org/abs/2605.20539) | Open large-scale curated dataset of real imaged seismic volumes plus well-log data (UK NDR) with an automated pipeline, built for training generative AI / seismic inversion priors. |

## Tools & Libraries (7)

| Resource | What it is |
|---|---|
| [lumisong/Awesome-Well-Log-ML-DL (curated resource list)](https://github.com/lumisong/Awesome-Well-Log-ML-DL) | Curated GitHub hub of well-log ML/DL resources: open-source repos, competition datasets, and papers organized by year (2018-2025) for petrophysics/formation evaluation. |
| [equinor/force-ml-2020-wells (FORCE 2020 well-log lithology competition code)](https://github.com/equinor/force-ml-2020-wells) | Equinor repository for the FORCE 2020 Well Log ML lithology-prediction contest (98 train / 10 test wells, XGB/CatBoost baselines); archived read-only Mar 2025. |
| [geosteering-no — NORCE open-source geosteering org (DISTINGUISH, GAN-geosteering, interactive benchmark, World Cup dataset)](https://github.com/geosteering-no) | GitHub org hosting open geosteering code & benchmarks: open DISTINGUISH workflow, GAN-geosteering, interactive-geosteering-benchmark (sequential-decision), and the 10,000-interpretations Geosteering World Cup 2021 dataset. |
| [quick_pp — Python toolkit for quick-look petrophysics with ML training pipelines](https://github.com/imranfadhil/quick_pp) | MIT-licensed petrophysics package (porosity, permeability, saturation, lithology, rock typing) with FastAPI backend, SvelteKit UI, CLI, and MLflow-tracked ML training/prediction pipelines; applied to open carbonate data in 2025 study. |
| [petrolib — Python Package for Petrophysical Evaluation](https://github.com/joshua-atolagbe/petrolib) | Open-source library for Vshale (Clavier/Stieber/Larionov), porosity (density/sonic), water saturation (Archie/Simandoux), permeability, neutron-density and Pickett plots, and automated pay-summary reports from LAS logs. |
| [PetroPy — petrophysics Python package for conventional/unconventional formation evaluation](https://github.com/toddheitmann/PetroPy) | Reads LAS into pandas, provides a basic petrophysical workflow and XML-template log viewer; widely used building block for log-ML preprocessing not currently listed alongside lasio/welly. |
| [OSDU Forum Common Python SDK — official subsurface platform client](https://community.opengroup.org/osdu/platform/system/sdks/common-python-sdk) | Official OSDU Forum Python SDK encapsulating calls to OSDU Core Storage/Search services; backend for the OSDU CLI — the maintained way to programmatically load/query well & log records on OSDU. |

## Standards & Frameworks (1)

| Resource | What it is |
|---|---|
| [Accenture OSDU-Ontology — open ontology for oil & gas / subsurface energy data](https://github.com/Accenture/OSDU-Ontology) | Open ontology for oil-and-gas and subsurface energy data built on OSDU schema/standards; useful for semantic mapping and knowledge-graph work over well/subsurface data. |

## Competitions (5)

| Resource | What it is |
|---|---|
| [SPWLA PDDA 2024 Machine Learning Competition — Fracture Identification from Image Logs](https://github.com/pddasig/Machine-Learning-Competition-2024) | 4th SPWLA Petrophysical Data-Driven Analytics ML contest: detect/localize fractures in resistivity borehole image logs; provides image-log data + fracture labels and baseline notebooks. |
| [SPWLA PDDA 2023 Machine Learning Competition — Automatic Well-Log Depth Shifting](https://github.com/pddasig/Machine-Learning-Competition-2023) | 3rd PDDA ML contest: data-driven automatic alignment of well logs to a reference log and correction of depth misalignments; open well-log data + scoring code. |
| [SPWLA PDDA 2021 Machine Learning Competition — Well-Log-Based Reservoir Property Estimation](https://github.com/pddasig/Machine-Learning-Competition-2021) | PDDA contest using logs from 9 training wells + 5 blind-test wells to predict reservoir properties; open data and evaluation harness. |
| [SPWLA PDDA 2020 Machine Learning Competition — Sonic Log Synthesis (DTC/DTS)](https://github.com/pddasig/Machine-Learning-Competition-2020) | 1st PDDA contest: generate synthetic compressional/shear sonic travel-time logs (DTC, DTS) from other logs; foundational open well-log ML benchmark. |
| [ThinkOnward — FORCE: Machine Predicted Lithology challenge](https://thinkonward.com/app/c/challenges/force-well-logs/overview) | Hosted FORCE lithology challenge: 98 training wells (18 log types), 10 open-test + 10-20 hidden wells, geoscientist-derived penalty matrix; data/predictions released under CC-BY-4.0 / Apache-2.0. |

