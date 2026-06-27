# Weather Climate Models

> Data-driven (deep learning) models that forecast weather and simulate climate directly from data, now matching or beating traditional physics-based numerical weather prediction (NWP) at a fraction of the compute. A flagship scientific-AI achievement alongside AlphaFold-class models.

## Why it matters

Operational NWP relies on supercomputers solving the primitive equations of the atmosphere; a single 10-day global forecast can cost hours of HPC time. Since 2022, neural models trained on ECMWF's ERA5 reanalysis (GraphCast, Pangu-Weather, FourCastNet) produce comparable or better skill in **seconds on a single GPU** — roughly three orders of magnitude cheaper. By 2025 ECMWF made its own AI model (AIFS) operational, GenCast won probabilistic skill at scale, and end-to-end systems (Aardvark) began replacing the entire pipeline including data assimilation. This family is the weather/climate analogue of foundation models in NLP.

## Taxonomy

| Approach | Idea | Representative models |
|---|---|---|
| **Graph neural network (GNN)** | Mesh-based message passing on a multi-resolution icosahedral grid | GraphCast, GenCast, AIFS |
| **Vision/3D Transformer** | Patch-based attention over lat/lon/level cubes | Pangu-Weather, Aurora, ClimaX |
| **Neural operator / Fourier** | Spherical Fourier Neural Operators (SFNO) learn solution operators | FourCastNet, FourCastNet v2/SFNO |
| **Diffusion / generative** | Probabilistic ensembles via denoising diffusion | GenCast, CorrDiff (downscaling) |
| **Hybrid ML + physics** | Differentiable dynamical core + learned subgrid physics | NeuralGCM |
| **End-to-end (no NWP)** | Ingest raw observations, skip data assimilation entirely | Aardvark Weather |
| **Foundation models** | Pretrain on heterogeneous data, fine-tune to many tasks | Aurora, ClimaX |

## Key models

| Model | Org | Type | Code / weights | Paper |
|---|---|---|---|---|
| **GraphCast** | Google DeepMind | GNN, deterministic 10-day | [github](https://github.com/google-deepmind/graphcast) | [Science 2023 / arXiv:2212.12794](https://arxiv.org/abs/2212.12794) |
| **GenCast** | Google DeepMind | Diffusion ensemble | [github](https://github.com/google-deepmind/graphcast) | [arXiv:2312.15796](https://arxiv.org/abs/2312.15796) |
| **Pangu-Weather** | Huawei Cloud | 3D Earth-Specific Transformer | [github](https://github.com/198808xc/Pangu-Weather) | [arXiv:2211.02556](https://arxiv.org/abs/2211.02556) |
| **Aurora** | Microsoft | Foundation model (3D Swin) | [github](https://github.com/microsoft/aurora) | [arXiv:2405.13063](https://arxiv.org/abs/2405.13063) |
| **FourCastNet** | NVIDIA | SFNO neural operator | [Earth2Studio](https://github.com/NVIDIA/earth2studio) | [arXiv:2202.11214](https://arxiv.org/abs/2202.11214) |
| **AIFS / AIFS-ENS** | ECMWF | GNN encoder + transformer (operational) | [HF: aifs-ens-1.0](https://huggingface.co/ecmwf/aifs-ens-1.0) | [arXiv:2406.01465](https://arxiv.org/abs/2406.01465) |
| **NeuralGCM** | Google Research | Hybrid ML + differentiable GCM | [github](https://github.com/neuralgcm/neuralgcm) | [Nature 2024](https://www.nature.com/articles/s41586-024-07744-y) |
| **ClimaX** | Microsoft | Transformer foundation model | [github](https://github.com/microsoft/ClimaX) | [arXiv:2301.10343](https://arxiv.org/abs/2301.10343) |
| **Aardvark Weather** | Cambridge / ATI | End-to-end (obs → forecast) | [github](https://github.com/annavaughan/aardvark-weather-public) | [Nature 2025](https://www.nature.com/articles/s41586-025-08897-0) |

## Tooling & frameworks

| Tool | Purpose | Link |
|---|---|---|
| **Anemoi** | ECMWF toolbox of building blocks for data-driven forecast models | [github](https://github.com/ecmwf/anemoi-core) |
| **ai-models** | ECMWF runner for GraphCast/Pangu/FourCastNet inference | [github](https://github.com/ecmwf-lab/ai-models) |
| **Earth2Studio / Earth2MIP** | NVIDIA inference, ensembling and downscaling stack | [github](https://github.com/NVIDIA/earth2studio) |
| **NeuralGCM** | JAX differentiable atmosphere solver + ML | [github](https://github.com/neuralgcm/neuralgcm) |

## Benchmarks & datasets

| Resource | Role | Link |
|---|---|---|
| **WeatherBench 2** | Standard eval framework + leaderboard for global forecasts | [github](https://github.com/google-research/weatherbench2) · [site](https://sites.research.google/weatherbench/) |
| **ERA5 reanalysis** | ECMWF 1940–present, ~0.25°; the de-facto training/ground-truth set | [docs](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) |
| **WeatherBench (v1)** | Original deep-learning weather benchmark | [github](https://github.com/pangeo-data/WeatherBench) |
| **ClimateBench** | Emulating climate-model projections from emissions | [github](https://github.com/duncanwp/ClimateBench) |
| **ARCO-ERA5** | Analysis-ready, cloud-optimized ERA5 (Google) | [github](https://github.com/google-research/arco-era5) |

## Key papers

| Paper | Venue / ID |
|---|---|
| FourCastNet: A Global Data-driven HRWF using Adaptive Fourier Neural Operators | [arXiv:2202.11214](https://arxiv.org/abs/2202.11214) |
| Pangu-Weather: Accurate medium-range global weather forecasting with 3D neural networks | [Nature 2023](https://www.nature.com/articles/s41586-023-06185-3) · [arXiv:2211.02556](https://arxiv.org/abs/2211.02556) |
| GraphCast: Learning skillful medium-range global weather forecasting | [Science 2023](https://www.science.org/doi/10.1126/science.adi2336) · [arXiv:2212.12794](https://arxiv.org/abs/2212.12794) |
| ClimaX: A foundation model for weather and climate | [arXiv:2301.10343](https://arxiv.org/abs/2301.10343) |
| Spherical Fourier Neural Operators (SFNO) | [arXiv:2306.03838](https://arxiv.org/abs/2306.03838) |
| GenCast: Diffusion-based ensemble forecasting for medium-range weather | [Nature 2024](https://www.nature.com/articles/s41586-024-08252-9) · [arXiv:2312.15796](https://arxiv.org/abs/2312.15796) |
| Neural general circulation models for weather and climate (NeuralGCM) | [Nature 2024](https://www.nature.com/articles/s41586-024-07744-y) |
| Aurora: A foundation model for the Earth system | [arXiv:2405.13063](https://arxiv.org/abs/2405.13063) |
| AIFS – ECMWF's data-driven forecasting system | [arXiv:2406.01465](https://arxiv.org/abs/2406.01465) |
| End-to-end data-driven weather prediction (Aardvark) | [Nature 2025](https://www.nature.com/articles/s41586-025-08897-0) · [arXiv:2404.00411](https://arxiv.org/abs/2404.00411) |
| WeatherBench 2: A benchmark for the next generation of data-driven models | [arXiv:2308.15560](https://arxiv.org/abs/2308.15560) |

## Cross-references in AIForge

- [`../Protein_Models/` — AlphaFold-class scientific models (sibling foundation-model family).
- [`../Quantum_ML/`](../Quantum_ML/) — physics-informed and quantum approaches to scientific ML.
- [`../Reservoir_Computing/`](../Reservoir_Computing/) — dynamical-systems / time-series forecasting methods.
- [`../../`](../../) — back to the 02_LLM_AND_AI_MODELS pillar index.

## Sources

- https://github.com/google-deepmind/graphcast
- https://www.science.org/doi/10.1126/science.adi2336
- https://www.nature.com/articles/s41586-024-08252-9 (GenCast)
- https://www.nature.com/articles/s41586-023-06185-3 (Pangu-Weather)
- https://arxiv.org/abs/2405.13063 (Aurora) · https://github.com/microsoft/aurora
- https://github.com/NVIDIA/earth2studio · https://arxiv.org/abs/2202.11214 (FourCastNet)
- https://arxiv.org/abs/2406.01465 · https://huggingface.co/ecmwf/aifs-ens-1.0 (AIFS)
- https://www.nature.com/articles/s41586-024-07744-y · https://github.com/neuralgcm/neuralgcm (NeuralGCM)
- https://github.com/microsoft/ClimaX · https://arxiv.org/abs/2301.10343
- https://www.nature.com/articles/s41586-025-08897-0 · https://arxiv.org/abs/2404.00411 (Aardvark)
- https://github.com/google-research/weatherbench2 · https://sites.research.google/weatherbench/

_Expanded from seed during a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
