# 21 Supply Chain and Logistics AI

> Applied AI/ML and operations research for forecasting demand, optimizing inventory and routing, coordinating warehouse robotics, and giving end-to-end supply-chain visibility and resilience.

## Why it matters

Supply chain and logistics is one of the largest enterprise AI markets: demand forecasting, inventory optimization, vehicle/fleet routing, warehouse automation, and last-mile delivery touch trillions of dollars in goods flow. It is a flagship vertical for real-world ML deployment at hyperscale — Amazon, Walmart, Maersk, DHL, Flexport, and project44 run forecasting and optimization models in production across millions of SKUs and shipments. The field uniquely blends time-series deep learning, neural combinatorial optimization, multi-agent reinforcement learning, and classical operations research, making it a rich integration target for modern foundation models.

## Taxonomy

| Sub-area | What it does | Representative approaches |
|---|---|---|
| Demand forecasting | Predict SKU/location demand across horizons | Gradient boosting, LSTM/TFT, time-series foundation models |
| Inventory & replenishment | Set safety stock, reorder points, allocation | Newsvendor/RL policies, forecast-then-optimize |
| Vehicle routing (VRP) | Plan cost-minimal delivery routes | OR solvers (CP/MIP), metaheuristics, neural CO |
| Network & facility design | Locate warehouses, flow allocation | MIP, simulation, optimization |
| Warehouse robotics | Coordinate AMRs, task allocation, MAPF | Multi-agent RL, multi-agent path finding |
| Last-mile & ETA | Delivery scheduling, arrival prediction | GNNs, spatio-temporal models |
| Visibility & risk | Track-and-trace, disruption prediction | Anomaly detection, event extraction |

## Key models & tools

| Tool / Model | Area | Link |
|---|---|---|
| Google OR-Tools | Routing (VRP/TSP), scheduling, CP/MIP | https://github.com/google/or-tools |
| PyVRP | State-of-the-art hybrid genetic VRP solver | https://github.com/PyVRP/PyVRP |
| VROOM | Open-source vehicle routing engine | https://github.com/VROOM-Project/vroom |
| RL4CO | RL benchmark/library for combinatorial optimization | https://github.com/ai4co/rl4co |
| Nixtla NeuralForecast | Scalable neural time-series forecasting | https://github.com/Nixtla/neuralforecast |
| Nixtla StatsForecast | Fast statistical/baseline forecasting at scale | https://github.com/Nixtla/statsforecast |
| Amazon Chronos | Pretrained time-series foundation models | https://github.com/amazon-science/chronos-forecasting |
| Google TimesFM | Decoder-only time-series foundation model | https://github.com/google-research/timesfm |
| PyTorch Forecasting (TFT) | Temporal Fusion Transformer & DeepAR impls | https://github.com/sktime/pytorch-forecasting |
| Amazon SCOT (research area) | Operations research & optimization at Amazon | https://www.amazon.science/research-areas/operations-research-and-optimization |

## Foundation & forecasting models

| Model | Type | Link |
|---|---|---|
| Chronos-2 | Universal (uni/multivariate + covariates) zero-shot TSFM | https://huggingface.co/amazon/chronos-2 |
| TimesFM 1.0 (200M) | Decoder-only pretrained forecaster | https://huggingface.co/google/timesfm-1.0-200m |
| Temporal Fusion Transformer | Interpretable multi-horizon attention model | https://arxiv.org/abs/1912.09363 |

## Benchmarks & datasets

| Benchmark / Dataset | Focus | Link |
|---|---|---|
| M5 Forecasting (Walmart) | Hierarchical retail demand forecasting | https://www.kaggle.com/c/m5-forecasting-accuracy |
| GIFT-Eval | General time-series forecasting evaluation | https://huggingface.co/spaces/Salesforce/GIFT-Eval |
| RWARE (Multi-Robot Warehouse) | Cooperative MARL warehouse env | https://github.com/semitable/robotic-warehouse |
| CVRPLIB | Capacitated VRP benchmark instances | http://vrp.galgos.inf.puc-rio.br/ |
| Monash TS Forecasting Archive | Forecasting benchmark repository | https://forecastingdata.org/ |

## Key papers

| Paper | Year | Link |
|---|---|---|
| Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting | 2019 | https://arxiv.org/abs/1912.09363 |
| Chronos: Learning the Language of Time Series | 2024 | https://arxiv.org/abs/2403.07815 |
| A decoder-only foundation model for time-series forecasting (TimesFM) | 2023 | https://arxiv.org/abs/2310.10688 |
| Attention, Learn to Solve Routing Problems! | 2018 | https://arxiv.org/abs/1803.08475 |
| POMO: Policy Optimization with Multiple Optima for RL | 2020 | https://arxiv.org/abs/2010.16011 |
| Learning to Delegate for Large-scale Vehicle Routing | 2021 | https://arxiv.org/abs/2107.04139 |
| RL4CO: An Extensive Reinforcement Learning for Combinatorial Optimization Benchmark | 2023 | https://arxiv.org/abs/2306.17100 |
| Learning to Simulate Complex Physics with Graph Networks | 2020 | https://arxiv.org/abs/2002.09405 |
| DeepFleet: Multi-Agent Foundation Models for Mobile Robots | 2025 | https://arxiv.org/abs/2508.08574 |

## Cross-references in AIForge

- [Robotics and Embodied AI](../10_Robotics_and_Embodied_AI/) — AMR/warehouse robot control and MARL coordination
- [Autonomous Vehicles AI](../11_Autonomous_Vehicles_AI/) — routing, fleet, and last-mile delivery autonomy
- [Retail and Ecommerce](../07_Retail_and_Ecommerce/) — downstream demand signals and inventory/SKU forecasting
- [Manufacturing and Industry AI](../08_Manufacturing_and_Industry_AI/) — upstream production planning and scheduling

## Sources

- https://github.com/google/or-tools
- https://github.com/PyVRP/PyVRP
- https://github.com/ai4co/rl4co
- https://github.com/Nixtla/neuralforecast
- https://github.com/amazon-science/chronos-forecasting
- https://github.com/google-research/timesfm
- https://arxiv.org/abs/1912.09363
- https://arxiv.org/abs/2310.10688
- https://arxiv.org/abs/1803.08475
- https://arxiv.org/abs/2306.17100
- https://www.amazon.science/research-areas/operations-research-and-optimization
- https://www.kaggle.com/c/m5-forecasting-accuracy

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
