# Wellbore Geology Prediction & Geosteering AI

> Applying **machine learning and deep learning** to subsurface geoscience: predicting the **geology encountered along a wellbore** from well logs and drilling data, classifying **lithology/lithofacies**, picking **formation tops**, and optimizing **geosteering** (real-time well-trajectory steering) for oil & gas and geothermal drilling.

## Why it matters

~10,000 horizontal wells are drilled worldwide every year, and accurate placement within a thin target reservoir ("staying in zone") is worth enormous amounts of recovered resource. Traditionally this depends on **manual interpretation by expert geologists in real time**. ML automates and accelerates lithology prediction, formation-top picking, and trajectory optimization — often **without the time lag** of manual workflows, enabling real-time **geosteering**.

## Core problems (the "subject")

| Task | What it predicts | Typical inputs |
|---|---|---|
| **Lithology / lithofacies classification** | Rock type per depth (sandstone, shale/claystone, limestone, marl, coal…) | Gamma-ray, resistivity, density, neutron porosity, sonic logs |
| **Formation top / boundary prediction** | Depth of stratigraphic boundaries ("tops") | Well logs, drilling parameters, offset/typewell data |
| **Wellbore geology / TVT prediction** | Geology / True Vertical Thickness along a horizontal well | Horizontal-well logs + typewell correlation |
| **Look-ahead / ahead-of-bit prediction** | Geology *ahead of* the drill bit | Logs + seismic + deep-learning earth models |
| **Geosteering optimization** | Real-time trajectory adjustments to maximize reservoir contact | Live logs + reservoir model + (deep) RL |
| **Petrophysical property estimation** | Porosity, permeability, water saturation, etc. | Log suites |

## Contents

| File | Topic |
|---|---|
| [ML_Techniques_for_Wellbore_Geology.md](./ML_Techniques_for_Wellbore_Geology.md) | Methods: GBDT, CNN/LSTM/Transformer, GAN earth models, deep RL geosteering, feature engineering for logs. |
| [Well_Log_Datasets.md](./Well_Log_Datasets.md) | Public datasets & benchmarks (FORCE 2020, Volve, Geolink, and more). |
| [Tools_and_Software.md](./Tools_and_Software.md) | Geosteering/petrophysics software (ROGII StarSteer, Petrel, Techlog) and open-source Python (lasio, welly, dlisio). |
| [Key_Papers_and_Resources.md](./Key_Papers_and_Resources.md) | Research reading list and reviews. |

## Key log measurements (quick reference)

- **Gamma-ray (GR):** clay/shale content — high in shale, low in clean sand.
- **Resistivity:** hydrocarbon vs water saturation (high resistivity often → hydrocarbons).
- **Density (RHOB) & Neutron porosity (NPHI):** porosity & lithology (crossover patterns).
- **Sonic (DT):** acoustic slowness — porosity & mechanical properties.
- **Caliper, PEF, SP:** borehole condition, lithology discriminators.

## Cross-references in AIForge
- Energy AI: [`../`](../)
- Time series / sequence models: [`../../../02_LLM_AND_AI_MODELS/Time_Series_Models`](../../../02_LLM_AND_AI_MODELS/Time_Series_Models/)
- Datasets hub: [`../../../03_DATASETS_TOOLS_AND_RESOURCES/Datasets`](../../../03_DATASETS_TOOLS_AND_RESOURCES/Datasets/)
- Reinforcement learning: [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/)
