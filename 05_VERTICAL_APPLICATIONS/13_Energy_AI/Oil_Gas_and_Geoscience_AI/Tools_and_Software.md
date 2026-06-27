# Tools & Software for Wellbore Geology & Geosteering

Commercial geoscience platforms and open-source Python tooling for well-log interpretation, lithology prediction, and geosteering.

## Geosteering & interpretation platforms

| Software | Vendor | Focus |
|---|---|---|
| **StarSteer** | **ROGII** | Industry geosteering platform — real-time well placement, log correlation, 3D geosteering. https://rogii.com/ |
| **Solo Cloud / Solo** | ROGII | Cloud geosteering & collaboration. |
| **Petrel** | SLB (Schlumberger) | Seismic-to-simulation, geomodeling, petrophysics. |
| **Techlog** | SLB | Wellbore/petrophysics interpretation. |
| **Kingdom** | S&P Global | Seismic & well interpretation. |
| **Decision Space / DSG** | Halliburton/Landmark | Geoscience & geosteering. |
| **Geolog** | Emerson/Paradigm | Petrophysics & formation evaluation. |

## Open-source Python ecosystem

| Library | What it does | Link |
|---|---|---|
| **lasio** | Read/write **LAS** well-log files. | https://github.com/kinnala/lasio · https://lasio.readthedocs.io/ |
| **welly** | Wells, curves, striplog handling (built on lasio). | https://github.com/agilescientific/welly |
| **striplog** | Lithology/interval (striplog) data structures. | https://github.com/agilescientific/striplog |
| **dlisio** | Read **DLIS/LIS** binary logs. | https://github.com/equinor/dlisio |
| **segyio** | Read/write **SEG-Y** seismic. | https://github.com/equinor/segyio |
| **scikit-learn / LightGBM / XGBoost** | Baseline lithofacies classifiers. | — |
| **PyTorch / TensorFlow** | CNN/LSTM/Transformer log models. | — |
| **Bruges** | Geophysics utilities (rock physics, AVO). | https://github.com/agilescientific/bruges |

## Standards & data platforms

- **OSDU** — open subsurface data platform/standard — https://osduforum.org/
- **WITSML** — real-time drilling data exchange (geosteering inputs).
- **RESQML** — reservoir model exchange.
- **Energistics** — standards body — https://www.energistics.org/

## Typical open-source workflow

1. **Load** logs with `lasio` / `welly` (LAS) or `dlisio` (DLIS).
2. **QC & normalize** curves (despike, GR normalization, unit harmonization).
3. **Feature engineering** (windows, ratios, DWT, typewell alignment).
4. **Model** with LightGBM (baseline) → CNN/BiLSTM/Transformer (deep).
5. **Validate** with GroupKFold by well; calibrate; map predictions back to depth.
6. **Deliver** as facies/tops curves for geosteering / geomodeling.

## Sources
- [ROGII (StarSteer / Solo)](https://rogii.com/)
- [lasio docs](https://lasio.readthedocs.io/) · [welly](https://github.com/agilescientific/welly) · [dlisio](https://github.com/equinor/dlisio) · [segyio](https://github.com/equinor/segyio)
- [OSDU Forum](https://osduforum.org/) · [Energistics standards](https://www.energistics.org/)
