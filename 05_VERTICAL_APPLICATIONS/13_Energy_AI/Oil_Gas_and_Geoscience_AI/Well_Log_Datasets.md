# Public Well-Log & Subsurface Datasets

Open datasets for machine learning on wellbore geology, lithology/lithofacies classification, and petrophysics.

## Benchmark datasets

| Dataset | Contents | Use case | Link |
|---|---|---|---|
| **FORCE 2020** | 118 wells (Norwegian Sea), train / public-test / blind-test; hand-crafted lithofacies & lithology labels; GR, neutron, sonic, density, etc. | The reference **lithology classification benchmark** (FORCE 2020 ML competition). | [Zenodo 4351156](https://zenodo.org/records/4351156) |
| **Volve** (Equinor, 2018) | ~40,000 files: well logs, geological models, seismic, production — 10 years of a full field (~200 km off Norway). | End-to-end subsurface ML, geomodeling. | [Equinor Volve data](https://www.equinor.com/energy/volve-data-sharing) |
| **Geolink** | North Sea wells with lithology/stratigraphy interpretation. | Lithology classification benchmarking (often paired with FORCE). | [discussion via TDS](https://towardsdatascience.com/public-datasets-for-machine-learning-in-geoscience-cf880862300a/) |
| **Well Logs Dataset for ML (Kaggle)** | Cleaned well-log tables for ML. | Quick experimentation. | [Kaggle](https://www.kaggle.com/datasets/faresazzam/well-logs-dataset-for-machine-learning/data) |

## Other open sources

- **NLOG (Netherlands)** — public Dutch subsurface & well data — https://www.nlog.nl/en
- **NPD / Sodir FactPages (Norway)** — Norwegian Offshore Directorate well data — https://factpages.sodir.no/
- **USGS / state geological surveys** — US well logs & core data — https://www.usgs.gov/
- **SEG Open Data** — seismic & geophysics — https://wiki.seg.org/wiki/Open_data
- **OSDU Data Platform** — industry open subsurface data standard — https://osduforum.org/

## File formats you'll meet

- **LAS** (Log ASCII Standard) — the common well-log text format.
- **DLIS** — binary log format (use `dlisio`).
- **SEG-Y** — seismic data.
- **RESQML / WITSML** — reservoir model & real-time drilling data standards.

## Tips for ML-ready data

- Different wells carry **different curve suites** — reconcile to a shared subset.
- **Normalize GR** and other tool-sensitive logs across wells/fields.
- Labels (facies/tops) are **interpreted by geoscientists** — treat them as gold but imperfect.
- Respect **well-level splits** (GroupKFold) to avoid optimistic leakage.

## Sources
- [FORCE 2020 dataset (Zenodo)](https://zenodo.org/records/4351156)
- [Volve (Equinor)](https://www.equinor.com/energy/volve-data-sharing)
- [Public Datasets for ML in Geoscience (TDS)](https://towardsdatascience.com/public-datasets-for-machine-learning-in-geoscience-cf880862300a/)
- [Lithology Classification Benchmark for ML (Mathematical Geosciences)](https://link.springer.com/article/10.1007/s11004-026-10300-1)
