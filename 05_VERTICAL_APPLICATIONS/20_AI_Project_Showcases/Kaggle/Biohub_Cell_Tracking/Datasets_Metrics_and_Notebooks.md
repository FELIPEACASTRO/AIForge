# Biohub Cell Tracking — Datasets, Metrics & Notebooks

> Pretraining datasets, local evaluation libraries, public Kaggle notebooks, and a concrete winning recipe for the **Biohub — Cell Tracking During Development** competition. All links verified live (July 2026).

## Local evaluation (iterate without spending submissions)

Replicate the leaderboard metric offline so you can tune freely:

| Library | Use | Link |
|---|---|---|
| **traccuracy** | Python suite for cell-tracking accuracy metrics (loaders, matchers, metrics; CLI + API); reconstruct the edge/division scoring locally | [github](https://github.com/live-image-tracking-tools/traccuracy) · [docs](https://traccuracy.readthedocs.io/) |
| **py-ctcmetrics** | Cell Tracking Challenge metrics + the **CHOTA** higher-order accuracy metric | [github CellTrackingChallenge/py-ctcmetrics](https://github.com/CellTrackingChallenge/py-ctcmetrics) |

> The official scoring code lives in the competition repo ([royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition)); mirror its edge-Jaccard + 0.1·division-Jaccard formula in your local harness for an exact proxy.

## Pretraining & augmentation datasets

More labeled 3D+t nuclei data helps detectors and trackers generalize:

| Dataset | Content | Link |
|---|---|---|
| **Zebrahub** | Same organism/modality — zebrafish light-sheet developmental atlas (imaging + scRNA-seq); the most on-distribution external data | [zebrahub.sf.czbiohub.org](https://zebrahub.sf.czbiohub.org/) · [Cell 2024 paper](https://www.sciencedirect.com/science/article/pii/S0092867424011474) · [data on Zenodo](https://zenodo.org/records/13761503) |
| **Cell Tracking Challenge (3D+t)** | Curated, annotated 3D+t datasets (e.g. Drosophila/Tribolium embryos) + the field's benchmark | [celltrackingchallenge.net/3d-datasets](https://celltrackingchallenge.net/3d-datasets/) · [CTC — Nature Methods 2023](https://www.nature.com/articles/s41592-023-01879-y) |
| **BlastoSPIM** | Largest 3D nuclei ground-truth for preimplantation mouse embryos (573 images, 11,708 nuclei); StarDist-3D models included | [blastospim.flatironinstitute.org](https://blastospim.flatironinstitute.org/html/) |
| **DynamicNuclearNet** (DeepCell) | ~130 movies, ~600k nuclei, ~2k divisions — large tracking training set | [bioRxiv](https://www.biorxiv.org/content/10.1101/803205v4.full) |

## Public Kaggle notebooks (study & fork)

| Notebook | What it teaches | Link |
|---|---|---|
| **Data Model, EDA, Baseline** | How the OME-Zarr images + `tracksdata` graph work; end-to-end baseline | [pilkwang](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline) |
| **Learned Graph w/ Gap Recovery** | Stronger learned linking with track-gap recovery | [pilkwang](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery) |
| **Classical Baseline** | Non-deep baseline pipeline | [xiaoleilian](https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline) |
| **Getting Started — Nearest Neighbor** | Minimal official-staff starter | [inversion](https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor) |

## Concrete winning recipe

1. **Start from** the official baseline repo and fork the **Learned Graph w/ Gap Recovery** notebook.
2. **Detection:** swap in / ensemble **Cellpose-SAM 3D** or **StarDist-3D**; calibrate the threshold so `T_pred ≈ T_true` (avoid the over-detection penalty); respect Z-anisotropy.
3. **Linking:** run **Ultrack** (multi-hypothesis, host tool) and/or **Trackastra** (transformer); **ensemble the edges**; add **gap recovery** to re-link tracks within 7 µm.
4. **Local optimization:** score with **traccuracy** replicating `edge_jaccard + 0.1·division_jaccard` before every submission.
5. **Divisions:** add a lightweight ±1-frame division detector to capture the +0.1 term — never at the cost of edge precision.
6. **External data:** pretrain detectors on **Zebrahub** (on-distribution), **CTC 3D+t**, and **BlastoSPIM** for robustness.

## Related

- [Competition Overview & Scoring](./Competition_Overview_and_Scoring.md) · [Segmentation & Tracking Tools](./Segmentation_and_Tracking_Tools.md) · [Playbook README](./README.md)
- Parent: [`../`](../) (Kaggle)

**Sources:** [traccuracy](https://github.com/live-image-tracking-tools/traccuracy) · [py-ctcmetrics](https://github.com/CellTrackingChallenge/py-ctcmetrics) · [Zebrahub](https://zebrahub.sf.czbiohub.org/) · [CTC 3D datasets](https://celltrackingchallenge.net/3d-datasets/) · [BlastoSPIM](https://blastospim.flatironinstitute.org/html/) · [DynamicNuclearNet](https://www.biorxiv.org/content/10.1101/803205v4.full) · [notebook: EDA baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline) · [notebook: gap recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)

**Keywords:** cell tracking datasets, traccuracy, py-ctcmetrics, CHOTA, Zebrahub, Cell Tracking Challenge, BlastoSPIM, DynamicNuclearNet, Kaggle notebooks, gap recovery, pretraining nuclei segmentation, Biohub competition playbook, leaderboard optimization.
