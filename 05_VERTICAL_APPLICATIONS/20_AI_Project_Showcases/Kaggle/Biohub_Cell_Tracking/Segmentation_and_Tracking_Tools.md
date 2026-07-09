# Biohub Cell Tracking — Segmentation & Tracking Tools

> The best open-source tools for each stage of a 3D cell-tracking pipeline for the Kaggle **Biohub — Cell Tracking During Development** competition: detection/segmentation → linking/tracking → post-processing. All repositories and papers verified live (July 2026).

A winning pipeline is usually **two stages**: (1) detect/segment cell nuclei per 3D frame, then (2) link detections across time into tracks (with divisions). The competition's evaluation weights **linking (edges) 10× over divisions**, so invest most in stages 1–2.

## Stage 1 — Detection & segmentation (3D nuclei)

| Tool | Why it fits | Link |
|---|---|---|
| **Cellpose-SAM** | 2025 generalist segmentation with "superhuman generalization"; robust to noise, anisotropic blur, undersampling — ideal for light-sheet; 3D support; Cellpose4/cpdino models | [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1) · [github MouseLand/cellpose](https://github.com/mouseland/cellpose) |
| **StarDist (StarDist-3D)** | Star-convex nuclei; benchmarked as the most accurate of 7 CNNs for 3D embryo nuclei — strong when nuclei are roughly convex | [github stardist/stardist](https://github.com/stardist/stardist) |
| **nnU-Net (3D)** | Self-configuring 3D U-Net; already applied to zebrafish light-sheet nuclei tracking; a strong, tune-free segmentation backbone | [github MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) · [zebrafish nnU-Net paper (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11580962/) |

**Tips:** handle **Z-anisotropy** explicitly (voxel spacing differs axially vs laterally); calibrate the detection threshold so you don't trip the over-detection penalty; ensembling detectors (Cellpose-SAM + StarDist-3D) can improve recall.

## Stage 2 — Linking & tracking (the score driver)

| Tool | Approach | Link |
|---|---|---|
| **Ultrack** ⭐ | **The competition host's own tool** — multi-hypothesis tracking that selects segments by maximizing overlap across frames; terabyte-scale 3D+t; works with or without deep learning. Expect a strong baseline out of the box | [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0) · [docs](https://royerlab.github.io/ultrack/) · [github royerlab/ultrack](https://github.com/royerlab/ultrack) |
| **Trackastra** | **Transformer** that learns pairwise cell associations (incl. divisions); **won the 7th Cell Tracking Challenge (ISBI 2024)**; best on 13 datasets | [ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73116-7_27) · [arXiv 2405.15700](https://arxiv.org/abs/2405.15700) · [github weigertlab/trackastra](https://github.com/weigertlab/trackastra) |
| **DeepCell / Caliban** | Deep-learning tracking (Van Valen Lab); good nucleus-centric tracker with division handling | [github vanvalenlab/deepcell-tf](https://github.com/vanvalenlab/deepcell-tf) |
| **btrack** | Bayesian multi-object tracker; fast, motion-model based; good classical baseline / ensemble member | [github quantumjot/btrack](https://github.com/quantumjot/btrack) |

**Tips:** run **Ultrack** (host tool, hard to beat as a baseline) and/or **Trackastra**, then **ensemble the edges**; add **gap recovery** to re-link tracks broken across frames (within the 7 µm matching radius). A light division detector (±1 frame) scrapes the +0.1 division term.

## Related infrastructure

- **Ultrack + inTRACKtive** ship as a pair for tracking + interactive 3D visualization ([inTRACKtive — Nature Methods 2025](https://www.nature.com/articles/s41592-025-02777-1) · [github royerlab/inTRACKtive](https://github.com/royerlab/inTRACKtive)).
- The imaging modality behind the data is single-objective light-sheet — see **DaXi** ([Nature Methods 2022](https://www.nature.com/articles/s41592-022-01417-2) · [github royerlab/daxi](https://github.com/royerlab/daxi)).

## Related

- [Competition Overview & Scoring](./Competition_Overview_and_Scoring.md) · [Datasets, Metrics & Notebooks](./Datasets_Metrics_and_Notebooks.md) · [Playbook README](./README.md)
- Parent: [`../`](../) (Kaggle)

**Sources:** [Cellpose-SAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1) · [cellpose](https://github.com/mouseland/cellpose) · [stardist](https://github.com/stardist/stardist) · [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) · [Ultrack](https://royerlab.github.io/ultrack/) · [Trackastra arXiv](https://arxiv.org/abs/2405.15700) · [trackastra repo](https://github.com/weigertlab/trackastra) · [deepcell-tf](https://github.com/vanvalenlab/deepcell-tf) · [btrack](https://github.com/quantumjot/btrack)

**Keywords:** 3D cell segmentation, Cellpose-SAM, StarDist-3D, nnU-Net, cell tracking, Ultrack, Trackastra transformer tracking, DeepCell Caliban, btrack Bayesian tracking, light-sheet nuclei, gap recovery, cell division detection, Kaggle Biohub pipeline.
