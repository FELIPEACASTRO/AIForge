# Surveys, Benchmarks & Key Papers

> The literature backbone for cell tracking during development — review articles to get the lay of the land, the community benchmark, and the landmark methods papers. Read these to understand *why* methods work and where the field is going. All links verified live (July 2026).

## Reviews & surveys (start here)

| Survey | Scope | Link |
|---|---|---|
| **A survey of deep learning methods on cell instance segmentation** | 198 papers (2020–2024): CNNs, encoder–decoder, RNNs, transformers, GANs | [Neural Computing & Applications 2025](https://link.springer.com/article/10.1007/s00521-025-11119-3) |
| **The Cell Tracking Challenge: 10 years of objective benchmarking** | The definitive benchmark retrospective; metrics, datasets, method landscape | [Nature Methods 2023](https://www.nature.com/articles/s41592-023-01879-y) |

## The benchmark

- **Cell Tracking Challenge (CTC)** — ongoing objective benchmark (since ISBI 2013) with annotated 2D/3D+t datasets, standardized metrics (SEG, DET, TRA, and the newer **CHOTA**), and a public leaderboard. Datasets: [3D+t](https://celltrackingchallenge.net/3d-datasets/) · results: [latest CTB](https://celltrackingchallenge.net/latest-ctb-results/). Benchmark insight: **no significant TRA difference** between ML and non-ML *linking* — detection + association design matters more than the linker family.

## Landmark & innovative methods papers

**Multi-hypothesis / large-scale:**
- **Ultrack** — pushing cell tracking across scales ([Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0); [arXiv 2308.04526](https://arxiv.org/pdf/2308.04526)).
- **Cell Tracking according to Biological Needs** — mitosis-aware multi-hypothesis + uncertainty ([arXiv 2403.15011](https://arxiv.org/abs/2403.15011)).

**Classical (still competitive):**
- **u-track** — robust single-particle tracking (LAP), Jaqaman et al., the reference method ([github DanuserLab/u-track](https://github.com/DanuserLab/u-track); [u-track3D](https://pubmed.ncbi.nlm.nih.gov/38113853/)).
- **TrackMate** — extensible single-particle tracking platform ([Methods 2017](https://www.sciencedirect.com/science/article/pii/S1046202316303346)).

**Deep learning / GNN / transformer:**
- **Trackastra** — transformer cell tracking, CTC-2024 winner ([arXiv 2405.15700](https://arxiv.org/abs/2405.15700)).
- **GNN for Cell Tracking** — graph + deep metric learning + edge classification ([arXiv 2202.04731](https://arxiv.org/abs/2202.04731)).
- **EmbedTrack** — joint segmentation + tracking via offsets/bandwidths ([arXiv 2204.10713](https://arxiv.org/abs/2204.10713)).
- **DeLTA** — DL segmentation, tracking & lineage reconstruction ([PLOS Comp Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673)).
- **Cell-as-Point** — one-stage efficient tracking ([arXiv 2411.14833](https://arxiv.org/abs/2411.14833)).
- **PyUAT** — uncertainty-aware tracking framework ([arXiv 2503.21914](https://arxiv.org/abs/2503.21914)).
- **Segment Anything for Cell Tracking** — zero-shot SAM2 ([arXiv 2509.09943](https://arxiv.org/abs/2509.09943)).

**Developmental atlases & imaging (context for the data):**
- **Zebrahub** — multimodal zebrafish developmental atlas ([Cell 2024](https://www.sciencedirect.com/science/article/pii/S0092867424011474)).
- **DaXi** — single-objective light-sheet microscope behind the imaging ([Nature Methods 2022](https://www.nature.com/articles/s41592-022-01417-2)).

**Segmentation cornerstones:**
- **Cellpose3** — one-click restoration for segmentation ([Nature Methods 2025](https://www.nature.com/articles/s41592-025-02595-5)).
- **InstanSeg** — embedding-based, portable segmentation ([arXiv 2408.15954](https://arxiv.org/abs/2408.15954)).
- **LIVECell** — large label-free segmentation dataset ([Nature Methods 2021](https://www.nature.com/articles/s41592-021-01249-6)).

## Reading path

1. Skim the two **surveys** for the landscape.
2. Read **Ultrack** + **Trackastra** (current SOTA the competition builds on).
3. Read **u-track** / **TrackMate** to understand the classical LAP backbone (gap closing, merge/split) that still wins.
4. Study **CTC metrics** so your local evaluation matches how tracking is judged.

## Related

- [Segmentation Methods Compendium](./Segmentation_Methods_Compendium.md) · [Tracking Methods Compendium](./Tracking_Methods_Compendium.md) · [Frameworks, Platforms & Extra Datasets](./Frameworks_Platforms_and_Extra_Datasets.md) · [Playbook README](./README.md)
- Parent: [`../`](../) (Kaggle)

**Sources:** [Segmentation survey](https://link.springer.com/article/10.1007/s00521-025-11119-3) · [CTC 10 years](https://www.nature.com/articles/s41592-023-01879-y) · [CTC results](https://celltrackingchallenge.net/latest-ctb-results/) · [Ultrack](https://www.nature.com/articles/s41592-025-02778-0) · [Biological Needs](https://arxiv.org/abs/2403.15011) · [u-track](https://github.com/DanuserLab/u-track) · [TrackMate](https://www.sciencedirect.com/science/article/pii/S1046202316303346) · [Trackastra](https://arxiv.org/abs/2405.15700) · [GNN tracking](https://arxiv.org/abs/2202.04731) · [EmbedTrack](https://arxiv.org/abs/2204.10713) · [DeLTA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673) · [Cell-as-Point](https://arxiv.org/abs/2411.14833) · [PyUAT](https://arxiv.org/abs/2503.21914) · [SAM tracking](https://arxiv.org/abs/2509.09943) · [Zebrahub](https://www.sciencedirect.com/science/article/pii/S0092867424011474) · [DaXi](https://www.nature.com/articles/s41592-022-01417-2) · [Cellpose3](https://www.nature.com/articles/s41592-025-02595-5) · [InstanSeg](https://arxiv.org/abs/2408.15954) · [LIVECell](https://www.nature.com/articles/s41592-021-01249-6)

**Keywords:** cell tracking survey, cell instance segmentation review, Cell Tracking Challenge, CHOTA metric, Ultrack, Trackastra, u-track, TrackMate, graph neural network tracking, EmbedTrack, DeLTA, PyUAT, Zebrahub, DaXi, Cellpose3, InstanSeg, LIVECell, landmark papers, developmental biology imaging, benchmark.
