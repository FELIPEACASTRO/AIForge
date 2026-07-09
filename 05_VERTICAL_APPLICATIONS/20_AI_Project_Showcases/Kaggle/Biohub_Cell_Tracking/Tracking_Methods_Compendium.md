# Cell Tracking Methods — Full Compendium

> An exhaustive, fact-checked index of **cell / particle tracking (data-association) methods** for development — classical linear-assignment, multi-hypothesis, graph/ILP, deep learning, GNN, transformer, and uncertainty-aware trackers. Broader than the Biohub Kaggle competition, so you can pick and ensemble the strongest linker. All links verified live (July 2026).

Data association across time is the **hardest and highest-scoring** part of cell tracking (the [Biohub metric](./Competition_Overview_and_Scoring.md) weights edges 10× over divisions). Below, methods are grouped by paradigm.

## Multi-hypothesis & overlap-based (state of the art for development)

| Method | Idea | Link |
|---|---|---|
| **Ultrack** ⭐ | Multi-hypothesis segmentation selection via ultrametric contour maps; terabyte-scale 3D+t; **the Biohub host's own tool** | [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0) · [docs](https://royerlab.github.io/ultrack/) · [github royerlab/ultrack](https://github.com/royerlab/ultrack) |
| **Cell Tracking according to Biological Needs** | Strong **mitosis-aware multi-hypothesis** tracker with aleatoric uncertainty | [arXiv 2403.15011](https://arxiv.org/abs/2403.15011) |

## Classical linear-assignment & particle tracking (still top-tier)

| Method | Idea | Link |
|---|---|---|
| **u-track / u-track3D** (Danuser Lab) | LAP-based multi-particle tracking; gap closing, merge/split; "still among the best after 15 years" | [github DanuserLab/u-track](https://github.com/DanuserLab/u-track) · [u-track3D (PubMed)](https://pubmed.ncbi.nlm.nih.gov/38113853/) |
| **TrackMate (LAP tracker)** | Fiji's extensible single-particle tracker (LAP after Jaqaman); integrates StarDist/Cellpose detectors | [imagej.net/plugins/trackmate](https://imagej.net/plugins/trackmate/) · [LAP trackers](https://imagej.net/plugins/trackmate/trackers/lap-trackers) · [TrackMate paper](https://www.sciencedirect.com/science/article/pii/S1046202316303346) |
| **TrackPy** | Crocker–Grier particle linking in Python | [github soft-matter/trackpy](https://github.com/soft-matter/trackpy) |
| **LapTrack** | Flexible LAP tracking as a Python library (custom cost functions) | [github yfukai/laptrack](https://github.com/yfukai/laptrack) |
| **btrack** | Bayesian multi-object tracker; fast motion models | [github quantumjot/btrack](https://github.com/quantumjot/btrack) |

## Graph / ILP / flow-based

| Method | Idea | Link |
|---|---|---|
| **EmbedTrack (KIT-GE)** | Single CNN for **joint segmentation + tracking** via offsets + clustering bandwidths; multiple CTC top-3/top-1 finishes | [arXiv 2204.10713](https://arxiv.org/abs/2204.10713) · [github kit-loe-ge/EmbedTrack](https://github.com/kit-loe-ge/EmbedTrack) |
| **PyUAT** | Open-source Python **uncertainty-aware tracking** framework, efficient & scalable | [arXiv 2503.21914](https://arxiv.org/abs/2503.21914) |

## Deep learning: GNN, transformer, end-to-end

| Method | Idea | Link |
|---|---|---|
| **Trackastra** | **Transformer** learning pairwise associations incl. divisions; **won the 7th Cell Tracking Challenge (ISBI 2024)** | [arXiv 2405.15700](https://arxiv.org/abs/2405.15700) · [github weigertlab/trackastra](https://github.com/weigertlab/trackastra) |
| **GNN for Cell Tracking** (Ben-Haim & Riklin-Raviv) | Whole time-lapse as a graph; deep metric learning + edge classifier; SOTA on 2D/3D | [ECCV 2022 / arXiv 2202.04731](https://arxiv.org/abs/2202.04731) |
| **DeepCell / Caliban** (Van Valen Lab) | DL segmentation + tracking + lineage; Siamese/association nets | [github vanvalenlab/deepcell-tf](https://github.com/vanvalenlab/deepcell-tf) |
| **DeLTA** | DL segmentation, tracking & **lineage reconstruction** (esp. bacteria/mother-machine, generalizes) | [PLOS Comp Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673) |
| **Cell-as-Point (CAP)** | One-stage, efficient point-based cell tracking | [arXiv 2411.14833](https://arxiv.org/abs/2411.14833) |
| **ELEPHANT** | Incremental deep learning for 3D lineages from sparse annotations (proofreading loop) | [github elephant-track/elephant](https://github.com/elephant-track/elephant) |

## Uncertainty & robustness

- **"How to make your cell tracker say 'I dunno'"** — calibrated uncertainty for tracking decisions ([arXiv 2503.09244](https://arxiv.org/abs/2503.09244)). Useful to prune low-confidence edges and avoid the over-detection penalty.
- **Segment Anything for Cell Tracking** — zero-shot SAM2-based tracking without fine-tuning ([arXiv 2509.09943](https://arxiv.org/abs/2509.09943)).

## Large-scale annotation / proofreading

- **Mastodon** — big-data cell tracking & lineaging in Fiji (successor to MaMuT/TrackMate) ([github mastodon-sc/mastodon](https://github.com/mastodon-sc/mastodon)).
- **CellTracksColab** — pool, analyze & visualize tracking results across datasets ([github CellMigrationLab/CellTracksColab](https://github.com/CellMigrationLab/CellTracksColab)).

## Practical guidance

1. **Baseline:** run **Ultrack** (host tool) + **Trackastra**; ensemble edges.
2. **Add gap recovery** (u-track/LapTrack-style) to re-link across missed frames within 7 µm.
3. **Divisions:** use a mitosis-aware tracker (Biological-Needs, Trackastra) for the +0.1 division term.
4. **Prune with uncertainty** to keep precision high.

## Related

- [Segmentation Methods Compendium](./Segmentation_Methods_Compendium.md) · [Frameworks, Platforms & Extra Datasets](./Frameworks_Platforms_and_Extra_Datasets.md) · [Surveys, Benchmarks & Key Papers](./Surveys_Benchmarks_and_Key_Papers.md) · [Playbook README](./README.md)
- Parent: [`../`](../) (Kaggle)

**Sources:** [Ultrack](https://royerlab.github.io/ultrack/) · [Biological Needs tracker](https://arxiv.org/abs/2403.15011) · [u-track](https://github.com/DanuserLab/u-track) · [u-track3D](https://pubmed.ncbi.nlm.nih.gov/38113853/) · [TrackMate](https://imagej.net/plugins/trackmate/) · [trackpy](https://github.com/soft-matter/trackpy) · [laptrack](https://github.com/yfukai/laptrack) · [btrack](https://github.com/quantumjot/btrack) · [EmbedTrack](https://arxiv.org/abs/2204.10713) · [PyUAT](https://arxiv.org/abs/2503.21914) · [Trackastra](https://arxiv.org/abs/2405.15700) · [GNN tracking](https://arxiv.org/abs/2202.04731) · [deepcell-tf](https://github.com/vanvalenlab/deepcell-tf) · [DeLTA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673) · [Cell-as-Point](https://arxiv.org/abs/2411.14833) · [ELEPHANT](https://github.com/elephant-track/elephant) · [uncertainty](https://arxiv.org/abs/2503.09244) · [SAM tracking](https://arxiv.org/abs/2509.09943) · [Mastodon](https://github.com/mastodon-sc/mastodon) · [CellTracksColab](https://github.com/CellMigrationLab/CellTracksColab)

**Keywords:** cell tracking, data association, Ultrack, Trackastra transformer, u-track LAP, TrackMate, trackpy, laptrack, btrack, EmbedTrack KIT-GE, graph neural network tracking, DeepCell Caliban, DeLTA lineage, PyUAT uncertainty, ELEPHANT, Mastodon, mitosis-aware multi-hypothesis, gap closing, cell division tracking, SAM2 tracking.
