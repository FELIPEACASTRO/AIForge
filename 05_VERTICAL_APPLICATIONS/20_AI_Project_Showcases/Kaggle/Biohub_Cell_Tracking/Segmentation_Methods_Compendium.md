# Cell Segmentation Methods — Full Compendium

> An exhaustive, fact-checked index of **cell / nucleus instance-segmentation methods** relevant to cell tracking during development — the detection stage that feeds any tracker. Broader than the Biohub Kaggle competition: it covers the whole field so you can pick, ensemble, or fine-tune the best detector. All links verified live (July 2026).

Segmentation quality is the ceiling on tracking quality: a tracker can only link what the detector finds. For the [Biohub competition](./Competition_Overview_and_Scoring.md), remember the **7 µm node-matching** tolerance and the **over-detection penalty** — favor precise, well-calibrated detectors.

## Generalist deep-learning segmenters

| Method | What it is | Link |
|---|---|---|
| **Cellpose** | The generalist gold standard; flow-field based; 2D/3D | [github mouseland/cellpose](https://github.com/mouseland/cellpose) |
| **Cellpose3** | Adds one-click **image restoration** (denoise/deblur/upsample) before segmentation — big wins on noisy light-sheet | [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02595-5) |
| **Cellpose-SAM** | 2025 SAM-backed Cellpose with "superhuman generalization"; robust to noise/anisotropy/undersampling | [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1) |
| **Omnipose** | Extends Cellpose for elongated/bacterial and anisotropic shapes | [github kevinjohncutler/omnipose](https://github.com/kevinjohncutler/omnipose) |
| **InstanSeg** | Embedding-based instance segmentation; accurate, portable, ≥60% faster than alternatives | [arXiv 2408.15954](https://arxiv.org/abs/2408.15954) |

## Shape-model & embedding methods (great for nuclei)

| Method | Strength | Link |
|---|---|---|
| **StarDist / StarDist-3D** | Star-convex polygons/polyhedra — **best-in-class for convex nuclei**, esp. dense clusters | [github stardist/stardist](https://github.com/stardist/stardist) |
| **EmbedSeg** | Embedding-based, morphology/dimension-agnostic instance segmentation for microscopy | [github juglab/EmbedSeg](https://github.com/juglab/EmbedSeg) |
| **SplineDist** | StarDist generalized to non-convex shapes via splines | [github uhlmanngroup/splinedist](https://github.com/uhlmanngroup/splinedist) |
| **nnU-Net (3D)** | Self-configuring 3D U-Net; strong tune-free backbone; used on zebrafish light-sheet | [github MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) · [zebrafish paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11580962/) |

## Foundation models & image restoration helpers

| Tool | Role | Link |
|---|---|---|
| **CellSAM** | Foundation segmentation model (SAM + CellFinder detector) | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.17.567630.full.pdf) · [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02879-w) |
| **CSBDeep / CARE** | Content-aware image restoration — denoise before segmenting | [github CSBDeep/CSBDeep](https://github.com/CSBDeep/CSBDeep) |
| **Noise2Void (n2v)** | Self-supervised denoising, no clean targets needed | [github juglab/n2v](https://github.com/juglab/n2v) |

## Interactive & applied tools

- **Celldetective** — AI image-analysis app bundling StarDist/Cellpose + btrack for dynamic cell interactions ([bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.03.15.585250v1)).
- **ilastik** — interactive machine-learning pixel/object classification & tracking ([ilastik.org](https://www.ilastik.org/)).

## How to choose (for developmental 3D+t)

1. **Nuclei, dense, roughly convex** → StarDist-3D (often the accuracy winner).
2. **Noisy / anisotropic light-sheet** → Cellpose3 restoration + Cellpose-SAM, or CARE/N2V denoise first.
3. **Need speed/portability** → InstanSeg.
4. **No labels / few labels** → fine-tune via [ZeroCostDL4Mic / BiaPy](./Frameworks_Platforms_and_Extra_Datasets.md).
5. **Ensemble** detectors and calibrate thresholds to respect the competition's over-detection penalty.

## Related

- [Tracking Methods Compendium](./Tracking_Methods_Compendium.md) · [Frameworks, Platforms & Extra Datasets](./Frameworks_Platforms_and_Extra_Datasets.md) · [Surveys, Benchmarks & Key Papers](./Surveys_Benchmarks_and_Key_Papers.md) · [Playbook README](./README.md)
- Parent: [`../`](../) (Kaggle)

**Sources:** [Cellpose](https://github.com/mouseland/cellpose) · [Cellpose3](https://www.nature.com/articles/s41592-025-02595-5) · [Cellpose-SAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1) · [Omnipose](https://github.com/kevinjohncutler/omnipose) · [InstanSeg](https://arxiv.org/abs/2408.15954) · [StarDist](https://github.com/stardist/stardist) · [EmbedSeg](https://github.com/juglab/EmbedSeg) · [SplineDist](https://github.com/uhlmanngroup/splinedist) · [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) · [CSBDeep](https://github.com/CSBDeep/CSBDeep) · [n2v](https://github.com/juglab/n2v) · [ilastik](https://www.ilastik.org/)

**Keywords:** cell segmentation, nucleus instance segmentation, Cellpose, Cellpose3, Cellpose-SAM, Omnipose, StarDist, StarDist-3D, InstanSeg, EmbedSeg, SplineDist, nnU-Net, CellSAM, CARE, Noise2Void, image restoration, ilastik, Celldetective, 3D microscopy, light-sheet nuclei.
