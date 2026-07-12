# Solar Filament Segmentation Methods

> Methods built **specifically for solar H-alpha filaments, prominences and fibrils** — from the classical thresholding-and-skeleton pipelines to modern U-Net / transformer segmentation, barb-and-chirality recognition, detection-classification-tracking systems, 304 A prominence detection, and heliophysics foundation models. Anchored to the [IEEE Big Data Cup 2026 — Pixel-Precise Segmentation of Solar Filaments](https://bigdataieee.org/BigData2026/cup/) on the [MAGFiLO](https://www.nature.com/articles/s41597-024-03876-y) GONG dataset. (Cross-domain thin-structure architectures, topology losses, instance-separation and metrics live on the sibling pages.)

## Deep-learning filament segmentation (full-disk H-alpha)

The core segmentation backbones, in roughly chronological order. Nearly all are U-Net descendants.

| Method | What / why |
|---|---|
| [Solar Filament Recognition via U-Net (Zhu et al. 2019)](https://arxiv.org/abs/1909.06580) | First end-to-end U-Net for extracting filaments from full-disk H-alpha (BBSO); code + model released. The foundational DL baseline the whole field builds on. |
| [Toward Filament Segmentation Using DNNs (Mask R-CNN, Ahmadzadeh et al. 2019)](https://arxiv.org/abs/1912.02743) | Instance segmentation of filaments with Mask R-CNN on BBSO H-alpha, labels drawn from HEK; early instance-aware approach from the same GSU/MLEco group behind MAGFiLO and the Cup. |
| [Improved U-Nets: ASPP U-Net + CGAN (Liu et al. 2021)](https://link.springer.com/article/10.1007/s11207-021-01920-3) | Adds Atrous Spatial Pyramid Pooling, modified padding and an expanded pathway to cut low-level noise; ASPP-UNet best on high-quality H-alpha, a CGAN variant best on low-quality/large data. |
| [AA-UNet (axial-attention encoder U-Net)](https://ieeexplore.ieee.org/document/10046547) | Replaces encoder conv blocks with axial-attention blocks to model non-adjacent-pixel context; robust to uneven image quality (Jaccard 0.63, MCC 0.77, F1 0.77). A named target architecture for the Cup. |
| [Attention U2-Net (nested residual U-blocks, Jiang & Li 2024)](https://www.mdpi.com/2218-1997/10/10/381) | Adds attention gates to a nested U2-Net (residual U-blocks) for multi-scale, high-precision filament recognition — a strong deep-supervision architecture to try on MAGFiLO. |
| [Transformer + CNN with multi-scale residual block (Applied Sciences 2024)](https://www.mdpi.com/2076-3417/14/9/3745) | Injects a Transformer into a CNN for global context plus multi-scale residual blocks; F1 91.2%, robust to sunspot interference. Beats improved U-Net / V-Net baselines. |
| [Improved VNet-based filament detection (ChinaXiv 2021)](https://chinaxiv.org/abs/202103.00106V1) | Builds a BBSO H-alpha + magnetogram dataset and uses an improved VNet to recover weak/faint filaments that thresholding misses. |
| [Flat U-Net (SCA/CSA ultralightweight, Zhu et al. 2025)](https://arxiv.org/abs/2502.07259) | Ultralightweight (~0.25 MB) U-Net with Simplified Channel Attention + Channel Self-Attention blocks; precision ~0.93, Dice 0.82 on full-disk H-alpha. Open data/models/code — a strong efficient baseline for the Cup. |
| [EdgeAttNet — barb-aware filament segmentation (2025)](https://arxiv.org/abs/2509.02964) | U-Net backbone with a learnable edge map injected into bottleneck self-attention (edge-transformed Key/Query) to capture fine barbs; benchmarked directly on MAGFiLO and beats U-Net / U-Net-transformer baselines. The most directly on-Cup recent method. |
| [Compound U-Net — multiscale feature extraction (HAS / MHAS)](https://zenodo.org/records/17230605) | Solar filament segmentation/detection network with released code and the HAS / MHAS datasets. |

## Detection, classification & tracking pipelines

Systems that combine detectors, segmenters and cross-day trackers — the full "derived tasks" the Cup implies (detect, classify, follow across the disk passage).

| System | What / why |
|---|---|
| [Semi-supervised YOLOv5 + U-Net (Diercke et al. 2024, A&A)](https://arxiv.org/abs/2402.15407) | Two-stage: YOLOv5 detection on ChroTel, then semi-supervised U-Net pixel segmentation, generalized to the full GONG archive; ~92% accuracy with reduced sunspot false positives. The canonical semi-supervised filament pipeline. |
| [DETR + U-Net + tracking — SWEFil (Reche & Cid 2025, Acta Astronautica)](https://doi.org/10.1016/j.actaastro.2025.09.053) | DETR for detection/classification, U-Net for instance segmentation, and a custom tracker on GONG H-alpha; introduces the SWEFil space-weather dataset. |
| [Filament detection, tracking & analysis on CHASE via ML (Zhu et al. 2024)](https://arxiv.org/abs/2402.14209) | U-Net + tracking on space-based CHASE H-alpha with a released annotated dataset (via SSDC-NJU) and code — a second, independent labeled filament set for transfer/robustness. |
| [Automation of filament tracking in the HELIO project](https://inspirehep.net/literature/1088640) | Automated solar-filament detection and cross-image tracking pipeline (Solar Physics; arXiv:1202.2072). |
| [Bidirectional Autoregressive Tracking of Solar Filaments](https://europepmc.org/article/PPR/PPR1180242) | Preprint on autoregressive forward/backward tracking of filaments through their disk passage. |
| [Using Deep Learning to Detect and Trace H-alpha Fibrils](https://ui.adsabs.harvard.edu/abs/2021AAS...23811315J/abstract) | Detection and tracing of fibrils — the thin curvilinear chromospheric cousins of filaments. |
| [Solar Event Detection with DL object detection (Baek et al. 2021)](https://ui.adsabs.harvard.edu/abs/2021SoPh..296..160B/abstract) | Object-detection deep learning applied to solar events including filaments; supporting detection reference. |
| [Solar Filament Detection Using Deep Learning (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4849934) | Applied deep-learning filament detection study. |
| [A Machine Learning Ecosystem for Filament Detection (MLEco)](https://doi.org/10.6084/m9.figshare.29647943) | The NSF/NSO GONG filament classification-localization-segmentation project (lead A. Ahmadzadeh) that produced MAGFiLO and the Cup. |

## Chirality, spine & barb recognition

Turning a mask into physics: medial spine, left/right-bearing barbs, and dextral/sinistral chirality — the fine-detail labels MAGFiLO encodes and the Cup rewards.

| Work | What / why |
|---|---|
| [Automatic Classification of Magnetic Chirality from H-alpha (Chalmers & Ahmadzadeh 2025)](https://arxiv.org/abs/2509.18214) | First reproducible chirality-classification baseline built directly on MAGFiLO; documents how the labels are consumed and split. From the same lab as the Cup. |
| [Filament chirality from footpoint / barb-bearing (Hao et al. 2015)](https://arxiv.org/abs/1506.08490) | Concrete algorithm: Dijkstra shortest-path spine recognition + connected-component barb labeling + barb-to-spine angle for left/right bearing, cross-checked with HMI polarity/PIL. Core recipe for spine + barb + chirality. |
| [Automated spine detection in H-alpha filaments (Aggarwal et al.)](http://article.sapub.org/10.5923.j.astronomy.20130204.02.html) | Dedicated spine-extraction algorithm (skeletonization/thinning of the segmented mask into a medial polyline). |

## Classical filament detection & characterization (pre-deep-learning)

Thresholding, region growing, morphology and graph skeletons — still valuable as strong non-DL baselines, weak-label priors and pre/post-processing.

| Method | What / why |
|---|---|
| [AAFDCC — Advanced Automated Filament Detection & Characterization (Bernasconi et al. 2005)](https://link.springer.com/article/10.1007/s11207-005-2766-y) | Foundational classical pipeline for H-alpha filament detection, spine, chirality and cross-image tracking (ran daily on BBSO). The algorithmic baseline MAGFiLO improves upon. |
| [Region-growing + image cleaning on Meudon (Fuller, Aboudarham & Bentley 2005)](https://link.springer.com/article/10.1007/s11207-005-8364-1) | Seminal cleaning + region-growing segmentation + pruned-skeleton description — the reference classical baseline for filament segmentation. |
| [Thresholding + SVM + morphology (Qu, Shih, Jing & Wang 2005)](https://link.springer.com/article/10.1007/s11207-005-5780-1) | SIDE enhancement, adaptive thresholding, an SVM to separate sunspots from filaments, and morphological thinning for spines on BBSO H-alpha. |
| [Filament recognition with a neural network (Zharkova & Schetinin 2005)](https://link.springer.com/article/10.1007/s11207-005-5622-1) | Early neural-network approach to filament recognition in H-alpha — the historical ML precursor before the U-Net era. |
| [Adaptive thresholding + graph skeleton (Yuan et al. 2011)](https://link.springer.com/article/10.1007/s11207-011-9798-2) | Hough disk detection, luminance correction, adaptive thresholding and morphological thinning + graph theory for spines across four observatories (>95% by number, >99% by area). |
| [Context-based sliding-window segmentation (IJACSA 2018)](https://doi.org/10.14569/ijacsa.2018.090538) | Sliding-window contextual thresholding for filament pixels in H-alpha — a lightweight classical segmenter. |
| [Advanced automated recognition on MLSO H-alpha, Cycle 23 (Hao, Fang & Chen 2013)](https://arxiv.org/abs/1303.6367) | Morphological detection + spine extraction over a full solar cycle; extracts position, area, spine and traces daily evolution. Basis for the later chirality/barb work. |
| [Automated detection of solar activities & filament statistics (Hao, Chen & Fang, IAUS340)](https://arxiv.org/abs/1804.03320) | Versatile automated detection returning position, area and spine, and tracing daily evolution across nearly three solar cycles. |
| [Active Contours Without Edges (ACWE) filament detection (Bandyopadhyay & Pant 2024)](https://arxiv.org/abs/2412.20749) | Variational/energy-based (Chan-Vese) segmentation with pre/post-processing — a modern non-DL baseline to benchmark deep models against. |
| [Filament_AutoDetect (janandd)](https://github.com/janandd/Filament_AutoDetect) | Open IDL code: local + size thresholding to binarize, label individual filaments, and track them across sequential daily images (length, size, fragments, centroid). A runnable reference implementation. |
| [Automated catalog of filament features 1988–2013 (Hao, Fang & Chen)](https://arxiv.org/abs/1511.04692) | The Nanjing-University automated BBSO/MLSO catalog spanning ~3 solar cycles (ApJS 221, 33): spine, area, length, tilt. Reference labels/method predating DL; good for weak-label priors. |

## Prominence detection (SDO/AIA 304 A, limb view)

Filaments seen against the dark sky beyond the limb are prominences; 304 A gives the multi-wavelength companion view.

| Work | What / why |
|---|---|
| [DL detection of prominences & AR features in 304 A (Zhou et al. 2024, ApJS)](https://ui.adsabs.harvard.edu/abs/2024ApJS..272....5Z) | Deep-learning detection of prominences and active-region features in SDO/AIA 304 A over Cycle 24 — the on-limb, multi-wavelength counterpart to on-disk H-alpha filaments. |
| [Automated 304 A prominence-eruption detection (Yashiro et al. 2020)](https://arxiv.org/abs/2005.11363) | Fully automated pipeline (polar transform + intensity ratio + monotonic height rise) turning detections into eruption alerts; live catalog at CDAW autoPE. |

## Heliophysics foundation models (a transfer route for filaments)

Self-supervised pretraining on large unlabeled solar archives, then fine-tuning for the label-scarce filament task.

| Model | What / why |
|---|---|
| [Surya — heliophysics foundation model (NASA–IBM 2025)](https://arxiv.org/abs/2508.14112) | 366M-param spatiotemporal transformer (spectral gating + long-short attention) pretrained on ~218 TB of SDO AIA/HMI; fine-tunable for active-region segmentation and other pixel tasks. The foundation-model route for filaments. |
| [SuryaBench — heliophysics ML benchmark](https://arxiv.org/abs/2508.14107) | Companion benchmark defining standardized downstream heliophysics/space-weather tasks and splits — a template for evaluating FM transfer to solar segmentation. |
| [SDO-FM — a foundation model for the Solar Dynamics Observatory (2024)](https://arxiv.org/abs/2410.02530) | FDL/NASA multi-instrument (AIA/HMI/EVE) SDO model producing a fused embedding space; an alternative solar SSL backbone and design blueprint. |
| [AI Foundation Model for Heliophysics: design & implementation (2024)](https://arxiv.org/abs/2410.10841) | First design-criteria paper for a heliophysics FM using SDO; argues SSL pretraining mitigates the label bottleneck — direct motivation for pretrain-then-fine-tune on scarce filament labels. |
| [Solar flare forecasting with foundational transformers (2025)](https://arxiv.org/abs/2510.23400) | Benchmarks SigLIP2 (image), VideoMAE (video) and Moirai2 (time-series) encoders on SDO/HMI + GOES — a recent recipe for adapting general vision/temporal FMs to solar tasks. |
| [CORONA-Fields — FMs for solar-wind phenomena (2025)](https://arxiv.org/abs/2511.09843) | Reuses a solar-imagery FM's embeddings (with Fourier neural fields) for downstream classification — evidence and caveats on transferring solar FM features to new tasks. |

## Adjacent solar deep learning (generative, QC, downstream)

Solar-specific techniques that surround a filament pipeline — synthetic augmentation, data cleaning, magnetic context, and the science that pixel-precise masks unlock.

| Work | What / why |
|---|---|
| [DDPM for solar imaging (Ramunno et al. 2024, A&A)](https://www.aanda.org/articles/aa/full_html/2024/06/aa47860-23/aa47860-23.html) | Denoising Diffusion Probabilistic Models on SDO/AIA data — the diffusion direction for solar, relevant to synthetic augmentation of thin structures. |
| [CME detection with synthetically-trained Mask R-CNN (2025)](https://arxiv.org/abs/2511.04589) | Instance segmentation of CME envelopes trained purely on ~10^5 synthetic coronagraph masks (GCS + ray-tracing), IoU 0.77 on real data — a solar-domain template for synthetic-data instance segmentation when labels are scarce. |
| [H-Alpha Anomalyzer (2025)](https://arxiv.org/abs/2509.14472) | Explainable anomaly flagger plus 2,000 annotated GONG observations — data-cleaning/QC for the same GONG H-alpha stream the Cup draws from. |
| [ML reconstruction of Polarity Inversion Lines from filaments (2024)](https://arxiv.org/abs/2405.06293) | Ties filaments to the PILs they lie along via ML — the magnetic-context link useful for auxiliary supervision / feature priors. |
| [Automatic detection of filament oscillations — multi-scale spectral pipeline (2026)](https://arxiv.org/abs/2607.01095) | End-to-end GONG H-alpha pipeline: DL detection/segmentation then Lomb-Scargle spectral analysis — a downstream science use-case showing why pixel-precise masks matter. |

## Method reviews (solar-filament-specific)

| Review | What / why |
|---|---|
| [On Solar Filament Detection Techniques: From Manual to Intelligent (Universe 2024)](https://www.semanticscholar.org/paper/4a2ce991f658abdb99912cdc14d507e6b23267ee) | Survey of filament detection/segmentation methods from manual charts through classical automation to deep learning. |
| [A Comparative Evaluation of Automated Solar Filament Detection (Hao et al. 2015)](https://link.springer.com/article/10.1007/s11207-014-0495-9) | Head-to-head evaluation of classical automated filament-detection codes and their agreement/limits — useful for choosing baselines and understanding metric behavior on H-alpha. |

## Related

Siblings: [Competition, Metric & Domain Data](./Competition_Metric_and_Domain_Data.md) · [Curvilinear / Fine Segmentation & Topology Losses](./Curvilinear_Fine_Segmentation_and_Losses.md) · [Instance Separation Methods](./Instance_Separation_Methods.md) · [Metrics, Tools & Benchmarks](./Metrics_Tools_and_Benchmarks.md) · [Surveys & Reviews](./Surveys_and_Reviews.md) · [Where to Search — Scholarly Platforms](./Where_to_Search_Scholarly_Platforms.md) · Parent: [Solar Filament Segmentation](./README.md) · [Science AI](../)

**Sources:** arXiv · NASA ADS · Solar Physics / Springer · IEEE Xplore · MDPI (Universe, Applied Sciences) · Astronomy & Astrophysics · Acta Astronautica · ApJS / IOP · Zenodo · figshare · ChinaXiv · INSPIRE-HEP · Europe PMC · SSRN · Semantic Scholar.

**Keywords:** solar filament segmentation, H-alpha filament detection, prominence segmentation, Flat U-Net, EdgeAttNet, barb-aware segmentation, filament chirality classification, dextral sinistral, spine barb extraction, AA-UNet, improved U-Net, U2-Net, DETR U-Net tracking, YOLOv5 semi-supervised filament, AAFDCC, Bernasconi, Mask R-CNN filament, CondInst, VNet solar filament, MAGFiLO, GONG H-alpha, 304 A prominence detection, heliophysics foundation model, Surya, SDO-FM, solar physics deep learning, IEEE Big Data Cup 2026.
