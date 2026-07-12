# Surveys & Reviews

> The review and survey literature that maps the whole solution space for pixel-precise solar filament segmentation — from deep learning in solar physics to curvilinear/vessel segmentation, instance separation, topology-aware losses, and the metric-validation cautions that decide the [IEEE Big Data Cup 2026](https://bigdataieee.org/BigData2026/cup/) (MAGFiLO).

A filament-segmentation solution borrows from many mature fields at once: solar/heliophysics ML, retinal-vessel and tubular-structure segmentation, instance separation of touching thin objects, topology-preserving losses, and rigorous segmentation evaluation. This page collects the **survey / review articles** across every one of those axes so the landscape can be read before diving into individual methods. For the primary methods and datasets, see the sibling pages linked at the bottom.

---

## Solar physics & heliophysics ML reviews

| Review / Survey | What / why |
|---|---|
| [Machine learning in solar physics (Asensio Ramos, Cheung, Chifu, Gafeira)](https://arxiv.org/abs/2306.15308) | Living Reviews in Solar Physics 2023 — the ~100-page authoritative review of ML/DL across solar physics including feature detection and segmentation; the anchor domain review for the filament challenge. |
| [Machine Learning in Heliophysics and Space Weather Forecasting (White Paper)](https://arxiv.org/abs/2006.12224) | Community white paper framing why automated filament detection matters (filaments → CME → geomagnetic storms) and the open data/method gaps in heliophysics ML. |
| [Deep Computer Vision for Solar Physics Big Data: Opportunities and Challenges](https://dblp.org/rec/conf/bigdataconf/ShenMLLJDXW24.html) | IEEE Big Data 2024 survey covering solar feature identification/tracking (filaments among them) — same venue lineage as the 2026 Cup. |
| [On Solar Filament Detection Techniques: From Manual to Intelligent](https://www.semanticscholar.org/paper/4a2ce991f658abdb99912cdc14d507e6b23267ee) | Universe review tracing filament detection/segmentation methods from classical hand-crafted pipelines to modern deep learning — the most on-subject method review. |
| [A Comparative Evaluation of Automated Solar Filament Detection (Hao et al.)](https://link.springer.com/article/10.1007/s11207-014-0495-9) | Solar Physics 2015 head-to-head evaluation of classical automated filament-detection codes and their agreement/limits — useful for choosing baselines and understanding metric behavior on H-alpha. |

## Curvilinear & tubular structure segmentation surveys

| Review / Survey | What / why |
|---|---|
| [A Survey on Curvilinear Object Segmentation in Multiple Applications (Bibiloni, González-Hidalgo, Massanet)](https://www.sciencedirect.com/science/article/abs/pii/S0031320316301704) | Foundational Pattern Recognition 2016 cross-application survey of curvilinear/thin-object segmentation (vessels, roads, airways) — the classic taxonomy directly analogous to filament spines. |
| [Segmentation and Classification Approaches of Clinically Relevant Curvilinear Structures: A Review (Rajitha KV et al.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10042761/) | J. Medical Systems 2023 review of conventional + DL methods for thin curvilinear structures (retinal vessels, corneal nerves, fungal filaments), covering thin-vessel/bifurcation challenges that mirror H-alpha filaments. |
| [Learning-based algorithms for vessel tracking: A review (Jia & Zhuang)](https://www.sciencedirect.com/science/article/abs/pii/S089561112030135X) | Computerized Medical Imaging and Graphics 2021 review focused on tracking/tracing elongated tubular structures — relevant to spine/centerline extraction and following of touching filaments. |

## Retinal vessel segmentation reviews (closest sibling domain)

| Review / Survey | What / why |
|---|---|
| [Overview of Deep Learning Methods for Retinal Vessel Segmentation (Gojić, Kundačina, Mišković, Dragan)](https://arxiv.org/abs/2306.06116) | Concise review of modern DL retinal-vessel methods, design characteristics, and evaluation metrics — the closest sibling domain (thin, low-contrast, connectivity-critical) to filaments. |
| [Retinal Vessel Segmentation Using Deep Learning: A Review (Chen, Zhang et al.)](https://ieeexplore.ieee.org/document/9504555/) | IEEE Access 2021 review of architectures and trends, emphasizing the class-imbalance and thin-vessel difficulty that also dominate filament pixels. |
| [Deep learning for retinal vessel segmentation: a systematic review of techniques and applications](https://link.springer.com/article/10.1007/s11517-025-03324-y) | PRISMA-style systematic review of 79 studies (2020–2024) comparing U-Net/FCN/Transformer/GAN with explicit thin-vessel discussion. |
| [A review of ML methods for retinal blood vessel segmentation and artery/vein classification (Mookiah et al.)](https://www.sciencedirect.com/science/article/abs/pii/S1361841520302693) | Medical Image Analysis 2021 review of 158 papers (2012–2020), strong on the datasets and metrics used for thin-structure evaluation. |

## Instance segmentation surveys

| Review / Survey | What / why |
|---|---|
| [Advances in instance segmentation: Technologies, metrics and applications](https://www.sciencedirect.com/science/article/pii/S0925231225002565) | Neurocomputing 2025 survey structured around exactly the challenge's three needs: instance-separation technologies, evaluation metrics, and applications (including medical). |
| [A review on 2D instance segmentation based on deep neural networks (Gu, Bai, Kong)](https://www.sciencedirect.com/science/article/abs/pii/S0262885622000300) | Image and Vision Computing 2022 review of 2D instance-segmentation paradigms (two-stage, one-stage, query-based) — the core reference for separating touching/overlapping filament instances. |
| [A Survey on Object Instance Segmentation (Sharma, Saifi et al.)](https://link.springer.com/article/10.1007/s42979-022-01407-3) | SN Computer Science 2022 survey of instance-segmentation methods, datasets, and metrics with a distinct taxonomy complementing the Neurocomputing survey. |

## Topology-aware segmentation & loss-function surveys

| Review / Survey | What / why |
|---|---|
| [awesome-topology-driven-image-analysis (curated resource list)](https://github.com/HuXiaoling/awesome-topology-driven-image-analysis) | Maintained by Xiaoling Hu (TopoLoss/DMT/Warping author): a curated, living index of topology-driven segmentation losses, metrics, and papers — the best single bibliography to keep this angle exhaustive. |
| [Topological Data Analysis and Topological Deep Learning Beyond Persistent Homology — A Review (Su, Liu, … Carlsson, Wei)](https://arxiv.org/abs/2507.19504) | 2025 comprehensive review of topological ML beyond persistent homology — background for the connectivity/topology-preserving losses used on curvilinear structures. |
| [High-level Prior-based Loss Functions for Medical Image Segmentation: A Survey (El Jurdi et al.)](https://arxiv.org/abs/2011.08018) | Survey categorizing shape/size/topology/inter-region prior losses — maps directly to enforcing thin-structure topology and connectivity in filament masks. |
| [Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook (Azad, Heidary, Yilmaz et al.)](https://arxiv.org/abs/2312.05391) | Unified taxonomy + benchmarking of 25 segmentation losses (Dice, focal, boundary, region, topology-aware) with a companion GitHub — a menu for tackling filament class imbalance. |
| [Loss odyssey in medical image segmentation (Ma, Chen et al.)](https://www.sciencedirect.com/science/article/abs/pii/S1361841521000815) | Medical Image Analysis 2021 systematic comparison of 20+ segmentation losses (with SegLossOdyssey code) — practical evidence on which losses help imbalanced thin-structure segmentation. |

## General deep-learning segmentation surveys

| Review / Survey | What / why |
|---|---|
| [Image Segmentation Using Deep Learning: A Survey (Minaee, Boykov, Porikli, Plaza, Kehtarnavaz, Terzopoulos)](https://arxiv.org/abs/2001.05566) | Foundational IEEE TPAMI 2022 survey covering semantic + instance architectures (FCN, encoder-decoder, pyramid, attention, GAN) — the canonical entry point for the whole method space. |
| [Medical Image Segmentation Review: The success of U-Net (Azad, Aghdam et al.)](https://arxiv.org/abs/2211.14830) | Comprehensive review + taxonomy of U-Net variants with fair benchmarking and an online paper/code repo — the backbone family most likely used on MAGFiLO filament masks. |
| [Modality-specific U-Net variants for biomedical image segmentation: a survey (Yin, Sun et al.)](https://link.springer.com/article/10.1007/s10462-022-10152-1) | Artificial Intelligence Review 2022 survey organizing U-Net variants by imaging modality and design motif — useful for selecting attention/residual/dense variants for low-contrast disk imagery. |
| [Transformers in medical imaging: A survey (Shamshad, Khan et al.)](https://www.sciencedirect.com/science/article/abs/pii/S1361841523000634) | Medical Image Analysis 2023 survey of transformer architectures for medical segmentation/detection — guidance for ViT/hybrid backbones on filament segmentation. |
| [Transformers in Vision: A Survey (Khan, Naseer, Hayat, Zamir, F. Khan, Shah)](https://dl.acm.org/doi/10.1145/3505244) | ACM Computing Surveys 2022 landmark survey of vision transformers (classification, detection, segmentation) — the reference for long-range context useful in tracing extended filaments. |
| [A review of the Segment Anything Model (SAM) for medical image analysis](https://www.sciencedirect.com/science/article/abs/pii/S0895611124001502) | Computerized Medical Imaging and Graphics 2024 review of SAM/foundation-model adaptation and fine-tuning — relevant for prompt-based or few-label filament pipelines. |
| [Segmentation in large-scale cellular electron microscopy with deep learning: A literature survey (Aswath, Alsahaf, Giepmans, Azzopardi)](https://arxiv.org/abs/2206.07171) | Medical Image Analysis 2023 survey of semantic + instance segmentation of thin, densely-packed neuronal EM structures — a mature analogue for separating touching elongated instances. |

## Metrics & validation reviews

| Review / Survey | What / why |
|---|---|
| [Metrics Reloaded — recommendations for image analysis validation (Maier-Hein, Reinke et al.)](https://www.nature.com/articles/s41592-023-02151-z) | Nature Methods 2024 problem-fingerprint framework for choosing the right metric per task (image/object/pixel level) — use it to justify the metric panel for the filament challenge. |
| [Understanding metric-related pitfalls in image analysis validation (Reinke, Maier-Hein et al.)](https://www.nature.com/articles/s41592-023-02150-0) | Nature Methods 2024 companion: a taxonomy of concrete metric failure modes (class imbalance, small/thin structures, empty references) — precisely the traps of thin-filament, faint-material evaluation. |
| [Common Limitations of Image Processing Metrics: A Picture Story (Reinke et al.)](https://arxiv.org/abs/2104.05642) | A living, heavily-illustrated catalog of how Dice/IoU/HD mislead on small or thin targets, class imbalance, and boundary noise — a fast visual sanity check when choosing filament metrics. |
| [Pitfalls of Topology-Aware Image Segmentation (Berger et al.)](https://arxiv.org/abs/2412.14619) | IPMI 2025 cautions paper: wrong connectivity (4- vs 8-/26-conn) choices, topological artifacts in ground truth, and metrics that lack expressive power (VOI entangles volume with topology) — with actionable fixes. |
| [Metrics for evaluating 3D medical image segmentation (Taha & Hanbury)](https://link.springer.com/article/10.1186/s12880-015-0068-x) | BMC Medical Imaging 2015 authoritative catalog/selection guide for 20+ metrics (Hausdorff, HD95, ASSD) with sensitivity/outlier behavior; ships the EvaluateSegmentation tool. |

## Road & crack segmentation reviews (curvilinear analogs)

| Review / Survey | What / why |
|---|---|
| [Deep learning-based road extraction from remote sensing imagery: Progress, problems, and perspectives](https://www.sciencedirect.com/science/article/pii/S0924271625002758) | ISPRS J. Photogrammetry & Remote Sensing 2025 review of extracting elongated network-like structures under clutter/occlusion — transferable connectivity/topology techniques for filament networks. |
| [Deep Learning for Crack Detection: A Review of Learning Paradigms, Generalizability, and Datasets (Zhang et al.)](https://arxiv.org/abs/2508.10256) | 2025 review (with an Awesome-Crack-Detection repo) of thin-crack segmentation — a strong non-medical analogue for pixel-precise thin curves, class imbalance, and generalization/dataset issues. |

## Foundation-model transfer & robustness reviews

| Review / Survey | What / why |
|---|---|
| [Bridging the Gap: Vision Foundation Models for Optical and Radio Astronomy](https://arxiv.org/abs/2409.11175) | Empirical study of adapting DINOv2/SAM-class VFMs to astronomical data under distribution shift with only hundreds/thousands of labels — best practices for fine-tuning generic FMs on scientific images. |
| [Geospatial FMs: evaluating and enhancing NASA-IBM Prithvi's domain adaptability](https://arxiv.org/abs/2409.00489) | Systematic evaluation of a scientific FM's domain adaptability across tasks/resolutions — methodology directly transferable to assessing FM transfer onto full-disk H-alpha. |
| [Revisiting Foundation Models for Cell Instance Segmentation](https://arxiv.org/abs/2603.17845) | MIDL 2026 (micro-sam authors): systematic evaluation of SAM/SAM2/SAM3 + microscopy variants with an automatic-prompt trick for instance segmentation — the state of SAM-based instance seg. |
| [Are Vision Foundation Models Foundational for Electron Microscopy Image Segmentation?](https://arxiv.org/abs/2602.08505) | Probes DINOv2/DINOv3/OpenCLIP for EM segmentation and finds severe multi-domain degradation with current PEFT — a cautionary read before betting a filament pipeline on a frozen FM. |

## Solar physics & prominence domain reviews (context)

| Review / Survey | What / why |
|---|---|
| [Solar Prominences: Observations — Parenti (Living Reviews in Solar Physics)](https://link.springer.com/article/10.12942/lrsp-2014-1) | Comprehensive observational review: filament fine structure (spine + barbs/feet), counterstreaming, chirality, and eruption phenomenology — the best single primer on the structures being segmented. |
| [Solar Prominences: Theory and Models — Gibson (Living Reviews in Solar Physics)](https://link.springer.com/article/10.1007/s41116-018-0016-2) | Theory counterpart: flux-rope vs sheared-arcade magnetic skeleton, helicity/chirality, and eruption mechanisms — explains what derived magnetic attributes physically mean. |
| [Physics of Solar Prominences II: Magnetic Structure and Dynamics — Mackay et al.](https://arxiv.org/abs/1001.1635) | Authoritative review of filament-channel formation, the chirality/hemispheric rule, barbs/feet, and eruption dynamics — ties observed morphology to magnetic topology. |
| [Prominence Oscillations — Arregui, Oliver & Ballester (Living Reviews in Solar Physics)](https://link.springer.com/article/10.12942/lrsp-2012-2) | Reviews large-amplitude filament oscillations that can precede eruptions and enable seismological field estimates — a time-domain attribute relevant to eruption precursors. |
| [Conditions for the Formation and Maintenance of Filaments (Invited Review) — S. F. Martin](https://link.springer.com/article/10.1023/A:1005026814076) | Canonical definition of dextral/sinistral chirality, right/left-bearing barbs, the filament channel, and the PIL — the barb-bearing→chirality rule that MAGFiLO labels encode. |
| [Solar Filament Physiognomy: Inferring Magnetic Quantities from Imaging Observations — P. F. Chen (review)](https://arxiv.org/abs/2511.15980) | 2025 review of inferring magnetic attributes (helicity sign, field-line curvature, twist, configuration) from H-alpha imaging alone — the theoretical map from pixels/spine/barbs to physical quantities. |

---

## Related

Siblings: [Competition, Metric & Domain Data](./Competition_Metric_and_Domain_Data.md) · [Solar Filament Segmentation Methods](./Solar_Filament_Segmentation_Methods.md) · [Curvilinear / Fine Segmentation & Topology Losses](./Curvilinear_Fine_Segmentation_and_Losses.md) · [Instance Separation Methods](./Instance_Separation_Methods.md) · [Metrics, Tools & Benchmarks](./Metrics_Tools_and_Benchmarks.md) · [Where to Search — Scholarly Platforms](./Where_to_Search_Scholarly_Platforms.md) · Parent: [Solar Filament Segmentation](./README.md) · [Science AI](../)

**Sources:** arXiv · Nature Methods · IEEE TPAMI / IEEE Access · ACM Computing Surveys · Living Reviews in Solar Physics · Space Science Reviews · Solar Physics (SpringerLink) · Medical Image Analysis · Pattern Recognition · Image and Vision Computing · Computerized Medical Imaging and Graphics · ISPRS J. Photogrammetry & Remote Sensing · BMC Medical Imaging · PMC / Semantic Scholar · DBLP · GitHub — every link above is drawn from the project's live-verified item manifest.

**Keywords:** solar filament segmentation survey, machine learning in solar physics review, heliophysics ML review, deep learning solar physics, curvilinear structure segmentation survey, tubular structure segmentation review, retinal vessel segmentation review, vessel tracking review, instance segmentation survey, 2D instance segmentation review, topology-aware segmentation survey, topological deep learning review, segmentation loss function survey, loss odyssey, medical image segmentation survey, U-Net review, vision transformer survey, transformers in medical imaging, Segment Anything Model review, electron microscopy segmentation survey, road extraction review, crack detection review, foundation model domain adaptation review, vision foundation models astronomy, metrics reloaded, image analysis validation pitfalls, topology-aware segmentation pitfalls, Hausdorff HD95 ASSD metrics, solar prominence review, filament chirality review, MAGFiLO, IEEE Big Data Cup 2026.
