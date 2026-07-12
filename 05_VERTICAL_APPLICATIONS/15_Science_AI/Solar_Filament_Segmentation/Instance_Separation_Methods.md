# Instance Separation Methods

> How to turn a filament **semantic mask** into **individual filament instances** — separating structures that touch, overlap or cross, and keeping each filament as **one coherent object** (exactly what the [IEEE Big Data Cup 2026](https://bigdataieee.org/BigData2026/cup/) on the [MAGFiLO](https://www.nature.com/articles/s41597-024-03876-y) dataset scores). Filaments are long, thin, branching and frequently cross one another, so the winning toolkit is borrowed from neurons, vessels, chromosomes, cells and worms. (Solar-specific segmentation, thin-structure architectures + topology losses, and the full metrics panel live on the sibling pages.)

## Orientation-aware separation & terminus pairing (the on-anchor approach)

Where two filaments cross, the pixels are shared; the fix is to reason about **local orientation** to "un-cross" the junction, then regroup the fragments belonging to the same filament.

| Method | What / why |
|---|---|
| [Intersection-to-Overpass — orientation-aware net + terminus pairing (Liu et al., CVPRW 2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8046259/) | **The most on-anchor prior art.** Six orientation-associated branches split crossing filaments at junctions ("turn intersections into overpasses"), then a terminus-pairing algorithm regroups fragments into individual filaments. Designed for exactly the touching/crossing-filament problem. |
| [Using Orientation to Distinguish Overlapping Chromosomes (2022)](https://arxiv.org/abs/2203.13004) | Predicts a Double-Angle orientation representation to disambiguate touching/overlapping elongated objects at their crossings — the same orientation cue as the filament-crossing problem, on chromosomes. |
| [Road connectivity via joint orientation + segmentation (Batra et al., CVPR 2019)](https://github.com/anilbatra2185/road_connectivity) | Adds an orientation-learning auxiliary task + connectivity refinement to reconnect fragmented thin roads. Orientation-aware supervision transfers directly to elongated filaments. |
| [DconnNet — Directional Connectivity-based Segmentation (Yang & Farsiu, CVPR 2023)](https://arxiv.org/abs/2304.00145) | Decouples and tracks directional connectivity to keep elongated structures continuous — directly targets the fragmented masks that break a filament into multiple false instances. |

## Distance/energy & watershed instance separation

Learn an energy or distance surface whose basins are objects, then flood — the classic recipe for splitting touching instances.

| Method | What / why |
|---|---|
| [Deep Watershed Transform for Instance Segmentation (Bai & Urtasun, CVPR 2017)](https://arxiv.org/abs/1611.08303) | Learns an energy map whose basins are object instances; the classic CNN + watershed recipe for splitting touching instances. |
| [The Mutex Watershed — parameter-free graph partitioning (Wolf et al., ECCV 2018)](https://arxiv.org/abs/1904.12654) | Signed-graph partitioning with attractive + repulsive edges, seedless and threshold-free — a graph-based separator well suited to affinity outputs on thin structures. |

## Star-convex & spline polygon regression (StarDist family)

Regress a per-pixel object outline so crowded, touching objects separate without box merging — proposal-free.

| Method | What / why |
|---|---|
| [StarDist — Cell Detection with Star-Convex Polygons (Schmidt et al., MICCAI 2018)](https://arxiv.org/abs/1806.03535) | Radial star-convex polygon regression that separates crowded touching objects without box merging; the foundational proposal-free instance model. |
| [StarDist (code)](https://github.com/stardist/stardist) | Reference implementation. Note: the star-convex assumption fits blobby nuclei, not long curvilinear filaments — best used as a comparative baseline and for compact filament fragments. |
| [StarDist-3D — Star-convex Polyhedra (Weigert et al., WACV 2020)](https://arxiv.org/abs/1908.03636) | 3D extension with anisotropy-aware polyhedra + NMS; relevant for volumetric thin-structure instance separation. |
| [MultiStar — instance segmentation of overlapping objects (Walter et al., ISBI 2021)](https://arxiv.org/abs/2011.13228) | Extends StarDist to genuinely **overlapping** objects by detecting overlap pixels and not suppressing the proposals of truly overlapping instances — targets overlap, not just touching. |
| [SplineDist — segmentation with spline curves (Mandal & Uhlmann, ISBI 2021)](https://www.biorxiv.org/content/10.1101/2020.10.27.357640) | Replaces the star-convex polygon with parametric spline curves, lifting the star-convexity restriction to capture non-convex / curved instance outlines. |
| [Single-shot star-convex polygon instance segmentation for spatially-correlated objects (2025)](https://arxiv.org/abs/2504.12078) | Recent single-shot star-convex approach tailored to spatially-correlated / overlapping biomedical objects; a modern baseline for the StarDist family. |

## Flow- & gradient-field methods (Cellpose family)

Predict a vector field whose sinks define instances; naturally handles variably-shaped and touching objects — and, crucially, elongated ones.

| Method | What / why |
|---|---|
| [Cellpose — a generalist algorithm for cellular segmentation (Stringer et al., Nature Methods 2021)](https://github.com/MouseLand/cellpose) | Predicts spatial flow-gradient fields whose sinks define instances; the reference gradient-tracking method for separating touching, variably-shaped objects. |
| [Omnipose — morphology-independent bacterial segmentation (Cutler et al., Nature Methods 2022)](https://www.nature.com/articles/s41592-022-01639-4) | Distance-field-gradient reformulation of Cellpose built specifically for **elongated, filamentous and branched** cells where Cellpose fails — the closest cell-seg analogue to solar filaments. |
| [Cellpose3 — one-click image restoration for segmentation (2024/2025)](https://www.biorxiv.org/content/10.1101/2024.02.10.579780v1) | Adds learned restoration for noisy / blurry / undersampled inputs before flow-based separation — relevant to low-SNR full-disk H-alpha. |
| [GAInS — Gradient Anomaly-aware Biomedical Instance Segmentation (2024)](https://arxiv.org/abs/2409.13988) | Separates touching / overlapping / crossing instances by exploiting gradient anomalies at instance interfaces. |

## Embedding / associative (proposal-free clustering)

Learn a per-pixel embedding that pulls same-instance pixels together and pushes different instances apart, then cluster — no boxes, no branches.

| Method | What / why |
|---|---|
| [Semantic Instance Segmentation with a Discriminative Loss (De Brabandere et al., 2017)](https://arxiv.org/abs/1708.02551) | Foundational proposal-free embedding: push same-instance pixels together / different apart, then cluster — the basis of most associative / embedding filament separators. |
| [Associative Embedding — joint detection and grouping (Newell et al., NeurIPS 2017)](https://papers.nips.cc/paper/6822-associative-embedding-end-to-end-learning-for-joint-detection-and-grouping) | Introduces the detect-and-group tag-embedding idea underpinning proposal-free instance grouping of pixel-wise predictions. |
| [Recurrent Pixel Embedding for Instance Grouping (Kong & Fowlkes, CVPR 2018)](https://arxiv.org/abs/1712.08273) | Hyperspherical pixel embeddings grouped by recurrent mean-shift as an RNN; end-to-end proposal-free instance separation. |
| [Hierarchical Lovász Embeddings for proposal-free panoptic (Kerola et al., CVPR 2021)](https://arxiv.org/abs/2106.04555) | Per-pixel embeddings encoding both instance and category via a hierarchical Lovász hinge loss, no proposals / branches — a strong proposal-free panoptic template. |
| [Panoptic-DeepLab — center + offset regression (Cheng et al., CVPR 2020)](https://arxiv.org/abs/1911.10194) | Class-agnostic instance branch via center heatmap + pixel-to-center offset regression; a clean bottom-up, proposal-free scheme adaptable to grouping filament pixels. |

## Affinity learning & graph partitioning

Predict inter-pixel affinities, then agglomerate / partition — the workhorse for separating densely packed touching thin processes (connectomics).

| Method | What / why |
|---|---|
| [MALIS structured loss for connectome reconstruction (Funke et al., 2017)](https://arxiv.org/abs/1709.02974) | 3D U-Net predicts inter-voxel affinities trained with a structured (MALIS) loss, then agglomerates — the affinity + agglomeration pipeline for separating touching thin processes. |
| [PatchPerPix — instance segmentation via dense local shape patches (Hirsch et al., ECCV 2020)](https://arxiv.org/abs/2001.07626) | Proposal-free, non-iterative assembly of dense local shape patches; explicitly designed for shapes with **crossovers** and dense clusters (e.g. *C. elegans* worms) — highly analogous to crossing filaments. |
| [Disco — densely-overlapping cell instance segmentation via collaborative coloring (2026)](https://arxiv.org/abs/2602.05420) | Graph-coloring formulation (explicit marking + implicit disambiguation) to separate densely touching / overlapping instances where naive 2-coloring fails. |
| [Global Neuron Shape Reasoning with Point Affinity Transformers (2024)](https://www.biorxiv.org/content/10.1101/2024.11.24.625067v3.full) | Global-shape reasoning via point affinities for neuron instance segmentation — reconstructing long, thin, branching objects that intertwine. |

## Sequential tracing & flood-filling (per-instance extraction)

Extract one instance at a time by iteratively growing / tracing it — a robust paradigm for densely entangled elongated structures.

| Method | What / why |
|---|---|
| [Flood-Filling Networks (Januszewski et al., Nature Methods 2018)](https://www.nature.com/articles/s41592-018-0049-4) | Recurrent CNN that iteratively floods / extends one neurite at a time in EM volumes — a leading paradigm for instance separation of densely packed touching tubular processes. |
| [Flood-Filling Networks (original preprint, 2016)](https://arxiv.org/abs/1611.00421) | The per-instance sequential-extraction formulation: a recurrent 3D CNN floods a single instance to trace an entire thin neurite. |
| [CurvSegFlow — time-conditioned flow matching for curvilinear structures (2026)](https://arxiv.org/abs/2606.21608) | Flow-matching iteratively refines a noisy init into thin structures via a learned velocity field; tested on microtubules with **dense filament crossings under low SNR** — very close to the filament setting. |
| [Curvi-Tracker — refinement by iterative tracking (Heng et al., Pattern Recognition 2026)](https://www.sciencedirect.com/science/article/pii/S0031320325014608) | Deploys tracker agents on foreground pixels with a Double-Confirmation backtracking mechanism to refine coarse curvilinear masks — a tracking route to recover continuity of thin instances. |
| [Splitting touching axons via watershed + seed propagation (EM, 2026)](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v1.full) | Watershed + seed propagation to separate touching / overlapping axon fibers into instances — a concrete recent recipe for thin-structure instance splitting. |

## Curvilinear-geometry-aware decomposition & path classification

Exploit the 1-D curve geometry itself (arc-length frames, path labels) to standardize and separate crossing/branching structures.

| Method | What / why |
|---|---|
| [FreSeg — Frenet–Serret frame-based decomposition of 3D curvilinear structures (2024)](https://arxiv.org/abs/2404.14435) | Decomposes a curvilinear structure into a smooth global curve + cylindrical local primitive via Frenet–Serret frames and arc-length parameterization, standardizing curved geometry for data-efficient part / instance segmentation. Published as IEEE TMI ([Europe PMC](https://europepmc.org/article/MED/40668707)). |
| [Joint Segmentation and Path Classification of Curvilinear Structures (2019)](https://arxiv.org/abs/1905.03892) | Jointly segments and classifies paths of thin structures (roads, neurons) so overlapping / crossing curves are separated into distinct instances. |

## Connectivity-preserving instance losses

Losses that penalize wrong splits/merges and broken/short instances — enforcing "one filament = one connected object" during training.

| Loss | What / why |
|---|---|
| [Efficient Connectivity-Preserving Instance Segmentation with a Supervoxel-Based Loss (Grim et al., AAAI 2025)](https://arxiv.org/abs/2501.01022) | Extends digital-topology "simple points" to supervoxels for a topology-aware **instance-seg** loss that prevents splits / merges of entangled neurites. Directly on instance separation of thin structures. |
| [Preserving instance continuity and length via connectivity-aware loss (2025)](https://arxiv.org/abs/2509.03154) | Negative Centerline Loss + Simplified Topology Loss reduce per-instance discontinuities and preserve elongated length under signal dropout — targets fragmented thin instances. |
| [Skea-Topo — skeleton-based boundary enhancement loss (Liu et al., IJCAI 2024)](https://arxiv.org/abs/2404.18539) | Skeleton-aware weighted loss + boundary-rectified term using **both** foreground and background skeletons to emphasize the topologically critical boundary pixels of **touching** objects — directly useful for separating touching filaments. |

## General instance-segmentation frameworks (baselines to adapt)

The standard detect-then-segment, dynamic-mask, query-based and contour-based frameworks — strong first submissions, especially the ones built for elongated / overlapping objects.

| Method | What / why |
|---|---|
| [Mask R-CNN (He et al., ICCV 2017)](https://arxiv.org/abs/1703.06870) | The original detect-then-segment instance framework — the reference baseline; already applied to filaments on BBSO H-alpha. |
| [DETR — end-to-end detection with transformers (Carion et al., ECCV 2020)](https://arxiv.org/abs/2005.12872) | Set-prediction / bipartite-matching paradigm underlying modern query-based instance and panoptic segmentation (the SWEFil DETR filament pipeline builds on it). |
| [CondInst — conditional convolutions for instance segmentation (Tian et al., ECCV 2020)](https://arxiv.org/abs/2003.05664) | Proposal-free dynamic-mask instance method; a natural fit for touching / overlapping filaments (a filament CondInst variant already handles fragmented filaments). |
| [SOLOv2 — dynamic and fast instance segmentation (Wang et al., NeurIPS 2020)](https://arxiv.org/abs/2003.10152) | Box-free, location-based instance segmentation with dynamic mask kernels + Matrix-NMS — avoids anchor-box merging on thin objects. |
| [Mask2Former — masked-attention mask transformer (Cheng et al., CVPR 2022)](https://arxiv.org/abs/2112.01527) | Query-based transformer unifying instance / panoptic / semantic seg; per-query masks can separate overlapping thin structures. |
| [Deep Snake for Real-Time Instance Segmentation (Peng et al., CVPR 2020)](https://github.com/zju3dv/snake) | Circular-convolution contour deformation for contour-based instance segmentation — a route to separating touching / overlapping elongated instances via evolving contours. |
| [BCNet — occlusion-aware instance segmentation with overlapping bilayers (Ke et al., CVPR 2021)](https://doi.org/10.1109/cvpr46437.2021.00401) | Models occluder and occludee as two overlapping graph-convolution layers — built for **overlapping / occluding** objects, the crossing-filament case. |
| [DiskMask — accurate instance segmentation of elongated or overlapping objects (ISBI 2020)](https://doi.org/10.1109/isbi45749.2020.9098435) | Focuses object features to a compact disk to disambiguate **elongated / overlapping** thin objects — squarely on-theme. |
| [Instance Segmentation of Dense and Overlapping Objects via Layering (2022)](https://arxiv.org/abs/2210.03551) | Represents overlapping instances as separable layers, sidestepping the shared-pixel problem for dense / overlapping objects. |
| [Overlapping-cell instance separation via diffusion models (2023)](https://www.biorxiv.org/content/10.1101/2023.07.07.548066) | Diffusion model that "breaks symmetry" to assign overlapping / touching pixels to distinct instances — a generative route to instance separation. |
| [PointRend — image segmentation as rendering (Kirillov et al., CVPR 2020)](https://arxiv.org/abs/1912.08193) | Adaptive point-based boundary refinement for crisp per-instance edges — addresses the Cup's pixel-precise boundary requirement on thin filament outlines. |

## Benchmarks for thin-structure instance separation

External benchmarks whose objects (neurons, wires, membranes) are long, thin, branching and intertwined — the closest analogues for pretraining and for stress-testing a filament instance separator.

| Benchmark | What / why |
|---|---|
| [FISBe — instance segmentation of long-range thin filamentous structures (CVPR 2024)](https://arxiv.org/abs/2404.00130) | The closest benchmark: instance segmentation of intertwined, long-range thin filamentous (neuron) structures, with dedicated topology-aware evaluation. [Zenodo data](https://zenodo.org/records/10875063). |
| [iShape — irregular shape instance segmentation + ASIS affinity baseline (2021)](https://arxiv.org/abs/2109.15068) | Extreme-aspect-ratio, heavily-overlapping, many-connected-component instances (Wire, Branch, Fence, Antenna) with an affinity-based baseline — the closest general-CV analog to touching / crossing filaments. |
| [SNEMI3D (ISBI 2013)](https://snemi3d.grand-challenge.org/) | 3D EM neurite instance segmentation ranked by Rand error; a core benchmark for separating touching thin structures via affinity / boundary learning. |
| [CREMI — Circuit Reconstruction from EM (MICCAI)](https://cremi.org/) | Neuron instance segmentation scored by Variation of Information + Adapted Rand — the gold standard for instance-separation + topology-aware evaluation of dense thin structures. |
| [ISBI 2012 EM Segmentation Challenge](http://brainiac2.mit.edu/isbi_challenge/) | Thin neuronal membrane boundaries; introduced topology-based metrics and was the birthplace of U-Net — the prototype touching-instance separation task. |

## Scoring instance separation

The split / merge and matched-instance metrics that specifically grade how well touching filaments are separated. (The full metrics panel — IoU, boundary IoU, Betti errors, HD95, pitfalls — is on the [Metrics, Tools & Benchmarks](./Metrics_Tools_and_Benchmarks.md) sibling page.)

| Metric | What / why |
|---|---|
| [Aggregated Jaccard Index (AJI) / MoNuSeg (Kumar et al., IEEE TMI 2017)](https://doi.org/10.1109/TMI.2017.2677499) | The standard instance-separation metric that jointly penalizes under- and over-segmentation of touching objects. |
| [Panoptic Quality (PQ = SQ × RQ) (Kirillov et al., CVPR 2019)](https://arxiv.org/abs/1801.00868) | The cleanest single number for "separate every touching / overlapping filament **and** segment it well": segmentation quality × recognition F1. |
| [Adjusted Rand Index (Hubert & Arabie, 1985)](https://link.springer.com/article/10.1007/BF01908075) | Chance-corrected partition agreement treating segmentation as pixel-pair clustering — penalizes wrong splits / merges of touching filaments. |
| [Variation of Information (Meilă, 2003)](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_14) | A true metric on partitions decomposing into false-split + false-merge information — ideal for diagnosing over- vs under-segmentation of overlapping filaments. |
| [CREMI metrics — VOI + Adapted Rand + CREMI score](https://cremi.org/metrics/) | The reference recipe for scoring instance segmentation of thin branching structures — directly transferable to touching / overlapping filaments. |
| [SNEMI3D — Adapted Rand Error definition](http://brainiac2.mit.edu/SNEMI3D/evaluation) | Origin of the Adapted Rand error used everywhere for neurite (thin-tubular) instance segmentation. |
| [scikit-image `skimage.metrics` — adapted_rand_error, variation_of_information](https://scikit-image.org/docs/stable/api/skimage.metrics.html) | Reference open-source implementations of Adapted Rand error and Variation of Information for instance / label maps — plug-and-play split / merge scoring. |

## Instance-labeling & skeleton post-processing tools

The CPU-side primitives that turn a separated mask into labeled instances and per-instance curvilinear descriptors.

| Tool | What / why |
|---|---|
| [connected-components-3d / cc3d](https://github.com/seung-lab/connected-components-3d) | Fast multi-label 2D/3D connected-components labeling (4/8/6/18/26-connectivity) with dust removal and largest-K — efficient instance labeling of separated filament components. |
| [scikit-image](https://github.com/scikit-image/scikit-image) | `segmentation.watershed` for marker-controlled splitting of touching filaments, `morphology.skeletonize` / `medial_axis` for centerlines, `measure.label` for instances — the workhorse for post-processing filament masks. |
| [skan](https://github.com/jni/skan) | Skeleton-to-graph analysis: per-branch length / tortuosity / junction stats from a skeletonized mask — turns filament skeletons into quantitative per-instance descriptors. |
| [Kimimaro](https://github.com/seung-lab/kimimaro) | TEASAR-based centerline / medial-axis skeletonization of labeled volumes, robust to anisotropy and complex morphology — clean graph skeletons for filament centerline extraction. |

## Related

Siblings: [Competition, Metric & Domain Data](./Competition_Metric_and_Domain_Data.md) · [Solar Filament Segmentation Methods](./Solar_Filament_Segmentation_Methods.md) · [Curvilinear / Fine Segmentation & Topology Losses](./Curvilinear_Fine_Segmentation_and_Losses.md) · [Metrics, Tools & Benchmarks](./Metrics_Tools_and_Benchmarks.md) · [Surveys & Reviews](./Surveys_and_Reviews.md) · [Where to Search — Scholarly Platforms](./Where_to_Search_Scholarly_Platforms.md) · Parent: [Solar Filament Segmentation](./README.md) · [Science AI](../)

**Sources:** arXiv · CVPR / ECCV / ICCV / NeurIPS / ICML / AAAI / IJCAI · MICCAI / ISBI / IEEE TMI · Nature Methods · bioRxiv · Pattern Recognition · Papers with Code · SNEMI3D / CREMI / ISBI EM challenges · Europe PMC · Zenodo · GitHub.

**Keywords:** instance segmentation, instance separation, touching objects, overlapping objects, crossing filaments, orientation-aware segmentation, terminus pairing, deep watershed, mutex watershed, StarDist, star-convex polygons, MultiStar, SplineDist, Cellpose, Omnipose, flow field segmentation, discriminative loss embedding, associative embedding, proposal-free instance segmentation, panoptic segmentation, affinity learning, MALIS, PatchPerPix, flood-filling networks, CurvSegFlow, Frenet-Serret, FreSeg, connectivity-preserving loss, supervoxel loss, Skea-Topo, Mask R-CNN, DETR, CondInst, SOLOv2, Mask2Former, Deep Snake, BCNet, DiskMask, FISBe, iShape, SNEMI3D, CREMI, aggregated Jaccard index, panoptic quality, adjusted Rand index, variation of information, solar filament segmentation, MAGFiLO, thin curvilinear structures.
