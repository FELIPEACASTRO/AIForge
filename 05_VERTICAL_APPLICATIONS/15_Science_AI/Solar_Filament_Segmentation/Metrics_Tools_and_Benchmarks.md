# Metrics, Tools & Benchmarks

> The measurement layer for pixel-precise, thin, curvilinear, instance-separated solar filament masks: overlap/boundary/topology/instance-separation metrics, the software libraries that compute and train against them, and the cross-domain benchmark datasets (retinal vessels, roads, cracks, neurons) that stand in for H-alpha filaments — the evaluation backbone behind the [IEEE Big Data Cup 2026](https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/) on MAGFiLO / GONG H-alpha.

The anchor challenge scores each filament as one coherent instance and grades fine barbs and faint material, using IoU, precision/recall, AP@IoU, hit/miss rates, and an organizers' Multi-scale IoU. Because plain IoU/Dice is notoriously blind to broken centerlines and split/merged instances on thin structures, a fair evaluation panel combines overlap, boundary, topology, and partition metrics — plus the pitfall guides that say when each one lies.

## Overlap & Detection Metrics

| Metric | What / why |
| --- | --- |
| [IoU / Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) | Primary per-pixel overlap (intersection/union); mIoU is the class/scale-averaged variant. Central to the Cup but insensitive to thin-structure boundary/topology errors. |
| [Sorensen–Dice coefficient / Dice-F1](https://doi.org/10.2307/1932409) | Dice = 2\|A∩B\|/(\|A\|+\|B\|), equals pixel F1; monotone with IoU. Standard region-overlap score for filament masks, weak on connectivity — pair with a topology metric. |
| [COCO Average Precision (AP@IoU / mAP)](https://arxiv.org/abs/1405.0312) | Defines AP at an IoU threshold and mAP averaged over IoU 0.50:0.05:0.95 — exactly the AP@IoU / hit-rate / miss-rate family used to score each filament as a detected instance. |
| [Panoptic Quality (PQ = SQ × RQ)](https://arxiv.org/abs/1801.00868) | Decomposes into Segmentation Quality (mean IoU of matched instances) and Recognition Quality (detection F1); the cleanest single number for "separate every touching filament AND segment it well." |

## Boundary & Distance Metrics

| Metric | What / why |
| --- | --- |
| [Boundary IoU](https://arxiv.org/abs/2103.16562) | Evaluates only a thin band around the contour, far more sensitive to spine/barb errors than Mask IoU without over-penalizing small objects. |
| [Normalized Surface Dice (Surface DSC at tolerance)](https://arxiv.org/abs/1809.04430) | Fraction of boundary correctly placed within τ pixels — a tolerant edge metric that grades barb precision without Hausdorff's outlier fragility. |
| [Hausdorff distance for image comparison](https://doi.org/10.1109/34.232073) | Foundational bidirectional point-set distance; basis of HD and the robust 95th-percentile HD95 used for boundary/spine localization error. |
| [Aggregated Jaccard Index (AJI)](https://doi.org/10.1109/TMI.2017.2677499) | Jointly penalizes under- and over-segmentation of touching objects; the standard instance-overlap metric for crowded thin masks (MoNuSeg). |

## Topology & Connectivity Metrics

| Metric | What / why |
| --- | --- |
| [Betti Matching Error (BM0/BM1)](https://arxiv.org/abs/2211.15272) | Spatially-aware fix to the position-blind Betti-number error: matches persistence barcodes so a topological feature must be in the right place; usable as metric and differentiable loss. |
| [Scalar Function Topology Divergence (SFTD)](https://arxiv.org/abs/2407.08364) | Localization-aware topological dissimilarity between sublevel sets; pinpoints where connectivity/hole errors occur and reportedly beats Betti-matching in 2D segmentation. |
| [APLS — Average Path Length Similarity](https://arxiv.org/abs/1908.09715) | Graph connectivity score comparing shortest-path lengths between nodes of predicted vs GT centerline graphs — a template for a filament-spine connectivity metric. |
| [DIADEM metric](https://link.springer.com/article/10.1007/s12021-011-9117-y) | Topology-aware tree/branch comparison matching bifurcations and terminations; transferable to filament spine/barb topology scoring. |

> The centerline-Dice (clDice) score is the most on-point connectivity metric for filament spines; its reference and library implementations are listed under Topology & Persistent-Homology Libraries below.

## Instance-Separation (Split / Merge) Metrics

| Metric | What / why |
| --- | --- |
| [Adjusted Rand Index (ARI)](https://link.springer.com/article/10.1007/BF01908075) | Chance-corrected agreement between two partitions; treats segmentation as pixel-pair clustering, penalizing wrong splits/merges of touching filaments. |
| [Rand index — original (Rand 1971)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1971.10482356) | Foundational pair-counting agreement measure underlying ARI and the connectomics adapted-Rand error. |
| [Variation of Information (VOI)](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_14) | A true metric on partitions splitting into false-split + false-merge information — ideal for diagnosing over/under-segmentation of overlapping filaments (but entangles volume with topology). |
| [CREMI metrics (VOI + Adapted Rand + CREMI score)](https://cremi.org/metrics/) | Reference recipe for scoring instance segmentation of thin branching structures; precise formulas and leaderboard conventions transferable to filaments. |
| [SNEMI3D Adapted Rand Error (ARAND)](http://brainiac2.mit.edu/SNEMI3D/evaluation) | Origin of the Adapted Rand error for neurite (thin-tubular) instance segmentation; the exact metric scikit-image implements as `adapted_rand_error`. |

## Metric Pitfalls & Selection Guides

| Reference | What / why |
| --- | --- |
| [Pitfalls of Topology-Aware Image Segmentation](https://arxiv.org/abs/2412.14619) | The cautions paper for this angle: wrong 4- vs 8-/26-connectivity choices, topological artifacts in ground truth, and metrics lacking expressive power (VOI). Actionable fair-evaluation recommendations. |
| [Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z) | Problem-fingerprint framework for choosing the right metric per task, with explicit instance/semantic-segmentation guidance — use it to justify the filament metric panel. |
| [Understanding metric-related pitfalls in image analysis validation](https://www.nature.com/articles/s41592-023-02150-0) | Taxonomy of concrete failure modes (class imbalance, small/thin structures, empty references) — precisely the traps in faint-material, thin-filament evaluation. |
| [Common Limitations of Image Processing Metrics: A Picture Story](https://arxiv.org/abs/2104.05642) | Heavily-illustrated catalog of how Dice/IoU/HD mislead on thin/small targets — a fast visual sanity check when reporting filament metrics. |
| [Taha & Hanbury — Metrics for 3D medical image segmentation](https://link.springer.com/article/10.1186/s12880-015-0068-x) | Authoritative selection guide for 20+ metrics (HD, HD95, ASSD) with sensitivity/outlier analysis; defines ASSD and warns raw HD is outlier-sensitive. |

## Evaluation Libraries (Metric Computation)

| Library | What / why |
| --- | --- |
| [MONAI metrics module](https://docs.monai.io/en/stable/metrics.html) | GPU-ready DiceMetric, MeanIoU, HausdorffDistanceMetric (HD95 via percentile), SurfaceDiceMetric, PanopticQualityMetric — fastest path to a correct evaluation harness. |
| [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) | 100+ distributed-ready metrics with a segmentation module (Dice, JaccardIndex/IoU, GeneralizedDiceScore, HausdorffDistance, MeanIoU) that accumulate over batches during training. |
| [surface-distance (DeepMind)](https://github.com/google-deepmind/surface-distance) | Reference Normalized Surface Dice, robust/percentile Hausdorff, and average surface distance — boundary metrics that matter where volumetric Dice is insensitive on thin edges. |
| [seg-metrics](https://github.com/Jingnan-Jia/segmentation_metrics) | Fast multi-label Dice/Jaccard/precision/recall plus HD, HD95, and mean/median surface distance (ASSD family), writing to CSV — convenient batch scoring. |
| [scikit-image `skimage.metrics`](https://scikit-image.org/docs/stable/api/skimage.metrics.html) | Open-source `adapted_rand_error` and `variation_of_information` for instance/label maps — plug-and-play split/merge scoring of touching filaments. |
| [EvaluateSegmentation (VISCERAL)](https://github.com/Visceral-Project/EvaluateSegmentation) | C++ CLI computing 20+ overlap and distance metrics (Dice, Jaccard, Hausdorff, ASSD, Rand, kappa) from Taha–Hanbury — a reproducible external cross-check. |

## Segmentation & Training Frameworks

| Library | What / why |
| --- | --- |
| [MONAI](https://github.com/Project-MONAI/MONAI) | PyTorch imaging toolkit; production home of segmentation losses (Dice, DiceCE, Generalized-Dice, Tversky, Focal, HausdorffDT, clDice) and metrics in one stack. |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Self-configuring framework that fingerprints the dataset and auto-sets preprocessing/patch/U-Net config; the default strong baseline and a high-value first Cup submission. |
| [segmentation_models.pytorch (smp)](https://github.com/qubvel-org/segmentation_models.pytorch) | 12 decoders (Unet, Unet++, FPN, DeepLabV3+, UPerNet, Segformer, DPT) × 800+ pretrained encoders with built-in losses/metrics — strong H-alpha baselines in two lines. |
| [Kornia](https://github.com/kornia/kornia) | Differentiable CV with `kornia.morphology` (erosion/dilation/opening/closing) — the primitives for custom soft-skeleton / soft-clDice losses inside the training graph. |
| [Albumentations](https://github.com/albumentations-team/albumentations) | Fast mask-synchronized spatial augmentation (rotate/flip/elastic/grid-distort) that preserves filament-mask alignment. (Original repo archived Jul 2025; successor AlbumentationsX.) |
| [SimpleITK](https://github.com/SimpleITK/SimpleITK) | Morphological watershed, connected-components, distance maps, plus HausdorffDistanceImageFilter / LabelOverlapMeasures for post-processing and metrics. |

## Topology & Persistent-Homology Libraries

| Library | What / why |
| --- | --- |
| [clDice (jocpae/clDice)](https://github.com/jocpae/clDice) | Reference PyTorch soft-clDice — the centerline-Dice topology-preserving loss and connectivity metric; soft-skeleton via iterative min/max-pooling. The single most on-point tool for broken filament centerlines (2D & 3D). |
| [MONAI SoftclDiceLoss / SoftDiceclDiceLoss](https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/cldice.py) | Production, packaged clDice inside MONAI (`monai.losses.cldice`) — drop in without reimplementing soft-skeletonization; blend soft-Dice + clDice with an alpha weight. |
| [Betti-Matching-3D](https://github.com/nstucki/Betti-Matching-3D) | Topologically-faithful loss AND metric via induced matching of persistence features; spatially matches predicted vs GT topology — stronger than Betti-number error for wrong splits/merges. C++/Python. |
| [GUDHI](https://github.com/GUDHI/gudhi-devel) | C++/Python TDA library whose CubicalComplex gives persistent homology directly on 2D image grids — the standard way to compute Betti/persistence topology errors on masks. |
| [ripser.py](https://github.com/scikit-tda/ripser.py) | Lean, fast persistent-homology engine with a lower-star image filtration mode — compute persistence barcodes on filament masks to quantify connectivity/holes. |
| [giotto-tda](https://github.com/giotto-ai/giotto-tda) | scikit-learn-compatible TDA toolbox with CubicalPersistence and Betti-curve/persistence-image transformers — clean pipeline API for topological features/metrics. |
| [persim](https://github.com/scikit-tda/persim) | Persistence-diagram distances (bottleneck, sliced-Wasserstein) and vectorizations (persistence images/landscapes) — quantify topological agreement as an eval add-on. |
| [TopologyLayer](https://github.com/bruel-gabrielsson/TopologyLayer) | Differentiable persistent-homology layer for PyTorch (LevelSetLayer2D + barcode featurizers) — add a topological regularizer to a filament net end-to-end. |
| [torchph](https://github.com/c-hofer/torchph) | PyTorch extensions to compute and differentiate through persistent homology (CUDA-accelerated) — alternative backend for custom topological losses/regularizers. |

## Skeleton, Morphology & Instance Tooling

| Library / Tool | What / why |
| --- | --- |
| [scikit-image](https://github.com/scikit-image/scikit-image) | `segmentation.watershed` (instance splitting), `morphology.skeletonize`/`medial_axis` (centerlines), `remove_small_objects`, distance transforms, `measure.label` — the workhorse for mask post-processing. |
| [skan](https://github.com/jni/skan) | Skeleton analysis: builds a graph from a skeletonized mask and reports per-branch length/tortuosity/junction stats — turns filament skeletons into quantitative per-instance descriptors. |
| [Kimimaro](https://github.com/seung-lab/kimimaro) | TEASAR-based centerline/medial-axis skeletonization of labeled 2D/3D volumes, robust to anisotropy; clean graph skeletons (distance-to-boundary per vertex) beyond `skimage.skeletonize`. |
| [connected-components-3d (cc3d)](https://github.com/seung-lab/connected-components-3d) | Fast multi-label 2D/3D connected-components labeling (4/8/6/18/26-connectivity) with dust removal and largest-K — efficient instance labeling of separated components. |
| [StarDist](https://github.com/stardist/stardist) | Star-convex polygon instance segmentation that cleanly separates touching objects; a comparative instance-separation baseline (star-convexity fits fragments/blobs, not long filaments). |
| [Cellpose](https://github.com/MouseLand/cellpose) | Flow-gradient instance segmentation whose sinks define instances — the reference gradient-tracking method for separating touching, variably-shaped objects. |
| [Cellpose3](https://www.biorxiv.org/content/10.1101/2024.02.10.579780v1) | Adds learned image restoration before flow-based instance separation — relevant to low-SNR full-disk H-alpha. |
| [Omnipose](https://www.nature.com/articles/s41592-022-01639-4) | Distance-field-gradient reformulation built for elongated, filamentous, branched cells where Cellpose fails — the closest cell-seg analogue to solar filaments. |
| [napari](https://github.com/napari/napari) | Fast multi-dimensional viewer with a Labels layer for interactive inspection/correction of filament masks and instance labels — invaluable for QA of MAGFiLO annotations and model outputs. |
| [micro-sam](https://github.com/computational-cell-analytics/micro-sam) | SAM fine-tuned for microscopy with interactive + automatic instance segmentation and a napari plugin — the closest analog for adapting SAM to fine, densely-packed structures. |
| [FilFinder](https://github.com/e-koch/FilFinder) | Astronomy-native filament detection + characterization: adaptive threshold → medial-axis skeleton → graph length, Gaussian width, rolling-Hough orientation/curvature; reads/writes FITS. A ready non-DL baseline. |

## Solar / Astronomy Toolchain

| Library | What / why |
| --- | --- |
| [SunPy](https://github.com/sunpy/sunpy) | Core solar-physics library: read H-alpha/AIA FITS (Map), unified search/download via Fido, coordinate frames and reprojection — the standard way to load MAGFiLO images and align multi-instrument data. |
| [astropy](https://github.com/astropy/astropy) | Underpins SunPy: `astropy.io.fits` for H-alpha FITS data/headers, `astropy.wcs` for world coordinates, units and coordinates — essential for correct ingest of solar imagery. |
| [drms](https://github.com/sunpy/drms) | Query metadata and export/download HMI/AIA (incl. 304A) from Stanford JSOC — pull magnetograms for chirality/PIL context around each filament. |
| [aiapy](https://github.com/LM-SAL/aiapy) | SunPy-affiliated SDO/AIA calibration, PSF deconvolution, and degradation correction — for normalizing space-based imagery alongside ground-based H-alpha. |

## Cross-Domain Benchmark Datasets

Solar filament labels are scarce, so the field pretrains and validates on thin-curvilinear analogues. These are the standard transfer sources and evaluation templates.

### Retinal Vessels (thin branching curves)

| Dataset | What / why |
| --- | --- |
| [DRIVE](https://drive.grand-challenge.org/) | The 40-image (20/20) retinal vessel pixel-segmentation benchmark; the near-universal thin-branching-curve pretrain/transfer source. |
| [STARE](https://cecas.clemson.edu/~ahoover/stare/) | 20 hand-labeled vessel images (two experts); classic DRIVE companion for cross-dataset generalization. |
| [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) | 28 child retinal images with central vessel reflex and uneven illumination — a harder thin-vessel transfer benchmark. |
| [HRF (High-Resolution Fundus)](https://www5.cs.fau.de/research/data/fundus-images/) | 45 high-res (3504×2336) images with binary vessel gold standard — high-resolution thin-vessel pretraining closest to full-disk H-alpha scale. |
| [FIVES](https://www.nature.com/articles/s41597-022-01564-3) | Largest pixelwise retinal set: 800 high-res multi-disease images with crowdsourced expert masks — a strong modern vessel pretrain source. |
| [RITE](https://eye.medicine.uiowa.edu/rite-dataset) | Built on DRIVE with artery/vein + vessel-tree labels (crossings labeled) — for instance/connectivity-aware training on touching thin structures. |
| [ROSE](https://arxiv.org/abs/2007.05201) | 229 OCT-angiography images with centerline/pixel labels; extreme thin-capillary complexity — good for centerline/topology (clDice-style) losses. |
| [IOSTAR](https://www.retinacheck.org/download-iostar-retinal-vessel-segmentation-dataset) | 30 SLO retinal images (1024×1024) with vessel + optic-disc + A/V labels — an additional cross-modality vessel transfer source. |

### Roads (long thin curvilinear networks)

| Dataset | What / why |
| --- | --- |
| [Massachusetts Roads](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset) | ~1171 aerial tiles (~1 m/px); the first large road benchmark — long thin objects on cluttered background, a core road-to-filament transfer source. |
| [DeepGlobe 2018 Road Extraction](https://arxiv.org/abs/1805.06561) | 6226 annotated 1024×1024 satellite tiles (50 cm) with pixel road masks — a large curvilinear-segmentation pretrain and evaluation template. |
| [SpaceNet 3 Roads + APLS](https://spacenet.ai/spacenet-roads-dataset/) | >8000 km labeled roads; introduced the APLS graph/connectivity metric — directly relevant if filaments are graded on spine/topology rather than pixels. |
| [RoadTracer](https://openaccess.thecvf.com/content_cvpr_2018/html/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.html) | Aerial road-graph dataset (15 cities) emphasizing junction/connectivity correctness — a model of centerline/spine tracing for branching thin structures. |

### Cracks (fine, low-contrast, tortuous)

| Dataset | What / why |
| --- | --- |
| [CrackForest (CFD)](https://github.com/cuilimeng/CrackForest-dataset) | 118 pixel-labeled urban pavement crack images; a classic fine 1-few-px curvilinear benchmark, a close analog to faint filament threads. |
| [Crack500 + FPHBN](https://github.com/fyangneil/pavement-crack-detection) | Largest pixel-annotated pavement-crack set (~500 imgs, 250/50/200 split); feature-pyramid + hard-sample reweighting directly relevant to thin-structure imbalance. |
| [CrackSeg9k](https://arxiv.org/abs/2208.13054) | Unified 9255-image crack benchmark merging and re-annotating prior sets — a one-stop pretrain corpus for fine curvilinear segmentation with consistent metrics. |
| [OmniCrack30k](https://github.com/ben-z-original/omnicrack30k) | 30k crack samples / ~9B px across 20+ datasets and materials; explicitly shows transfer-learning effectiveness (nnU-Net) — a direct methodological precedent for filament transfer. |

### Neurons / EM (separating touching thin instances)

| Benchmark | What / why |
| --- | --- |
| [ISBI 2012 EM Segmentation Challenge](http://brainiac2.mit.edu/isbi_challenge/) | 30-slice ssTEM stack for thin neuronal membrane boundaries; introduced topology-based metrics (Rand/warping) and was the birthplace of U-Net. |
| [SNEMI3D (ISBI 2013)](https://snemi3d.grand-challenge.org/) | 3D EM neurite instance segmentation ranked by Rand error — the core benchmark for separating touching thin structures via affinity/boundary learning. |
| [CREMI](https://cremi.org/) | Adult-fly-brain EM neuron segmentation scored by Variation of Information, Rand, and Tolerant Edit Distance — the gold standard for instance-separation + topology-aware evaluation. |
| [FISBe](https://arxiv.org/abs/2404.00130) | Real-world benchmark for instance segmentation of long-range thin filamentous (neuron) structures with dedicated topology-aware metrics — the closest general-CV analog to touching/crossing filaments. |

### Topology-Accuracy & Irregular-Shape Benchmarks

| Benchmark | What / why |
| --- | --- |
| [TopoMortar](https://arxiv.org/abs/2503.03365) | Purpose-built thin-structure (brick-wall) benchmark with accurate/pseudo/noisy labels; its controlled comparison of 8 topology losses (clDice wins) helps pick a filament loss. |
| [iShape](https://arxiv.org/abs/2109.15068) | Benchmark of extreme-aspect-ratio, heavily-overlapping, many-component instances (Wire, Branch, Fence, Antenna) with an affinity baseline — the closest general-CV analog to touching/crossing filaments. |

### Solar ML Benchmarks (evaluation-protocol templates)

| Benchmark | What / why |
| --- | --- |
| [SuryaBench](https://arxiv.org/abs/2508.14107) | Companion benchmark to the Surya heliophysics foundation model defining standardized downstream tasks and splits — a template for evaluating FM transfer to solar segmentation. |
| [SWAN-SF](https://ui.adsabs.harvard.edu/abs/2020harv.data..102A/abstract) | Multivariate-time-series solar-flare benchmark (SDO/HMI SHARP, Cycle 24) on Harvard Dataverse — a reference example of a rigorous, reproducible solar-ML challenge dataset and protocol. |

## Related

[Datasets & H-alpha Imagery](./Competition_Metric_and_Domain_Data.md) · [Methods & Architectures](./Solar_Filament_Segmentation_Methods.md) · [Topology & Connectivity Losses](./Curvilinear_Fine_Segmentation_and_Losses.md) · [Instance Separation](./Instance_Separation_Methods.md) · [Cross-Domain Analogues](./Curvilinear_Fine_Segmentation_and_Losses.md) · [Foundation Models & Transfer](./Solar_Filament_Segmentation_Methods.md) · Parent: [Solar Filament Segmentation](./README.md)

**Sources:** arXiv; GitHub; Nature / Springer / IEEE / IOP / Elsevier journals; MONAI, scikit-image & SunPy documentation; Grand Challenge (DRIVE, SNEMI3D); CREMI; MIT brainiac2 (ISBI/SNEMI3D); SpaceNet; Kaggle; NASA ADS; Harvard Dataverse — all links live-verified in the AIForge master item index.

**Keywords:** solar filament segmentation metrics, thin structure evaluation, IoU mIoU Dice F1, AP@IoU mAP, panoptic quality, boundary IoU, surface Dice HD95 Hausdorff, clDice centerline Dice, Betti matching error persistent homology, adjusted Rand index variation of information, aggregated Jaccard index, APLS connectivity metric, metric pitfalls Metrics Reloaded, MONAI TorchMetrics surface-distance, GUDHI ripser giotto-tda TopologyLayer, scikit-image skan Kimimaro connected-components, StarDist Cellpose Omnipose napari, SunPy astropy drms aiapy FilFinder, DRIVE STARE CHASE_DB1 HRF FIVES retinal vessel benchmark, DeepGlobe SpaceNet Massachusetts roads, CrackForest Crack500 CrackSeg9k OmniCrack30k, SNEMI3D CREMI FISBe neuron instance segmentation, TopoMortar iShape topology benchmark, SuryaBench SWAN-SF, MAGFiLO GONG H-alpha, IEEE Big Data Cup 2026.
