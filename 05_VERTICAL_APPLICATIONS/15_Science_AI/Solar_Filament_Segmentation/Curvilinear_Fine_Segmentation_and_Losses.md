# Curvilinear / Fine Segmentation & Topology Losses

> The transferable toolkit for pixel-precise solar filament segmentation: cross-domain architectures for thin, curved structures (retinal vessels, roads, cracks, neurons) plus the topology-preserving and connectivity losses that stop filament spines and faint barbs from fragmenting.

Solar filaments are long, thin, low-contrast, branching curves — the same geometry that decades of work on blood vessels, road networks, pavement cracks and neurites already solve. This page indexes the **curvilinear / tubular segmentation architectures** and the **topology / connectivity / imbalance losses** worth porting to the [IEEE Big Data Cup 2026](https://bigdataieee.org/BigData2026/cup/) (MAGFiLO). For solar-native models see the Methods page; for instance-separation of touching filaments and for metrics/tools/benchmarks see the sibling pages linked at the bottom.

---

## Curvilinear / tubular architectures

### Vessel & tubular backbones

| Method | What / why |
|---|---|
| [Dynamic Snake Convolution (DSCNet)](https://arxiv.org/abs/2307.08388) | Deformable "snake" convolution that adaptively follows slender, tortuous tubes, with a persistent-homology continuity loss — the single most on-point architecture for thin, curved filament spines. |
| [Frangi multiscale vesselness filter](https://link.springer.com/chapter/10.1007/BFb0056195) | MICCAI 1998 classic: Hessian-eigenvalue multiscale tubular enhancement; still a strong unsupervised baseline / preprocessing feature-channel for elongated filaments. |
| [DeepVesselNet](https://arxiv.org/abs/1803.09340) | Vessel segmentation + centerline + bifurcation via cross-hair filters and a class-balanced FP-corrected loss — directly addresses centerline extraction under severe foreground imbalance. |
| [CS-Net](https://github.com/iMED-Lab/CS-Net) | MICCAI 2019 self-attention encoder-decoder purpose-built for curvilinear structures (vessels, nerve fibers) across modalities; repo also hosts CS2-Net. |
| [CS2-Net](https://arxiv.org/abs/2010.07486) | 2D/3D curvilinear segmentation with directional 1x3 / 3x1 kernels + spatial/channel attention, validated on 9 datasets / 6 modalities — a state-of-the-art thin-structure backbone. |
| [SA-UNet](https://arxiv.org/abs/2004.03696) | Lightweight retinal-vessel net with a spatial-attention bottleneck and structured dropout; strong on DRIVE/CHASE_DB1 with few labels — a good small-data filament baseline. |
| [R2U-Net](https://arxiv.org/abs/1802.06955) | Recurrent + residual U-Net; feature accumulation via recurrent conv layers improves thin-structure representation (benchmarked on DRIVE vessels). |
| [IterNet](https://arxiv.org/abs/1912.05763) | Stacked weight-shared mini-UNets that exploit network structural redundancy to recover thin/obscured vessels from the prediction itself; learns from only 10-20 labeled images. |
| [CE-Net](https://arxiv.org/abs/1903.02740) | ResNet encoder + dense atrous conv + residual multi-kernel pooling to preserve fine spatial detail; beats U-Net on vessel detection. |
| [FR-UNet](https://github.com/lseventeen/FR-UNet) | Full-resolution multiresolution-interaction network with dual-threshold iteration to recover weak vessel pixels and connectivity — strong for hairline structures. |
| [URVSM (Universal Vessel Segmentation)](https://arxiv.org/abs/2502.06987) | Modality-agnostic thin-vessel segmentation without per-modality fine-tuning — a recent generalization strategy to borrow for cross-instrument H-alpha. |

### Road & crack delineation (satellite / pavement analogs)

| Method | What / why |
|---|---|
| [D-LinkNet](https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge) | LinkNet + pretrained encoder + dilated center block; DeepGlobe road-extraction winner and a widely reused encoder-decoder for thin-line segmentation on large images. |
| [RoadTracer](https://arxiv.org/abs/1802.03680) | Iterative CNN-guided graph search that traces road-network graphs directly, avoiding noisy segmentation post-processing — a tracing paradigm for filament graphs/skeletons. |
| [Sat2Graph](https://arxiv.org/abs/2007.09547) | Graph-tensor encoding unifying pixel segmentation and graph tracing in one model; handles junctions/overlaps (stacked roads) — relevant to overlapping/crossing filaments. |
| [Road connectivity via joint orientation + segmentation](https://github.com/anilbatra2185/road_connectivity) | CVPR 2019: adds an orientation-learning auxiliary task + connectivity refinement to fix fragmented thin roads — orientation-aware supervision transfers to elongated filaments. |
| [RNGDet](https://arxiv.org/abs/2202.07824) | Transformer + imitation learning that generates a road-network graph vertex-by-vertex and handles complex intersections — graph-detection for connected thin-structure output. |
| [DeepCrack (Liu et al.)](https://github.com/yhlleo/DeepCrack) | SegNet encoder-decoder with multi-scale deep supervision for pixel-wise crack segmentation; cracks are thin, low-contrast and tortuous — a close analog to filaments. |
| [CrackFormer](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_CrackFormer_Transformer_Network_for_Fine-Grained_Crack_Detection_ICCV_2021_paper.html) | Transformer SegNet with 1x1 self-attention + scaling-attention for fine-grained thin-crack detection (CrackFormer-II extends it). |
| [Local Intensity Order Transformation](https://arxiv.org/abs/2202.12587) | Robust curvilinear-object segmentation (retinal vessels, pavement cracks) via a local intensity-order transform resilient to contrast variation. |

### Connectivity- & topology-aware architectures

| Method | What / why |
|---|---|
| [DconnNet](https://arxiv.org/abs/2304.00145) | CVPR 2023: decouples and tracks directional connectivity to keep elongated structures continuous — directly targets the fragmented masks a filament metric penalizes. |
| [GLCP](https://arxiv.org/abs/2507.21328) | MICCAI 2025 oral: multi-head network jointly predicting segmentation, skeleton and local-discontinuity maps to fix both broad and local breaks in tubular masks. |
| [SDF-TopoNet](https://arxiv.org/abs/2503.14523) | Two-stage signed-distance-function pretraining + Betti-matching topological fine-tuning for thin tubular structures — a transferable recipe for connectivity-correct spines. |
| [TopoVST](https://arxiv.org/abs/2603.14909) | Sphere-graph GNN + wave-propagation skeleton tracking with space-occupancy filtering to suppress spurious skeletons — spine/centerline ideas for filament spines and barbs. |
| [UCS](https://arxiv.org/abs/2504.04034) | Adapts SAM to curvilinear segmentation via a Sparse Adapter + FFT high-pass curve prompts with strong cross-domain generalization — a foundation-model route to fine H-alpha curves. |
| [Beyond the Pixel-Wise Loss for Topology-Aware Delineation](https://arxiv.org/abs/1712.02190) | CVPR 2018: VGG-feature topology loss + iterative self-refinement for roads/vessels/neurons — an early, influential recipe for reconnecting gaps in thin-line prediction. |
| [Joint Segmentation and Path Classification of Curvilinear Structures](https://arxiv.org/abs/1905.03892) | Joint pixel segmentation + path/instance classification of thin structures (roads, neurons). |
| [Plug-and-play reconnecting regularization](https://arxiv.org/abs/2408.12943) | A learned reconnecting regularizer that snaps back broken/touching thin structures (vessels, roads) as a plug-in to any curvilinear segmenter. |
| [Leveraging Persistence Image for Curvilinear Structure Segmentation](https://arxiv.org/abs/2601.18045) | Uses persistent-homology (persistence images) to boost robustness and connectivity of curvilinear masks. |
| [rNCA: Self-Repairing Segmentation Masks](https://openreview.net/forum?id=RtXHOVnyYP) | Neural-cellular-automata repair of fragmented retinal vessels; reports Dice/clDice gains and reduced Betti (beta0/beta1) errors. |
| [YoloCurvSeg](https://arxiv.org/abs/2212.05566) | Weakly-supervised curvilinear segmentation via skeleton-based image synthesis; reaches ~97% of full supervision from a single noisy skeleton — useful under scarce filament labels. |
| [Holistically-Nested Edge Detection (HED)](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html) | Deeply-supervised FCN for image-to-image thin edge prediction with multi-scale side outputs — the deep-supervision recipe underpinning DeepCrack and many thin-structure nets. |

### Cross-domain robustness (topology-preserving)

| Method | What / why |
|---|---|
| [TopoTTA](https://arxiv.org/abs/2508.00442) | Plug-and-play test-time adaptation that preserves tubular topology under domain shift (+31.8% clDice reported) — sits at the intersection of domain shift and connectivity. |
| [CST (Curvilinear Structure-preserving Cross-domain Translation)](https://arxiv.org/abs/2510.19679) | Adds a topological structure-consistency module to unpaired image translation so thin curvilinear detail survives domain shift — useful for cross-instrument (GONG/KSO/ChroTel) augmentation. |

### Active contours, snakes & contour-based delineation

| Method | What / why |
|---|---|
| [Snakes: Active Contour Models (Kass, Witkin, Terzopoulos)](https://link.springer.com/article/10.1007/BF00133570) | IJCV 1988 foundation: energy-minimizing deformable spline pulled to object contours — the conceptual ancestor of all deep-snake / active-contour delineation. |
| [Learning Deep Structured Active Contours (DSAC)](https://arxiv.org/abs/1803.06329) | CNN predicts per-instance active-contour-model energy terms, trained end-to-end through contour evolution — bridges snakes and deep nets for shape-regularized outlines. |
| [DARNet (Deep Active Ray Network)](https://arxiv.org/abs/1905.05889) | Active rays in polar coordinates with CNN-predicted energy maps; end-to-end contour evolution that avoids self-intersection. |
| [Curve-GCN](https://github.com/fidler-lab/curve-gcn) | GCN predicting polygon/spline control points simultaneously, supporting curved (spline) boundaries — useful for spline-based annotation/segmentation of curved filaments. |
| [Deep Snake](https://github.com/zju3dv/snake) | CVPR 2020: circular-convolution contour deformation for contour-based instance segmentation — a route to outlining evolving, elongated shapes. |

### Foundational & general segmentation backbones

| Method | What / why |
|---|---|
| [U-Net](https://arxiv.org/abs/1505.04597) | The encoder-decoder + skip-connection architecture underlying essentially every filament and vessel segmenter — the load-bearing classic. |
| [Attention U-Net](https://arxiv.org/abs/1804.03999) | Attention gates in skip connections that suppress background and highlight target structures with little overhead — a building block layered into many curvilinear nets. |
| [UNet++](https://arxiv.org/abs/1912.05074) | Nested dense-skip U-Net; a ubiquitous medical/thin-structure baseline. |
| [HRNet](https://arxiv.org/abs/1908.07919) | Maintains high-resolution streams throughout — spatially precise and well-suited to thin filaments/barbs at the imaging resolution limit. |
| [DeepLabv3+](https://arxiv.org/abs/1802.02611) | Canonical atrous / ASPP encoder-decoder and near-universal semantic-segmentation baseline. |
| [SegFormer](https://arxiv.org/abs/2105.15203) | Efficient hierarchical transformer + lightweight MLP decoder — the standard modern transformer segmentation baseline. |
| [TransUNet](https://arxiv.org/abs/2102.04306) | Most-cited CNN+Transformer U-Net hybrid; a standard global-context baseline for medical/thin-structure segmentation. |
| [Swin-Unet](https://arxiv.org/abs/2105.05537) | Pure Swin-transformer U-shaped network — a widely benchmarked transformer segmentation baseline. |

---

## Topology-preserving & connectivity losses

### Centerline / skeleton connectivity losses

| Loss | What / why |
|---|---|
| [clDice / soft-clDice](https://arxiv.org/abs/2003.07311) | CVPR 2021: centerline-Dice topology-preserving loss on morphological skeleta; provably preserves connectedness of tubular nets — the single most on-point loss to keep filament spines connected. |
| [clDice reference implementation (jocpae/clDice)](https://github.com/jocpae/clDice) | Reference PyTorch soft-clDice with differentiable soft-skeleton via iterative min/max-pooling; 2D & 3D, MIT. |
| [MONAI SoftclDiceLoss / SoftDiceclDiceLoss](https://github.com/Project-MONAI/MONAI/blob/1.5.0/monai/losses/cldice.py) | Production, packaged clDice inside MONAI; SoftDiceclDiceLoss blends soft-Dice + clDice with an alpha weight — the recommended stable combo for tubular masks. |
| [cbDice (Centerline Boundary Dice)](https://arxiv.org/abs/2407.01517) | MICCAI 2024: adds skeleton-radius / boundary awareness to clDice to fix diameter imbalance and translation/deformation sensitivity — useful because filaments vary in width. |
| [cbDice code (PengchengShi1220/cbDice)](https://github.com/PengchengShi1220/cbDice) | Reference implementation of the cbDice loss/metric. |
| [Skeleton Recall Loss](https://arxiv.org/abs/2404.03010) | ECCV 2024: CPU-cheap skeleton-based connectivity loss for thin tubes (vessels/roads/cracks), 2D+3D, binary+multiclass — a cheaper alternative to differentiable skeletonization. |
| [Centerline-Cross Entropy Loss (clCE)](https://papers.miccai.org/miccai-2024/770-Paper1081.html) | MICCAI 2024: fuses cross-entropy robustness with clDice topological focus; better overlap AND connectivity and more robust to noisy annotations than clDice. |
| [Skea-Topo (Skeleton-based Boundary Enhancement Loss)](https://arxiv.org/abs/2404.18539) | IJCAI 2024: skeleton-aware weighted loss + boundary-rectified term using both foreground and background skeletons to emphasize the boundary pixels of touching objects. |
| [TCSN (Bezier topological representation)](https://doi.org/10.1016/j.engappai.2025.110045) | First to use differentiable multiscale Bezier curves as a topological loss/representation for curvilinear segmentation — a novel connectivity loss beyond clDice. |
| [Morphological Skeleton Loss (coronary centerline tracking)](https://hal.science/hal-03724882/document) | Skeleton/centerline-based loss for tubular vessel structures with automatic centerline tracking. |

### Persistent-homology / discrete-Morse / Betti / Euler losses

| Loss | What / why |
|---|---|
| [Topology-Preserving Deep Image Segmentation (TopoLoss)](https://arxiv.org/abs/1906.05404) | NeurIPS 2019 (Hu et al.): differentiable persistent-homology loss enforcing matching Betti numbers with the ground truth — the foundational topology-aware loss against broken/spurious fragments. |
| [Clough et al. Persistent-Homology Topological Loss](https://arxiv.org/abs/1910.01877) | IEEE TMI 2020: constrains a segmentation to a PRIOR Betti number without ground-truth labels — the classic topology-as-unsupervised-prior reference. |
| [DMT-loss (Discrete Morse Theory)](https://arxiv.org/abs/2103.09992) | ICLR 2021: uses discrete Morse theory to extract global 1D skeletons / 2D patches and penalize the whole critical structure — cleaner than noisy PH critical points. |
| [Homotopy Warping loss](https://arxiv.org/abs/2112.07812) | NeurIPS 2022: identifies topologically-critical pixels via digital topology + distance transform and warps the GT to focus training there — improves thin-structure connectivity. |
| [Betti Matching Loss (Induced Matching of Persistence Barcodes)](https://proceedings.mlr.press/v202/stucki23a.html) | ICML 2023: spatially-correct matching of persistence barcodes so topological features are matched in the right place — fixes the ambiguous-matching flaw of plain PH losses. |
| [Efficient Betti Matching (3D)](https://arxiv.org/abs/2407.04683) | MICCAI 2024: highly optimized C++/Python Betti-matching loss + metric with a large speedup over Cubical Ripser, enabling topology-aware training on volumes. |
| [Betti-Matching-3D (nstucki/Betti-Matching-3D)](https://github.com/nstucki/Betti-Matching-3D) | Reference C++/Python implementation of the Betti-matching loss and metric. |
| [Multi-class Betti Matching](https://arxiv.org/abs/2403.11001) | MICCAI 2024: extends Betti-matching to N classes by projecting to N single-class PH problems — relevant if filaments are separated into multiple instance/class channels. |
| [Spatial-Aware Persistent Feature Matching (SPFM)](https://arxiv.org/abs/2412.02076) | Adds original-spatial-domain information to persistent-feature matching to further cut incorrect matches vs Betti-matching on large tubular datasets. |
| [Topograph (Graph-Based Strictly Topology-Preserving Loss)](https://arxiv.org/abs/2411.03228) | ICLR 2025: builds a component graph of prediction vs GT to find critical regions; a strict homotopy-equivalence metric, 3-6x faster than PH losses. |
| [Topology-Aware Focal Loss (TAFL)](https://arxiv.org/abs/2304.12223) | CVPRW 2023: focal loss + Wasserstein/Sinkhorn distance between persistence diagrams — handles class imbalance and topological error jointly, apt when filaments are a tiny minority. |
| [TopoSculpt](https://arxiv.org/abs/2509.03938) | Whole-region modeling + persistent-homology (Betti) curriculum refinement to fix topological breaks in fine tubular shapes. |
| [Fast Euler-Characteristic Topology Loss](https://arxiv.org/abs/2507.23763) | Fast 2D/3D Euler-characteristic topology loss + violation-map correction network — a much cheaper alternative to persistent-homology losses for enforcing connectivity. |
| [Topology-Guaranteed Image Segmentation](https://arxiv.org/abs/2601.11409) | PH + PDE-smoothing that guarantees connectivity/genus AND preserves width (line thickness/length) — rare in also protecting the fine width of curvilinear structures. |
| [ContextLoss (CLoss)](https://arxiv.org/abs/2506.11134) | ICIP 2025: considers each topological error together with its whole context in the critical-pixel mask; repairs up to 44% more missed connections than prior SOTA. |
| [GATS (Geometric Assessment-driven Topological Smoothing)](https://arxiv.org/abs/2311.04116) | Geometry-aware alternative to morphological thinning: estimates tubular radius and uses average pooling to avoid over-thinning during soft-skeletonization, cutting Betti errors ~9%. |

### Instance-level connectivity losses (see also Instance Separation)

| Loss | What / why |
|---|---|
| [Supervoxel-Based Connectivity-Preserving Loss](https://arxiv.org/abs/2501.01022) | AAAI 2025: extends digital-topology "simple points" to supervoxels for an instance-seg loss that prevents splits/merges of entangled neurites — applies to separating touching filaments. |
| [Negative Centerline + Simplified Topology Loss](https://arxiv.org/abs/2509.03154) | Reduces per-instance discontinuities and preserves elongated length under signal dropout — directly targets fragmented thin instances. |

### Region, boundary & class-imbalance losses

| Loss | What / why |
|---|---|
| [Generalised Dice Loss](https://arxiv.org/abs/1707.03237) | Class-rebalanced Dice for rare-foreground segmentation — the standard imbalance-robust Dice variant, and filaments occupy a tiny fraction of the disk. |
| [Tversky Loss](https://arxiv.org/abs/1706.05721) | Tunable precision/recall (FP vs FN) loss for highly imbalanced masks — routinely used to boost thin-structure recall. |
| [Focal Loss](https://arxiv.org/abs/1708.02002) | Cornerstone class-imbalance loss that down-weights easy background — a standard baseline when the target is a sparse minority. |
| [Lovász-Softmax Loss](https://arxiv.org/abs/1705.08790) | Direct convex surrogate for optimizing IoU/Jaccard — the exact overlap family filament challenges score on. |
| [Boundary Loss for Highly Unbalanced Segmentation](https://arxiv.org/abs/1812.07032) | Contour-distance loss complementary to region losses; sharpens the pixel-precise boundaries and barb edges filament scoring demands. |
| [Active Contour (AC) Loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.html) | CVPR 2019: differentiable length + region (contour-energy) loss encoding boundary length and area for sharper boundaries — a plug-in for precise filament edges. |

---

## Related

- Parent: [Solar Filament Segmentation — Research Compendium](./README.md) (in [Science AI](../))
- Sibling pages: [Competition, Metric & Domain Data](./Competition_Metric_and_Domain_Data.md) · [Solar Filament Segmentation Methods](./Solar_Filament_Segmentation_Methods.md) · [Instance Separation Methods](./Instance_Separation_Methods.md) · [Metrics, Tools & Benchmarks](./Metrics_Tools_and_Benchmarks.md) · [Surveys & Reviews](./Surveys_and_Reviews.md) · [Where to Search — Scholarly Platforms](./Where_to_Search_Scholarly_Platforms.md)
- Fundamentals: [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Computer_Vision/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Computer_Vision/)

**Sources:** arXiv, OpenReview, CVF Open Access, MICCAI Proceedings, PMLR, SpringerLink, ScienceDirect (Engineering Applications of AI), HAL, GitHub (jocpae/clDice, MONAI, Betti-Matching-3D, CS-Net, FR-UNet, D-LinkNet, DeepCrack, curve-gcn, road_connectivity, zju3dv/snake) — every link above is drawn from the project's live-verified item manifest.

**Keywords:** curvilinear structure segmentation, tubular structure segmentation, thin structure segmentation, topology-preserving loss, connectivity loss, clDice, soft-clDice, cbDice, centerline Dice, Skeleton Recall Loss, centerline cross-entropy, persistent homology loss, TopoLoss, discrete Morse theory loss, DMT loss, homotopy warping, Betti matching, Betti number error, Euler characteristic loss, Topograph, GATS, Skea-Topo, TopoSculpt, ContextLoss, Dynamic Snake Convolution, DSCNet, CS-Net, CS2-Net, SA-UNet, IterNet, CE-Net, FR-UNet, DeepVesselNet, Frangi vesselness, retinal vessel segmentation, road extraction, D-LinkNet, RoadTracer, Sat2Graph, crack detection, DeepCrack, CrackFormer, active contour, snakes, DSAC, DARNet, Curve-GCN, Deep Snake, HED, U-Net, Attention U-Net, UNet++, HRNet, DeepLabv3+, SegFormer, TransUNet, Swin-Unet, Tversky loss, focal loss, Lovasz-Softmax, boundary loss, generalized Dice, solar filament segmentation, MAGFiLO, spine barb detection, space weather.
