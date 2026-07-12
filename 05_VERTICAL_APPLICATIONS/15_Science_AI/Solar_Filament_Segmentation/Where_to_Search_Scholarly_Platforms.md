# Where to Search — Scholarly Platforms

> A map of *where* to search the solar-filament / curvilinear-segmentation literature across the scholarly and preprint platforms in AIForge's 156-source catalog — grouped by platform, with the primary indexes flagged and representative live-verified hits for each.

This page is a discovery guide, not a reading list: it tells you which index to open for which slice of the subject (H-alpha filament segmentation, fine/curvilinear structures, thin-structure instance separation, and topology-aware losses/metrics), and shows the exact records each platform surfaced. Every link below was live-verified. For the deep dives into the papers themselves, follow the sibling pages under **Related**.

---

## 1. Solar-physics & astronomy primary indexes

### NASA ADS — `ui.adsabs.harvard.edu` — PRIMARY INDEX for this subject
The Astrophysics Data System is *the* index for solar physics; search filaments here by bibcode, journal (Solar Physics, ApJ, A&A, ApJS), or full text. Richest single source for the on-subject astronomy literature.

| Record | What it is |
|---|---|
| [EdgeAttNet: Towards Barb-Aware Filament Segmentation](https://ui.adsabs.harvard.edu/abs/2025arXiv250902964S/abstract) | U-Net + learnable edge map/self-attention for barb-aware filament boundaries; evaluated on MAGFiLO. |
| [A universal method for solar filament detection from Hα observations using semi-supervised deep learning](https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.213D) | YOLOv5 + semi-supervised U-Net over the full GONG H-alpha archive (Diercke et al., A&A 686, A213). |
| [Solar filament detection, classification, and tracking with deep learning](https://ui.adsabs.harvard.edu/abs/2024sais.conf...69R/abstract) | DETR detection/classification + U-Net instance segmentation + custom tracking on GONG H-alpha. |
| [Solar Filament Segmentation Based on Improved U-Nets](https://ui.adsabs.harvard.edu/abs/2021SoPh..296..176L) | ASPP-U-Net / CGAN variants on full-disk H-alpha (Huairou + BBSO). |
| [Solar Filament Recognition Based on Deep Learning](https://ui.adsabs.harvard.edu/abs/2019SoPh..294..117Z/abstract) | Early end-to-end U-Net for full-disk H-alpha filament recognition (Zhu et al.). |
| [Toward Filament Segmentation Using Deep Neural Networks](https://ui.adsabs.harvard.edu/abs/2019arXiv191202743A) | Mask R-CNN instance segmentation of filaments (Ahmadzadeh et al., the MAGFiLO/Cup lab). |
| [Statistical Analyses of Solar Prominences and Active Region Features in 304 Å Filtergrams Detected via Deep Learning](https://ui.adsabs.harvard.edu/abs/2024ApJS..272....5Z) | DL detection of prominences/features in SDO/AIA 304 Å over cycle 24. |
| [Using Deep Learning to Detect and Trace Hα Fibrils](https://ui.adsabs.harvard.edu/abs/2021AAS...23811315J/abstract) | Fibril (thin curvilinear chromospheric structure) detection/tracing. |
| [Automatic Solar Filament Segmentation and Characterization](https://ui.adsabs.harvard.edu/abs/2011SoPh..272..101Y/abstract) | Classical full-disk H-alpha segmentation with spine/skeleton characterization. |
| [Solar Event Detection Using Deep-Learning-Based Object Detection Methods](https://ui.adsabs.harvard.edu/abs/2021SoPh..296..160B/abstract) | Object-detection DL for solar events including filaments. |

### INSPIRE HEP — `inspirehep.net`
High-energy-physics index; thin coverage of solar imaging, but it does carry a few adjacent solar-feature and curvilinear-segmentation records.

| Record | What it is |
|---|---|
| [Automation of the filament tracking in the framework of the HELIO project](https://inspirehep.net/literature/1088640) | Automated solar filament detection/tracking pipeline (closest on-subject match). |
| [A quantum mechanics-based algorithm for vessel segmentation in retinal images](https://inspirehep.net/literature/3077519) | Retinal-vessel (fine curvilinear/tubular) segmentation crossover. |
| [Automated Coronal Hole Detection using Local Intensity Thresholding Techniques](https://inspirehep.net/literature/820048) | Solar-feature (coronal-hole) segmentation; nearest related solar-segmentation record. |

---

## 2. Preprint servers & full-text mirrors (where the ML methods live)

### arXiv — `arxiv.org` — PRIMARY INDEX for this subject
The single most productive source for both the solar-filament DL methods and the curvilinear/topology-loss machine-learning literature. Search `astro-ph.SR` for solar, `cs.CV` / `eess.IV` for segmentation methods.

| Record | What it is |
|---|---|
| [Flat U-Net: An Efficient Ultralightweight Model for Solar Filament Segmentation in Full-disk Hα Images](https://arxiv.org/abs/2502.07259) | Lightweight U-Net with channel attention for full-disk H-alpha filaments. |
| [EdgeAttNet: Towards Barb-Aware Filament Segmentation](https://arxiv.org/abs/2509.02964) | Barb/spine-aware filament segmentation benchmarked on MAGFiLO. |
| [Automatic Classification of Magnetic Chirality of Solar Filaments from H-Alpha Observations](https://arxiv.org/abs/2509.18214) | First reproducible chirality baseline on MAGFiLO. |
| [A Universal Method for Solar Filament Detection ... Semi-supervised Deep Learning](https://arxiv.org/abs/2402.15407) | YOLOv5 + U-Net semi-supervised pixel-wise segmentation. |
| [Developing an Automated Detection, Tracking and Analysis Method for Solar Filaments Observed by CHASE](https://arxiv.org/abs/2402.14209) | Filament detection/tracking on CHASE H-alpha data. |
| [Toward Filament Segmentation Using Deep Neural Networks](https://arxiv.org/abs/1912.02743) | Early Mask R-CNN full-disk H-alpha filament instances (BBSO). |
| [Solar Filament Recognition Based on Deep Learning](https://arxiv.org/abs/1909.06580) | CNN/U-Net full-disk H-alpha filament recognition. |
| [Can We Determine the Filament Chirality by the Filament Footpoint Location or the Barb-bearing?](https://arxiv.org/abs/1506.08490) | Spine extraction (Dijkstra) + barb-bearing chirality. |
| [Solar Filament Physiognomy: Inferring Magnetic Quantities from Imaging Observations](https://arxiv.org/abs/2511.15980) | Barb-bearing/chirality-to-helicity inference from imaging. |
| [clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://arxiv.org/abs/2003.07311) | The canonical centerline-Dice topology loss (CVPR 2021). |
| [Centerline Boundary Dice Loss for Vascular Segmentation (cbDice)](https://arxiv.org/abs/2407.01517) | Radius-aware extension of clDice (MICCAI 2024). |
| [Skeleton Recall Loss for ... Thin Tubular Structures](https://arxiv.org/abs/2404.03010) | CPU-efficient skeleton/connectivity loss; reports Betti errors. |
| [Efficient Betti Matching Enables Topology-Aware 3D Segmentation via Persistent Homology](https://arxiv.org/abs/2407.04683) | Fast Betti-matching loss via persistent homology. |
| [Betti Matching: Topologically Faithful Image Segmentation](https://arxiv.org/abs/2211.15272) | Betti-matching topological metric + loss. |
| [Topology-Preserving Deep Image Segmentation](https://arxiv.org/abs/1906.05404) | Foundational persistent-homology Betti-number loss. |
| [Topology-Preserving Image Segmentation with Spatial-Aware Persistent Feature Matching](https://arxiv.org/abs/2412.02076) | Spatial-aware persistent-feature matching topology loss. |
| [Dynamic Snake Convolution ... for Tubular Structure Segmentation](https://arxiv.org/abs/2307.08388) | DSCNet deformable conv for thin/tortuous structures. |
| [Local Intensity Order Transformation for Robust Curvilinear Object Segmentation](https://arxiv.org/abs/2202.12587) | Curvilinear structures: retinal vessels, pavement cracks. |
| [A plug-and-play framework for curvilinear structure segmentation ... reconnecting regularization](https://arxiv.org/abs/2408.12943) | Connectivity/reconnection-focused curvilinear segmentation. |
| [FISBe: A real-world benchmark ... instance segmentation of long-range thin filamentous structures](https://arxiv.org/abs/2404.00130) | Instance segmentation of intertwined thin filaments (CVPR 2024). |
| [GAInS: Gradient Anomaly-aware Biomedical Instance Segmentation](https://arxiv.org/abs/2409.13988) | Separating touching/overlapping/crossing instances. |
| [Instance Segmentation of Dense and Overlapping Objects via Layering](https://arxiv.org/abs/2210.03551) | Layering approach for dense/overlapping instances. |

### ar5iv — `ar5iv.labs.arxiv.org`
LaTeXML HTML renderings of arXiv papers — the fastest way to read (and deep-link into) the same preprints without a PDF.

| Record | What it is |
|---|---|
| [Flat U-Net (HTML)](https://ar5iv.labs.arxiv.org/html/2502.07259) | Solar filament segmentation. |
| [A Universal Method for Solar Filament Detection (HTML)](https://ar5iv.labs.arxiv.org/html/2402.15407) | Solar filament detection/segmentation. |
| [Toward Filament Segmentation Using DNNs (HTML)](https://ar5iv.labs.arxiv.org/html/1912.02743) | Solar filament segmentation. |
| [clDice (HTML)](https://ar5iv.labs.arxiv.org/html/2003.07311) | Topology loss for tubular structures. |
| [cbDice (HTML)](https://ar5iv.labs.arxiv.org/html/2407.01517) | Topology-aware loss/metric. |
| [Skeleton Recall Loss (HTML)](https://ar5iv.labs.arxiv.org/html/2404.03010) | Skeleton/topology loss; Betti errors. |
| [FISBe (HTML)](https://ar5iv.labs.arxiv.org/html/2404.00130) | Instance separation of thin filaments. |
| [Topology-Preserving Deep Image Segmentation (HTML)](https://ar5iv.labs.arxiv.org/html/1906.05404) | Persistent-homology topology loss. |

### alphaXiv — `alphaxiv.org`
Community discussion layer over arXiv; useful for reading annotations/threads on the key methods.

| Record | What it is |
|---|---|
| [clDice](https://www.alphaxiv.org/abs/2003.07311) | Topology loss. |
| [Dynamic Snake Convolution](https://www.alphaxiv.org/abs/2307.08388) | Tubular-structure segmentation. |
| [Spatial-Aware Persistent Feature Matching](https://www.alphaxiv.org/abs/2412.02076) | Topology metric. |
| [Flat U-Net](https://www.alphaxiv.org/abs/2502.07259) | Solar filament segmentation. |
| [EdgeAttNet](https://www.alphaxiv.org/abs/2509.02964) | Barb-aware filament segmentation. |
| [cbDice](https://www.alphaxiv.org/abs/2407.01517) | Topology-aware loss. |
| [FISBe](https://www.alphaxiv.org/abs/2404.00130) | Instance separation of thin filaments. |

### SciRate — `scirate.com`
arXiv "scite/rating" mirror; handy for gauging attention on a paper. Also has a live topic-search endpoint.

| Record | What it is |
|---|---|
| [clDice](https://scirate.com/arxiv/2003.07311) | Topology loss (arXiv mirror). |
| [cbDice](https://scirate.com/arxiv/2407.01517) | Topology-aware loss. |
| [Flat U-Net](https://scirate.com/arxiv/2502.07259) | Solar filament segmentation. |
| [EdgeAttNet](https://scirate.com/arxiv/2509.02964) | Barb-aware filament segmentation. |
| [FISBe](https://scirate.com/arxiv/2404.00130) | Instance separation of thin filaments. |
| [SciRate search: "solar filament segmentation"](https://scirate.com/search?q=solar+filament+segmentation) | Re-run the topic query. |

---

## 3. ML / CS-focused paper indexes

### Papers with Code — `paperswithcode.com` — PRIMARY INDEX for this subject
Best place to find code + leaderboards for segmentation losses/benchmarks and the solar-filament DL baselines.

| Record | What it is |
|---|---|
| [clDice — A Novel Topology-Preserving Loss Function](https://paperswithcode.com/paper/cldice-a-topology-preserving-loss-function) | Centerline-Dice loss with code + vessel/road/neuron results. |
| [Skeleton Recall Loss for Connectivity](https://paperswithcode.com/paper/skeleton-recall-loss-for-connectivity) | Connectivity-preserving loss for vessels/roads/cracks. |
| [FISBe: benchmark for instance segmentation of long-range thin filamentous structures](https://paperswithcode.com/paper/fisbe-a-real-world-benchmark-dataset-for) | Instance-separation benchmark (dataset page also live). |
| [Topologically Faithful Image Segmentation (Betti matching)](https://paperswithcode.com/paper/topologically-faithful-image-segmentation-via) | Betti-matching metric + loss. |
| [Topology-Aware Segmentation Using Discrete Morse Theory (DMT-loss)](https://paperswithcode.com/paper/topology-aware-segmentation-using-discrete) | Global-structure loss with Betti error / ARI / VOI. |
| [Solar Filament Recognition Based on Deep Learning](https://paperswithcode.com/paper/solar-filament-recognition-based-on-deep) | U-Net filament recognition in full-disk H-alpha. |
| [A Universal Method for Solar Filament Detection](https://paperswithcode.com/paper/a-universal-method-for-solar-filament) | Semi-supervised deep filament detection across H-alpha networks. |

### Hugging Face Papers — `huggingface.co/papers`
Paper pages linked to models/datasets/community discussion; light coverage here, focused on the topology-loss and instance-benchmark side.

| Record | What it is |
|---|---|
| [clDice](https://huggingface.co/papers/2003.07311) | Topology-preserving tubular loss. |
| [Skeleton Recall Loss](https://huggingface.co/papers/2404.03010) | Connectivity-preserving thin-structure loss. |
| [FISBe](https://huggingface.co/papers/2404.00130) | Instance separation of thin filamentous structures. |

### OpenReview — `openreview.net`
Peer-review threads for the topology-aware losses and instance-segmentation methods (ICLR / ICML / MIDL / MICCAI-workshop tracks).

| Record | What it is |
|---|---|
| [Topology-Aware Segmentation Using Discrete Morse Theory (DMT-loss, ICLR 2021)](https://openreview.net/forum?id=LGgdb4TS4Z) | Global-structure loss evaluated with Betti error. |
| [Topologically Faithful Image Segmentation (Betti matching, ICML 2023)](https://openreview.net/forum?id=vlaPdKdbGK) | Betti-matching error as metric + differentiable loss. |
| [rNCA: Self-Repairing Segmentation Masks](https://openreview.net/forum?id=RtXHOVnyYP) | Neural-cellular-automata repair of fragmented vessels; clDice/Betti gains. |
| [Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss](https://openreview.net/forum?id=NhLBhx5BVY) | Connectivity-preserving *instance* segmentation of fine structures. |
| [Major Vessel Segmentation on X-ray Coronary Angiography](https://openreview.net/forum?id=H1lTh8unKN) | Curvilinear (coronary) segmentation with connectivity penalty. |

### DBLP — `dblp.org` — PRIMARY INDEX for the CS/ML side
Authoritative bibliographic index for CS venues; use it to pin exact venue/author records and cross-link CoRR (arXiv) versions.

| Record | What it is |
|---|---|
| [clDice (CVPR 2021)](https://dblp.org/rec/conf/cvpr/ShitPSEUZPBM21.html) | Shit, Paetzold, Sekuboyina et al.; CoRR record also listed. |
| [cbDice (MICCAI 2024)](https://dblp.org/rec/conf/miccai/ShiHYGLM24.html) | Boundary/radius-aware extension of clDice. |
| [Skeleton Recall Loss](https://dblp.org/rec/journals/corr/abs-2404-03010.html) | CoRR record (ECCV 2024). |
| [Betti matching (ICML 2023)](https://dblp.org/rec/conf/icml/StuckiPSMB23.html) | Stucki, Paetzold, Shit, Menze, Bauer. |
| [Efficient Betti Matching](https://dblp.org/rec/journals/corr/abs-2407-04683.html) | Efficient Betti-matching loss/metric for 3D. |
| [FISBe (CVPR 2024)](https://dblp.org/rec/conf/cvpr/Mais0MKRRMIK24.html) | Mais et al.; dataset + CoRR records also listed. |
| [Toward Filament Segmentation Using DNNs (IEEE BigData 2019)](https://dblp.org/rec/conf/bigdataconf/AhmadzadehMKAJ19.html) | Same GSU/MLEco group behind MAGFiLO and the 2026 Cup. |
| [EdgeAttNet (ICDM 2025)](https://dblp.org/rec/conf/icdm/SolomonMLA25.html) | Barb-aware filament segmentation on MAGFiLO. |
| [Flat U-Net](https://dblp.org/rec/journals/corr/abs-2502-07259.html) | Ultralightweight full-disk H-alpha filament U-Net. |
| [Deep Computer Vision for Solar Physics Big Data (IEEE BigData 2024)](https://dblp.org/rec/conf/bigdataconf/ShenMLLJDXW24.html) | Survey covering filament identification/tracking; same venue lineage as the Cup. |

---

## 4. Scholarly knowledge graphs & metadata search

### Semantic Scholar — `semanticscholar.org` — PRIMARY INDEX for this subject
Broad AI-augmented graph across astronomy + CS + biomedicine; strong for citation-following and the MAGFiLO data descriptor.

| Record | What it is |
|---|---|
| [A universal method for solar filament detection ... semi-supervised deep learning](https://www.semanticscholar.org/paper/8920cd0d46b2d1b2bc55f4c3478ebab9f5235754) | U-Net pixel-wise filament segmentation across instruments (A&A). |
| [A dataset of manually annotated filaments from H-alpha observations (MAGFiLO)](https://www.semanticscholar.org/paper/bd086f6cebe9e6e0526465e8f3e0b919ea11c0da) | The MAGFiLO dataset descriptor underlying the IEEE Big Data Cup 2026. |
| [Solar filament detection, classification, and tracking with deep learning](https://www.semanticscholar.org/paper/c7fd5efe48e25f45ea53be525d644643cf9f6102) | DETR + U-Net + tracking on GONG H-alpha. |
| [On Solar Filament Detection Techniques: From Manual to Intelligent](https://www.semanticscholar.org/paper/4a2ce991f658abdb99912cdc14d507e6b23267ee) | Review of filament detection/segmentation methods (Universe, 2026). |
| [Solar Filament Detection Based on an Improved Deep Learning Model](https://www.semanticscholar.org/paper/3febeabdc68c95291179ebcbf47f0b7bf96c94d0) | Transformer+CNN filament detector (Applied Sciences). |
| [Automatic detection of solar filament oscillations. I.](https://www.semanticscholar.org/paper/2185228632cd1cd13030a0460b8f6484e047e443) | Multi-scale spectral filament-oscillation pipeline. |
| [Evaluating DL Architectures for Actin Filament Segmentation ... Cryo-ET](https://www.semanticscholar.org/paper/12ccaffae7f69191629d6b9156dfb343181454f7) | Fine/curvilinear (actin) filament segmentation crossover. |

### OpenAlex — `openalex.org` — PRIMARY INDEX for this subject
Open scholarly graph (Works IDs); excellent for programmatic discovery and linking DOIs to arXiv/Zenodo versions.

| Record | What it is |
|---|---|
| [A dataset of manually annotated filaments from H-alpha observations (MAGFiLO)](https://openalex.org/W4402901132) | The MAGFiLO/GONG dataset (Scientific Data). |
| [Flat U-Net](https://openalex.org/W4407555329) | Ultralightweight full-disk H-alpha filament segmentation (ApJ). |
| [EdgeAttNet: Towards Barb-Aware Filament Segmentation](https://openalex.org/W7134894056) | Filament barb segmentation (IEEE ICDMW). |
| [Automatic Classification of Magnetic Chirality of Solar Filaments](https://openalex.org/W7134888825) | Filament chirality classification (IEEE ICDMW). |
| [Solar Filament Segmentation Based on Improved U-Nets](https://openalex.org/W3215738039) | ASPP-U-Net / CGAN (Solar Physics). |
| [Toward Filament Segmentation Using Deep Neural Networks](https://openalex.org/W3006868604) | Early GONG filament segmentation (IEEE Big Data). |
| [Automated High-Precision Recognition of Solar Filaments ... Improved U2-Net](https://openalex.org/W4402989669) | Attention U2-Net filament recognition (Universe). |
| [Can we determine the filament chirality by ... barb-bearing?](https://openalex.org/W1800572925) | Spine/barb/chirality physics. |
| [clDice — a Novel Topology-Preserving Loss Function](https://openalex.org/W3010789210) | Topology loss (CVPR). |
| [The Centerline-Cross Entropy Loss for Vessel-Like Structure Segmentation](https://openalex.org/W4403152501) | Topology-consistent centerline loss (MICCAI). |
| [Efficient Connectivity-Preserving Instance Segmentation ... Supervoxel-Based Loss](https://openalex.org/W4409369717) | Instance separation of thin/connected structures (AAAI). |
| [A plug-and-play framework for curvilinear structure segmentation](https://openalex.org/W4399600776) | Curvilinear/tubular reconnection (Neurocomputing). |

### Crossref Metadata Search — `search.crossref.org`
DOI-level metadata; the place to confirm the canonical journal-of-record version. Start from the search page, then the DOI landings it returns.

| Record | What it is |
|---|---|
| [Crossref search: "solar filament segmentation"](https://search.crossref.org/?q=solar+filament+segmentation&from_ui=yes) | Human front-end over the Crossref index. |
| [Automatic Solar Filament Segmentation and Characterization](https://doi.org/10.1007/s11207-011-9798-2) | Classic filament segmentation/characterization (Solar Physics 272:101). |
| [Solar Filament Segmentation Based on AA-UNet (ICARCE 2022)](https://doi.org/10.1109/icarce55724.2022.10046547) | Axial-attention encoder U-Net for filaments. |
| [Flat U-Net (ApJ)](https://doi.org/10.3847/1538-4357/adadff) | Ultralightweight full-disk H-alpha filament model. |
| [An Automatic Segmentation Algorithm for Solar Filaments ... Context-based Sliding Window](https://doi.org/10.14569/ijacsa.2018.090538) | Classical sliding-window filament segmentation. |
| [Centerline Boundary Dice Loss (cbDice, MICCAI 2024)](https://doi.org/10.1007/978-3-031-72111-3_5) | Topology-aware loss/metric. |
| [YoloCurvSeg: You Only Label One Noisy Skeleton ...](https://doi.org/10.1016/j.media.2023.102937) | Skeleton-supervised curvilinear segmentation. |
| [Joint Skeleton and Boundary Features Networks for Curvilinear Structure Segmentation](https://doi.org/10.1007/978-981-99-4761-4_20) | Skeleton+boundary curvilinear net. |
| [Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers (BCNet, CVPR 2021)](https://doi.org/10.1109/cvpr46437.2021.00401) | Instance separation of overlapping/occluding objects. |
| [DiskMask: ... Instance Segmentation of Elongated or Overlapping Objects (ISBI 2020)](https://doi.org/10.1109/isbi45749.2020.9098435) | Elongated/overlapping thin-object instance separation. |
| [GONG Catalog of Solar Filament Oscillations Near Solar Maximum (ApJS)](https://doi.org/10.3847/1538-4365/aabde7) | GONG H-alpha filament catalog. |

### Dimensions — `app.dimensions.ai`
Full-text scholarly search (publications, grants, patents). Use these live query entry points.

| Record | What it is |
|---|---|
| [Dimensions search: solar filament segmentation](https://app.dimensions.ai/discover/publication?search_text=solar%20filament%20segmentation&search_type=kws&search_field=full_search) | Filament segmentation/detection publications. |
| [Dimensions search: clDice tubular structure topology segmentation](https://app.dimensions.ai/discover/publication?search_text=clDice%20tubular%20structure%20topology%20segmentation&search_type=kws&search_field=full_search) | Topology-aware loss / curvilinear literature. |

### Lens — `lens.org`
Scholarly + patent search. Live query entry points.

| Record | What it is |
|---|---|
| [Lens scholar search: solar filament segmentation](https://www.lens.org/lens/search/scholar/list?q=solar%20filament%20segmentation) | Filament segmentation/detection records. |
| [Lens scholar search: clDice tubular structure topology](https://www.lens.org/lens/search/scholar/list?q=clDice%20tubular%20structure%20topology) | Topology-aware / curvilinear literature. |

### Google Scholar — `scholar.google.com`
Broadest full-text reach; good for tracking citations and grey literature (including the Cup page itself). Start from these queries.

| Record | What it is |
|---|---|
| [A dataset of manually annotated filaments (MAGFiLO)](https://www.nature.com/articles/s41597-024-03876-y) | The MAGFiLO dataset paper (Scientific Data). |
| [Pixel-Precise Segmentation of Solar Filaments — IEEE Big Data Cup 2026](https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/) | The anchor challenge page. |
| [EdgeAttNet](https://arxiv.org/abs/2509.02964) | Barb-aware filament segmentation (MAGFiLO). |
| [Flat U-Net](https://arxiv.org/abs/2502.07259) | Full-disk H-alpha filament segmentation. |
| [A Universal Method for Solar Filament Detection](https://arxiv.org/abs/2402.15407) | Semi-supervised filament detection. |
| [Toward Filament Segmentation Using DNNs](https://arxiv.org/abs/1912.02743) | Early DNN filament segmentation. |
| [clDice](https://arxiv.org/abs/2003.07311) | Centerline-Dice topology loss. |
| [cbDice](https://arxiv.org/abs/2407.01517) | Boundary/diameter-aware clDice extension. |
| [YoloCurvSeg](https://arxiv.org/abs/2212.05566) | Curvilinear/vessel-style thin-structure segmentation. |
| [Joint Segmentation and Path Classification of Curvilinear Structures](https://arxiv.org/abs/1905.03892) | Roads + neurons; joint segmentation + path/instance classification. |
| [Leveraging Persistence Image ... Curvilinear Structure Segmentation](https://arxiv.org/abs/2601.18045) | Persistent-homology for curvilinear segmentation. |
| [Scholar search: solar filament segmentation H-alpha MAGFiLO](https://scholar.google.com/scholar?q=solar+filament+segmentation+H-alpha+MAGFiLO) | Re-run the filament query. |
| [Scholar search: clDice cbDice topology-aware tubular/curvilinear segmentation](https://scholar.google.com/scholar?q=clDice+cbDice+topology-aware+loss+tubular+curvilinear+segmentation) | Re-run the topology-loss query. |

---

## 5. Open-access aggregators & harvesters

### BASE — `base-search.net`
Bielefeld Academic Search Engine harvests OA repositories worldwide. Start from the query pages; some record pages resolve via the DOI landings BASE indexes.

| Record | What it is |
|---|---|
| [BASE search: solar filament segmentation H-alpha](https://www.base-search.net/Search/Results?lookfor=solar+filament+segmentation+H-alpha&type=all) | Native BASE query page. |
| [BASE search: curvilinear structure segmentation clDice topology](https://www.base-search.net/Search/Results?lookfor=curvilinear+structure+segmentation+clDice+topology&type=all) | Topology/curvilinear facet query. |
| [Solar Filament Segmentation Based on Improved U-Nets](https://doi.org/10.1007/s11207-021-01920-3) | Record BASE indexes (Solar Physics). |
| [Flat U-Net (ApJ)](https://doi.org/10.3847/1538-4357/adadff) | Record BASE indexes (also arXiv 2502.07259). |
| [EdgeAttNet (arXiv, uses MAGFiLO)](https://arxiv.org/abs/2509.02964) | arXiv full text harvested by BASE. |
| [clDice (CVPR 2021)](https://doi.org/10.1109/cvpr46437.2021.01629) | Topology loss; arXiv also harvested. |
| [cbDice (MICCAI 2024)](https://doi.org/10.1007/978-3-031-72111-3_5) | Topology-aware metric/loss. |

### CORE — `core.ac.uk`
World's largest OA full-text aggregator. Query pages plus one confirmed landing.

| Record | What it is |
|---|---|
| [Solar Filament Recognition Based on Deep Learning (CORE landing)](https://core.ac.uk/display/334858924) | CORE full-text record (arXiv 1909.06580). |
| [CORE search: solar filament segmentation H-alpha](https://core.ac.uk/search?q=solar+filament+segmentation+H-alpha) | Filament facet query. |
| [CORE search: clDice topology-preserving tubular segmentation](https://core.ac.uk/search?q=clDice+topology-preserving+tubular+structure+segmentation) | Topology-loss facet query. |
| [CORE search: MAGFiLO GONG filament detection](https://core.ac.uk/search?q=MAGFiLO+GONG+filament+detection) | MAGFiLO/GONG dataset facet. |

### OpenAIRE Explore — `explore.openaire.eu`
European Research Graph over OA outputs, datasets, and software. Search page plus the DOI records the API returned.

| Record | What it is |
|---|---|
| [OpenAIRE search: solar filament segmentation](https://explore.openaire.eu/search/find?keyword=solar%20filament%20segmentation) | Search over the OpenAIRE graph. |
| [A Machine Learning Ecosystem for Filament Detection (MLEco figshare record)](https://doi.org/10.6084/m9.figshare.29647943) | The MAGFiLO/GONG pipeline project (Ahmadzadeh). |
| [A Universal Method for Solar Filament Detection (A&A)](https://doi.org/10.1051/0004-6361/202348314) | YOLOv5 + U-Net across ChroTel/GONG/Kanzelhöhe. |
| [Solar Filament Segmentation Based on Improved U-Nets](https://doi.org/10.1007/s11207-021-01920-3) | ASPP-U-Net / CGAN. |
| [clDice (CVPR 2021)](https://doi.org/10.1109/cvpr46437.2021.01629) | Topology-aware loss. |
| [Topology-Preserving Deep Image Segmentation](https://doi.org/10.48550/arxiv.1906.05404) | Betti-number topology loss. |
| [Topologically Faithful Image Segmentation (Betti matching)](https://doi.org/10.48550/arxiv.2211.15272) | Betti-matching metric + loss. |

### Internet Archive Scholar — `scholar.archive.org`
Preservation-focused full-text index (good for grey/preprint copies). Use these live query pages.

| Record | What it is |
|---|---|
| [IA Scholar: solar filament segmentation H-alpha](https://scholar.archive.org/search?q=solar+filament+segmentation+H-alpha) | Indexes the arXiv filament-segmentation preprints. |
| [IA Scholar: MAGFiLO filaments GONG](https://scholar.archive.org/search?q=MAGFiLO+filaments+GONG) | Surfaces the MAGFiLO Scientific Data descriptor. |
| [IA Scholar: clDice topology tubular structure](https://scholar.archive.org/search?q=clDice+topology+tubular+structure+segmentation) | Topology-loss preprints. |
| [IA Scholar: topology-aware Betti skeleton segmentation](https://scholar.archive.org/search?q=topology-aware+Betti+skeleton+segmentation+clDice) | Topology metrics sub-topic. |
| [IA Scholar: instance segmentation thin filamentous curvilinear](https://scholar.archive.org/search?q=instance+segmentation+thin+filamentous+curvilinear) | Instance-separation / FISBe-style work. |

---

## 6. Biomedical / life-science indexes (curvilinear & topology crossover)

The MAGFiLO data descriptor is a *Scientific Data* paper, so it lives in the biomedical indexes too — and these platforms are the richest source for the fine/curvilinear-structure and topology-aware-loss methods (vessels, neurons, axons) that transfer directly to filament spines and barbs.

### PubMed — `pubmed.ncbi.nlm.nih.gov` — PRIMARY INDEX (MAGFiLO descriptor + curvilinear crossover)

| Record | What it is |
|---|---|
| [A dataset of manually annotated filaments from H-alpha observations (MAGFiLO / GONG)](https://pubmed.ncbi.nlm.nih.gov/39333537/) | Direct subject hit: MAGFiLO segmentation/spine/chirality dataset (PMID 39333537). |
| [Extraction of metadata from solar disk H-alpha observations (Sacramento Peak)](https://pubmed.ncbi.nlm.nih.gov/42010261/) | Full-disk solar H-alpha dataset paper. |
| [Topology-aware segmentation for tubular structure in 3D microscopy](https://pubmed.ncbi.nlm.nih.gov/42341831/) | Topology-aware tubular/curvilinear segmentation. |
| [nnLoGoNet: retinal vessel segmentation with Skeleton Recall Loss](https://pubmed.ncbi.nlm.nih.gov/42045476/) | Skeleton-based topology loss for vessels. |
| [Topo-UNet: topology-aware multi-task network for pulmonary vessel segmentation](https://pubmed.ncbi.nlm.nih.gov/42035594/) | Topology-aware vessel segmentation. |
| [Frenet-Serret Frame-Based Decomposition for Part Segmentation of 3-D Curvilinear Structures](https://pubmed.ncbi.nlm.nih.gov/40668707/) | Part/instance segmentation of crossing/branching curvilinear structures (IEEE TMI). |
| [Masked Vascular Structure Segmentation and Completion in Retinal Images](https://pubmed.ncbi.nlm.nih.gov/40031819/) | Connectivity-preserving vessel segmentation/completion. |
| [Topology-aware multiclass segmentation of the Circle of Willis](https://pubmed.ncbi.nlm.nih.gov/41637822/) | Topology-aware vessel-tree segmentation. |
| [Skeleton-guided 3D CNN for tubular structure segmentation](https://pubmed.ncbi.nlm.nih.gov/39264412/) | Skeleton/centerline-guided tubular segmentation. |
| [Expanded tube attention for tubular structure segmentation](https://pubmed.ncbi.nlm.nih.gov/38112883/) | Tubular/curvilinear structure segmentation. |
| [Self-Supervised Learning to Improve Topology-Optimized Axon Segmentation and Centerline Detection](https://pubmed.ncbi.nlm.nih.gov/40503110/) | Topology-optimized axon segmentation + centerline. |
| [TopoRF-Net: Topology-Aware Road Segmentation](https://pubmed.ncbi.nlm.nih.gov/41471424/) | Topology-aware road (curvilinear) segmentation — the roads analog of filaments. |
| [CrackDFANet: pixel-level pavement crack recognition](https://pubmed.ncbi.nlm.nih.gov/33919128/) | Thin crack segmentation with complex topology. |

### Europe PMC — `europepmc.org` — PRIMARY INDEX (mirrors PubMed + preprints)

| Record | What it is |
|---|---|
| [A dataset of manually annotated filaments from H-alpha observations (MAGFiLO v1.0)](https://europepmc.org/article/MED/39333537) | Direct subject hit: the MAGFiLO/GONG dataset (PMID 39333537). |
| [Extraction of metadata from solar disk H-alpha observations (Sacramento Peak)](https://europepmc.org/article/MED/42010261) | Solar full-disk H-alpha data/metadata. |
| [Bidirectional Autoregressive Tracking Method for Solar Filaments](https://europepmc.org/article/PPR/PPR1180242) | Preprint on solar filament tracking. |
| [Topology-aware segmentation for tubular structure in 3D microscopy](https://europepmc.org/article/MED/42341831) | Topology-aware tubular segmentation. |
| [Frenet-Serret Frame-Based Decomposition ... 3-D Curvilinear Structures](https://europepmc.org/article/MED/40668707) | Part/instance segmentation of curvilinear structures. |
| [Topology aware multitask cascaded U-Net for cerebrovascular segmentation](https://europepmc.org/article/MED/39636790) | Topology-aware vessel-tree segmentation. |
| [VISTA-Z: Vascular Imaging and Segmentation for Topology Analysis in Zebrafish](https://europepmc.org/article/PPR/PPR1146592) | Vascular segmentation for topology analysis. |
| [Performance of Frangi-Hessian Pseudo-Labels for Retinal Vessel Segmentation](https://europepmc.org/article/PPR/PPR1269100) | Retinal curvilinear vessel segmentation (preprint). |

### bioRxiv — `biorxiv.org` — PRIMARY preprint server for the biology crossover

| Record | What it is |
|---|---|
| [tUbe net: a generalisable deep learning tool for 3D vessel segmentation](https://www.biorxiv.org/content/10.1101/2023.07.24.550334v1.full) | 3D CNN for tubular vessel segmentation across scales. |
| [Segmentation of 3D blood vessel networks using unsupervised deep learning](https://www.biorxiv.org/content/10.1101/2023.04.30.538453v1.full) | Uses clDice to enforce topology up to homotopy equivalence. |
| [Dual-Field Microvascular Segmentation ... Retinal Vasculature Mapping](https://www.biorxiv.org/content/10.1101/2024.11.27.625635v4.full) | Retinal vessel segmentation with clDice-based connectivity. |
| [Global Neuron Shape Reasoning with Point Affinity Transformers](https://www.biorxiv.org/content/10.1101/2024.11.24.625067v3.full) | Neuron instance segmentation / global shape reasoning. |
| [Semantic segmentation of microscopic neuroanatomical data ... topological priors](https://www.biorxiv.org/content/10.1101/2020.02.18.955237v1.full) | Topology-aware segmentation of neuronal/curvilinear microscopy. |
| [Spontaneous breaking of symmetry in overlapping cell instance segmentation (diffusion models)](https://www.biorxiv.org/content/10.1101/2023.07.07.548066) | Instance separation of overlapping/touching objects. |
| [Automated analysis of whole brain vasculature using machine learning (VesSAP)](https://www.biorxiv.org/content/10.1101/613257v1.full) | Vessel segmentation + tracing pipeline. |
| [Deep learning-based decoding of axonal ultrastructure (EM)](https://www.biorxiv.org/content/10.64898/2026.05.26.727755v1.full) | Watershed + seed propagation to split touching axon fibers. |

### medRxiv — `medrxiv.org`
Clinical-preprint server; vessel/angiography curvilinear segmentation.

| Record | What it is |
|---|---|
| [AutoMorph: Automated Retinal Vascular Morphology Quantification](https://www.medrxiv.org/content/10.1101/2022.05.26.22274795v1.full) | Retinal vessel + artery/vein segmentation pipeline. |
| [AngioNet: CNN for Vessel Segmentation in X-ray Angiography](https://www.medrxiv.org/content/10.1101/2021.01.25.21250488v1.full) | Coronary vessel-tree segmentation emphasizing continuity. |
| [Selective ensemble methods for DL segmentation of major vessels in coronary angiography](https://www.medrxiv.org/content/10.1101/2021.09.13.21263481v1.full) | Coronary artery (curvilinear) segmentation. |
| [Deep generative models for vessel segmentation in CT angiography of the brain](https://www.medrxiv.org/content/10.1101/2025.03.07.25322919) | Cerebral vessel (tubular) segmentation. |
| [DL Pipeline for 3D Morphology of Cerebral Small Perforating Arteries (7T MRI)](https://www.medrxiv.org/content/10.1101/2024.10.03.24314845) | Segmentation + morphology of thin perforating arteries. |

---

## 7. Data, code & institutional repositories

### Zenodo — `zenodo.org`
CERN-hosted repository for datasets, code, and versioned artifacts — where several filament and topology-loss releases live.

| Record | What it is |
|---|---|
| [EdgeAttNet (record)](https://zenodo.org/records/17051538) | Barb-aware filament segmentation code, trained on MAGFiLO (Solomon, Martens, Liu, Angryk). |
| [Flat U-Net (code)](https://zenodo.org/records/14610155) | Software release for the Flat U-Net filament model. |
| [Compound U-Net (HAS / MHAS datasets)](https://zenodo.org/records/17230605) | Solar filament segmentation/detection code + datasets. |
| [FISBe (record)](https://zenodo.org/records/10875063) | Instance-segmentation benchmark for thin filamentous structures + topology metrics. |
| [associated_data: clDice Loss for Road Crack Segmentation](https://zenodo.org/records/15696263) | Dataset/code studying clDice on curvilinear cracks. |
| [CP_SDUNet: road extraction with centerline-preserving Dice loss](https://zenodo.org/records/15686671) | Connectivity/centerline-preserving curvilinear extraction. |
| [TopCoW: topology-aware Circle-of-Willis segmentation data](https://zenodo.org/records/15692630) | Topology-aware vessel-segmentation challenge data. |
| [Best Performing TopCoW Segmentation Dockers](https://zenodo.org/records/15665435) | Winning containers for the CoW topology challenge. |

### Figshare — `figshare.com`

| Record | What it is |
|---|---|
| [A Machine Learning Ecosystem for Filament Detection (MLEcoFi)](https://figshare.com/ndownloader/files/56562266) | The NSF/NSO GONG H-alpha filament project behind MAGFiLO. |

### HAL / Inria HAL — `hal.science`
French national OA archive; strong on centerline/skeleton losses and vessel segmentation from European groups.

| Record | What it is |
|---|---|
| [ccDice: A Topology-Aware Dice Score Based on Connected Components](https://hal.science/hal-04653406v3/document) | Topology-aware metric, alternative to clDice/cbDice (MICCAI TGI3). |
| [Coronary artery centerline tracking with the Morphological Skeleton Loss](https://hal.science/hal-03724882/document) | Skeleton/centerline loss for tubular vessels. |
| [A plug-and-play framework for curvilinear structure segmentation ... reconnecting regularization](https://hal.science/hal-04798632v1/file/Neurocomp__arxiv.pdf) | Connectivity-preserving curvilinear segmentation (Neurocomputing). |
| [One-shot active learning for vessel segmentation](https://hal.science/hal-05144113v1/file/PREPRINT_One_shot_active_learning_for_vessel_segmentation.pdf) | Fine tubular (vessel) segmentation. |
| [Vessel Segmentation with Automatic Centerline Extraction using Tubular & Directional Filters](https://inria.hal.science/inria-00418401/document) | Tubular segmentation with centerline extraction (Inria HAL). |

### SSRN — `ssrn.com`

| Record | What it is |
|---|---|
| [Solar Filament Detection Using Deep Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4849934) | Direct subject: deep-learning filament detection. |
| [A Pyramid Auxiliary Supervised U-Net for Road Crack Detection](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4757909) | Fine/curvilinear (pavement crack) segmentation. |
| [FACS-Net: Frequency-Aware Crack Segmentation ... via Topology Preservation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5384487) | Thin curvilinear crack segmentation with a topology-preserving loss. |

### engrXiv — `engrxiv.org` (via OSF)
Engineering preprints; the pavement-crack analog of thin-structure segmentation.

| Record | What it is |
|---|---|
| [A Novel Adaptive Pixels Segmentation Algorithm for Pavement Crack Detection](https://osf.io/preprints/engrxiv/cyqes_v1/) | Pixel-level crack segmentation (curvilinear). |
| [In-field railhead crack detection using digital image correlation](https://osf.io/preprints/engrxiv/zhg5c_v1/) | Curvilinear crack detection/localization. |
| [Dynamic deformable attention (DDANet) for semantic segmentation](https://osf.io/preprints/engrxiv/wcm24_v1/) | Deformable-attention segmentation architecture. |
| [Deep Learning Architectures for Automated Image Segmentation](https://osf.io/preprints/engrxiv/qt27a_v1/) | General DL image-segmentation overview. |

### Research Square — `researchsquare.com`

| Record | What it is |
|---|---|
| [Subdural vein-based segmentation model for spinal dural AV fistula (CE-MRA)](https://www.researchsquare.com/article/rs-9984374/v1) | Fine/curvilinear vein segmentation. |

### TechRxiv — `techrxiv.org`

| Record | What it is |
|---|---|
| [Retinal blood vessel segmentation using a modified U-Net](https://www.techrxiv.org/articles/preprint/Retinal_blood_vessel_segmentation_using_a_deep_learning_method_based_on_modified_U-NET_model/16653238) | Retinal-vessel (curvilinear) segmentation on DRIVE. |

### ChinaXiv — `chinaxiv.org`

| Record | What it is |
|---|---|
| [An Improved VNet-Based Solar Filament Detection Method](https://chinaxiv.org/abs/202103.00106V1) | Direct subject: improved V-Net for weak/faint filaments (BBSO H-alpha + magnetograms). |

### Authorea — `authorea.com`
Carries solar-ML preprints; the confirmed hit here is *adjacent only* (solar imaging, not filament segmentation).

| Record | What it is |
|---|---|
| [Data Augmentation of Magnetograms for Solar Flare Prediction using GANs](https://www.authorea.com/users/542662/articles/601019-data-augmentation-of-magnetograms-for-solar-flare-prediction-using-generative-adversarial-networks) | Adjacent: solar-ML (GAN magnetogram augmentation for flare prediction), not filament segmentation. |

### ESS Open Archive — `essopenarchive.org`
Earth/space-science preprints; the confirmed hit is *adjacent only* (solar flare forecasting).

| Record | What it is |
|---|---|
| [Multi-Wavelength Transformer-Based 24-Hour Solar Flare Forecasting](https://essopenarchive.org/doi/pdf/10.22541/essoar.177213252.29321624/v1) | Adjacent: solar-image DL (flare forecasting), demonstrates the venue hosts solar-ML. |

### EasyChair Preprints — `easychair.org`
Conference preprint host; the confirmed hit is *adjacent only* (generic medical attention-U-Net).

| Record | What it is |
|---|---|
| [Medical Image Segmentation Using Advanced Attention UNet](https://easychair.org/publications/preprint/5WVQ) | Adjacent: attention-gated residual U-Net (Dice+CE), not filament-specific. |

### Scilit — `scilit.com`
General publication index; only broad image-segmentation records surfaced (adjacent, not on-subject) — listed to document coverage.

| Record | What it is |
|---|---|
| [Point Cloud Segmentation Algorithm Based on Improved Euclidean Clustering](https://www.scilit.com/publications/b7f4e5414cc33fcdb06b357848bccccf) | Adjacent: general segmentation. |
| [Geometric segmentation of 3D scanned surfaces](https://www.scilit.com/publications/e4518d2ada2cb23009a4fdcfe588bad8) | Adjacent: general segmentation. |
| [Segmentation of range images as the search for geometric parametric models](https://www.scilit.com/publications/59c99bc6d63887e4efe572478c3eb199) | Adjacent: general segmentation. |
| [Random Walks for Image Segmentation](https://www.scilit.com/publications/8456235992950be4d7d9b6a66ddcaaf7) | Adjacent: general image segmentation. |

---

## Out of scope

Several servers in AIForge's 156-platform catalog are domain-specific to fields with no bearing on solar/curvilinear pixel segmentation and returned nothing usable for this subject. In particular, **economics, law, and humanities preprint servers** (e.g. RePEc/EconStor/SSRN economics series, legal-studies repositories, and humanities/social-science archives such as SocArXiv/LawArXiv/PhilArchive-style venues) are out of scope: they do not index H-alpha imaging, tubular-structure segmentation, or topology-aware losses. Use the astronomy, ML/CS, and biomedical indexes above instead. The three "adjacent only" hits (Authorea, ESS Open Archive, EasyChair) are retained solely to show those venues *can* carry solar-image or medical-segmentation ML, not because they hold on-subject filament work.

---

## Related

- Parent: [Solar Filament Segmentation (section index)](./README.md)
- Siblings: [Datasets & Data Sources](./Competition_Metric_and_Domain_Data.md) · [Models & Architectures](./Solar_Filament_Segmentation_Methods.md) · [Topology-Aware Losses & Metrics](./Curvilinear_Fine_Segmentation_and_Losses.md) · [Instance Separation of Thin Structures](./Instance_Separation_Methods.md) · [IEEE Big Data Cup 2026 (MAGFiLO)](./Competition_Metric_and_Domain_Data.md)

**Sources:** Compiled from AIForge's 156-platform scholarly/preprint catalog; every link above was live-verified. Primary indexes for this subject: NASA ADS, arXiv, Papers with Code, Semantic Scholar, OpenAlex, DBLP, and PubMed/Europe PMC (for the MAGFiLO *Scientific Data* descriptor); with bioRxiv/medRxiv, Zenodo, Figshare, HAL, INSPIRE HEP, CORE, BASE, OpenAIRE, Crossref, Dimensions, Lens, Google Scholar, Internet Archive Scholar, ar5iv/alphaXiv/SciRate, and additional preprint servers.

**Keywords:** solar filament segmentation, H-alpha filaments, MAGFiLO dataset, GONG H-alpha, IEEE Big Data Cup 2026, scholarly search platforms, where to search, NASA ADS, arXiv, Papers with Code, Semantic Scholar, OpenAlex, DBLP, PubMed, Europe PMC, bioRxiv, curvilinear structure segmentation, tubular structure segmentation, thin structure instance separation, topology-aware loss, clDice, cbDice, Betti matching, skeleton recall loss, centerline Dice, preprint servers, literature discovery.
