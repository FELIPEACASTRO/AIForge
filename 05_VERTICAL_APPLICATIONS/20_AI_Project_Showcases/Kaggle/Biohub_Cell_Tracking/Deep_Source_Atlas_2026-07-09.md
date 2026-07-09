# Deep Source Atlas - Biohub Cell Tracking

> Devastation-grade source expansion for `Biohub - Cell Tracking During Development`, captured on 2026-07-09. This atlas separates official competition evidence, live Kaggle signals, host-lab tools, methods, datasets, evaluation libraries, model hubs, and risk-monitoring leads.

## Search Method

Sources were collected through four channels:

1. Kaggle CLI/API read-only queries for live competition metadata, leaderboard, notebooks, notebook files, and discussion topics.
2. Official or primary web sources: Kaggle, Royer Lab, CZ Biohub, GitHub repositories, project docs, benchmark sites, journals, and model hubs.
3. arXiv metadata search for cell tracking, microscopy transformers, and cell segmentation foundation models.
4. bioRxiv metadata search for exact terms. The exact bioRxiv queries for `Cellpose-SAM`, `CellSAM`, `Ultrack`, and `Zebrahub` returned zero through the local bioRxiv skill, so those preprints were instead routed through direct web sources when available.

## Highest-Priority Findings

| Priority | Finding | Why it matters |
|---|---|---|
| P0 | The official metric rewards edge correctness much more than division detection. | Most leaderboard gains should come from robust detection and temporal edge linking before division polish. |
| P0 | Public notebooks are converging around U-Net/transformer outputs, ILP-style relinking, gap recovery, short-track filtering, blend preprocessing, and rule-based postprocessing. | The strongest public signals are not a single model, but a graph-cleaning and ensemble pattern. |
| P0 | Notebook outputs expose `run_stats.csv`, `submission.csv`, `kaggle_test_splits_50ep.json`, and GEFF/Zarr graph artifacts. | Reproducibility work should parse outputs and metadata before downloading large competition data. |
| P1 | Kaggle CLI output still has no reliable country field for teams or notebook authors. | Country attribution must remain strict and evidence-based. |
| P1 | Trackastra, Ultrack, tracksdata, GEFF, traccuracy, and py-ctcmetrics form the strongest reproducibility backbone. | These tools cover linking, joint segmentation/tracking, graph data, exchange format, and evaluation. |
| P1 | Cellpose-SAM/CellposeDINO, CellSAM, StarDist, nnU-Net, u-Segment3D, and BioImage.IO are the most relevant segmentation/model sources to monitor. | The competition is 3D+t tracking, but the first failure mode is still detection/segmentation quality. |
| P2 | The public discussion title about possible train/test overlap is a risk-monitoring signal, not a confirmed fact from CLI content. | Treat it as a host/community item to watch, not as an exploit recipe. |

## Official Competition And Host Sources

| Source | Role | URL |
|---|---|---|
| Kaggle competition page | Official task, data, rules, leaderboard, code, and discussion entry point. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development |
| Kaggle data page | Official data framing for detecting, tracking, and linking cells in 3D+t. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data |
| Kaggle code page | Public notebooks and baselines. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/code |
| Kaggle discussions | Host and competitor updates. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion |
| Kaggle leaderboard | Volatile public score frontier. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard |
| Biohub Kaggle organization | Official host organization route. | https://www.kaggle.com/organizations/biohub |
| Royer Lab official competition repo | Baseline code, data model, evaluation, prediction, and visualization. | https://github.com/royerlab/kaggle-cell-tracking-competition |
| Official metric spec | Source of truth for edge Jaccard, adjusted edge Jaccard, division Jaccard, and final score. | https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md |
| Royer Group at Biohub | Host-lab scientific context. | https://biohub.org/royer/ |
| Royer Lab GitHub | Host-lab software ecosystem. | https://github.com/royerlab |

## Live Kaggle Signals

| Signal | Snapshot |
|---|---|
| CLI version | `Kaggle CLI 2.2.3` |
| Competition ref | `biohub-cell-tracking-during-development` |
| Deadline | 2026-09-29 23:59 UTC |
| Reward | 60,000 USD |
| Teams | 951 latest observed; 950 in an earlier 2026-07-08 snapshot |
| User entered | true |
| Top public score observed | 0.910 |
| Score cluster near frontier | Many teams from 0.898 to 0.910 in the first visible leaderboard page |
| Top notebook themes | nearest-neighbor starter, learned graph with gap recovery, LB897 baseline, data-model EDA, V4 U-Net ILP reproduction, rule-based V14, 3D U-Net, blend preprocessing, ensemble |
| Recent notebook themes | blend preprocessing, complete journey, min-track recall, division image support, deep-center blend, relink/slot candidates |
| Discussion themes | official Discord/get-started, welcome thread, resource sharing, possible train/test overlap, node id meaning, scaling LAP baseline, visualization |

## Live Kaggle Notebook Leads

| Ref | Why it matters |
|---|---|
| `inversion/cell-tracking-getting-started-w-nearest-neighbor` | Starter baseline; useful for data-model sanity and nearest-neighbor lower bound. |
| `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` | High-value graph/linking and gap-recovery direction. |
| `yusuketogashi/lb897-baseline` | Strong public baseline around the 0.897 region. |
| `pilkwang/biohub-cell-tracking-data-model-eda-baseline` | Data model and exploratory analysis reference. |
| `yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction` | U-Net plus ILP reproduction route. |
| `seshurajup/lb-0-857-rule-based-v14` | Evidence that rule-based graph logic can be competitive. |
| `xiaoleilian/biohub-cell-tracking-classical-baseline` | Classical baseline for non-deep-learning comparison. |
| `xiaoleilian/biohub-cell-tracking-3d-u-net` | 3D U-Net route for detection/segmentation. |
| `pilkwang/biohub-cell-tracking-blend-preprocessings` | Active preprocessing blend signal, last run 2026-07-08. |
| `amanatar/biohub-cell-tracking-ensemble` | Ensemble direction. |
| `beicicc/biohub-exp043-yusuke-dataset-mintrack-recall` | Short-track and recall-focused candidate. |
| `beicicc/biohub-exp042-vmerckle-div-image-support` | Division image-support candidate. |
| `vmerckle/biohub-cand-gap-image-anchor-reuse-0708002923` | Gap and image-anchor reuse candidate. |
| `vmerckle/biohub-cand-relink-division-slot-0707123554` | Division slot relinking candidate. |
| `llccqq624/biohub-cell-tracking-deepcenter-blend` | Deep-center blend candidate. |

## Host-Lab And Graph Infrastructure

| Source | Role | URL |
|---|---|---|
| Ultrack GitHub | Joint segmentation/tracking under segmentation uncertainty; scales to large 3D+t data. | https://github.com/royerlab/ultrack |
| Ultrack paper | Peer-reviewed method for large-scale 2D, 3D, and multichannel tracking. | https://www.nature.com/articles/s41592-025-02778-0 |
| ultrack-td | C++ rewrite of Ultrack core segmentation with tracksdata compatibility. | https://github.com/royerlab/ultrack-td |
| tracksdata | Common graph data structure for multi-object tracking. | https://github.com/royerlab/tracksdata |
| tracksdata docs | API docs for graph handling. | https://royerlab.github.io/tracksdata/ |
| inTRACKtive GitHub | Browser visualization and sharing of cell tracking data. | https://github.com/royerlab/inTRACKtive |
| inTRACKtive paper | Peer-reviewed browser visualization context. | https://www.nature.com/articles/s41592-025-02777-1 |
| GEFF GitHub | Reference implementation of Graph Exchange File Format. | https://github.com/funkelab/geff |
| GEFF docs | Zarr-based graph exchange format specification. | https://live-image-tracking-tools.github.io/geff/ |
| GEFF JOSS paper | Published software paper for graph exchange in tracking. | https://joss.theoj.org/papers/10.21105/joss.10143 |
| motile | Candidate-graph tracking optimization library from Funke Lab. | https://funkelab.github.io/motile/ |

## Tracking And Linking Methods

| Source | Relevance | URL |
|---|---|---|
| Trackastra GitHub | Transformer-based association model with pretrained models and napari plugin. | https://github.com/weigertlab/trackastra |
| Trackastra arXiv | Transformer-based cell tracking paper. | https://arxiv.org/abs/2405.15700 |
| Trackastra ECCV PDF | Conference version and method details. | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf |
| btrack GitHub | Bayesian multi-object tracking for crowded time-lapse microscopy. | https://github.com/quantumjot/btrack |
| btrack paper | Bayesian lineage tracking publication. | https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.734559/full |
| LapTrack | LAP-style tracking reference route. | https://github.com/yfukai/laptrack |
| u-track | MATLAB LAP and gap-closing tracker. | https://github.com/DanuserLab/u-track |
| u-track3D | 3D-oriented u-track workflow. | https://github.com/DanuserLab/u-track3D |
| TrackMate GitHub | Fiji tracking platform for particle/cell tracking workflows. | https://github.com/trackmate-sc/TrackMate |
| TrackMate ImageJ docs | User-facing Fiji/TrackMate documentation. | https://imagej.net/plugins/trackmate/ |
| Cell-TRACTR GitHub | Transformer-based end-to-end segmentation/tracking route. | https://github.com/owen24819/Cell-TRACTR |
| Cell-TRACTR paper | End-to-end transformer method for microscopy cell segmentation/tracking. | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013071 |

## Segmentation And Foundation Model Sources

| Source | Relevance | URL |
|---|---|---|
| Cellpose GitHub | Generalist cell/nucleus segmentation, including Cellpose-SAM/Cellpose4. | https://github.com/mouseland/cellpose |
| Cellpose docs - models | Current built-in model names and Cellpose-SAM/CellposeDINO details. | https://cellpose.readthedocs.io/en/latest/models.html |
| Cellpose-SAM preprint | Cellpose-SAM foundation model preprint. | https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1 |
| Cellpose Hugging Face Space | Hosted Cellpose-SAM demo route. | https://huggingface.co/spaces/mouseland/cellpose |
| Cellpose-SAM Hugging Face model | Direct model route referenced by Cellpose docs. | https://huggingface.co/mouseland/cellpose-sam |
| Cellpose cpsam ONNX | ONNX export route for Cellpose-SAM. | https://huggingface.co/keejkrej/cellpose-cpsam-onnx |
| CellSAM Nature Methods | Universal cell segmentation model paper. | https://www.nature.com/articles/s41592-025-02879-w |
| CellSAM GitHub | CellSAM inference repository. | https://github.com/vanvalenlab/cellsam |
| CellSAM docs | CellSAM documentation. | https://vanvalenlab.github.io/cellSAM/ |
| CellSAM deployed app | Public CellSAM application route. | https://cellsam.deepcell.org/ |
| StarDist GitHub | 2D/3D star-convex cell/nuclei detection. | https://github.com/stardist/stardist |
| StarDist FAQ | 2D/3D microscopy details and practical constraints. | https://stardist.net/faq/ |
| StarDist 3D WACV paper | 3D object detection and segmentation in microscopy. | https://openaccess.thecvf.com/content_WACV_2020/html/Weigert_Star-Convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.html |
| nnU-Net | Strong self-configuring segmentation baseline. | https://github.com/MIC-DKFZ/nnUNet |
| u-Segment3D | Converts 2D segmentations into 3D consensus segmentation. | https://github.com/DanuserLab/u-segment3D |
| BioImage Model Zoo | Standardized pretrained bioimage models. | https://bioimage.io/ |

## Data, Benchmarks, And Pretraining Sources

| Source | Relevance | URL |
|---|---|---|
| Zebrahub | Closest biological context: zebrafish development atlas with light-sheet imaging. | https://zebrahub.sf.czbiohub.org/ |
| Zebrahub paper | Published multimodal zebrafish developmental atlas. | https://www.cell.com/cell/fulltext/S0092-8674(24)01147-4 |
| Zebrahub PubMed | PubMed route for the Cell paper. | https://pubmed.ncbi.nlm.nih.gov/39454574/ |
| Zebrahub Biohub news | Plain-language host explanation of imaging and scRNA-seq resources. | https://biohub.org/blog/zebrahub-tracks-zebrafish-development/ |
| Cell Tracking Challenge datasets | 2D+t and 3D+t benchmark datasets. | https://celltrackingchallenge.net/datasets/ |
| Cell Tracking Challenge benchmark | Current benchmark results and algorithms. | https://celltrackingchallenge.net/latest-ctb-results/ |
| CTC 10-year paper | Objective benchmarking history and dataset expansion. | https://www.nature.com/articles/s41592-023-01879-y |
| BlastoSPIM | 3D nuclear instance segmentation/tracking in mouse embryo data. | https://blastospim.flatironinstitute.org/html/ |
| BlastoSPIM paper | Published nuclear instance segmentation/tracking dataset. | https://journals.biologists.com/dev/article/151/21/dev202817/362603/Nuclear-instance-segmentation-and-tracking-for |
| BlastoSPIM processing pipeline | Practical segmentation/tracking workflow. | https://princetonuniversity.github.io/blastospim-processing-pipeline/ |
| DeepCell Datasets | Biological image datasets with single-cell annotations. | https://datasets.deepcell.org/ |
| DynamicNuclearNet source route | DeepCell dataset entry and Caliban route. | https://github.com/vanvalenlab/deepcell-datasets |
| Caliban docs | Nuclear segmentation and tracking pipeline. | https://deepcell.readthedocs.io/en/latest/app-gallery/caliban.html |
| Caliban repository | DynamicNuclearNet and Caliban analysis scripts. | https://github.com/vanvalenlab/Caliban-2024_Schwartz_et_al |
| LIVECell | Large manually annotated cell segmentation dataset. | https://www.nature.com/articles/s41592-021-01249-6 |
| IDR / OMERO | Public imaging repository context. | https://idr.openmicroscopy.org/ |
| BBBC035 | Simulated HL60 CTC dataset route. | https://bbbc.broadinstitute.org/BBBC035 |
| BBBC032 | Mouse embryo blastocyst 3D nuclei route. | https://bbbc.broadinstitute.org/BBBC032 |

## Evaluation Sources

| Source | Role | URL |
|---|---|---|
| Official metric spec | Exact competition scoring. | https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md |
| traccuracy GitHub | General tracking evaluation toolkit. | https://github.com/live-image-tracking-tools/traccuracy |
| traccuracy CTC example | CTC metrics and division evaluation examples. | https://traccuracy.readthedocs.io/en/latest/examples/ctc.html |
| py-ctcmetrics GitHub | Python implementation of CTC, MOTChallenge, and CHOTA metrics. | https://github.com/celltrackingchallenge/py-ctcmetrics |
| py-ctcmetrics PyPI | Installable package route. | https://pypi.org/project/py-ctcmetrics/ |
| CHOTA arXiv | Higher-order lineage-sensitive tracking metric. | https://arxiv.org/abs/2408.11571 |

## arXiv Leads From Local Search

| Paper | Why to monitor | URL |
|---|---|---|
| Trackastra: Transformer-based cell tracking for live-cell microscopy | Directly relevant transformer association model. | https://arxiv.org/abs/2405.15700 |
| How To Make Your Cell Tracker Say I dunno! | Uncertainty-aware cell tracking signal. | https://arxiv.org/abs/2503.09244 |
| EfficientCellSeg | Volumetric cell segmentation lead. | https://arxiv.org/abs/2204.03014 |
| Revisiting foundation models for cell instance segmentation | Recent foundation-model audit for cell segmentation. | https://arxiv.org/abs/2603.17845 |
| UCell: rethinking generalizability and scaling of bio-medical vision models | Generalization/scaling signal for biomedical vision models. | https://arxiv.org/abs/2604.00243 |
| CHOTA: A Higher Order Accuracy Metric for Cell Tracking | Metric signal for lineage/global coherence. | https://arxiv.org/abs/2408.11571 |

## Search Queries Worth Repeating

- `biohub-cell-tracking-during-development notebook gap recovery GEFF`
- `biohub-cell-tracking-during-development unet ilp reproduction`
- `biohub-cell-tracking-during-development division image support`
- `biohub-cell-tracking-during-development mintrack recall`
- `site:kaggle.com/code biohub-cell-tracking-during-development geff submission.csv run_stats`
- `site:kaggle.com/competitions/biohub-cell-tracking-during-development/discussion overlap train test`
- `Royer Lab tracksdata GEFF Ultrack zebrafish embryo cell tracking`
- `Trackastra pretrained models cell tracking transformer divisions`
- `Cellpose-SAM cpsam_v2 cpdino 3D microscopy nuclei segmentation`
- `Zebrahub lineage reconstruction light-sheet imaging tracks`

## Related Biohub Documents

- [README](./README.md)
- [Devastating Double Check - 2026-07-09](./Devastating_Double_Check_2026-07-09.md)
- [Kaggle Notebook and Discussion Radar](./Kaggle_Notebook_Discussion_Radar_2026-07-09.md)
- [ML and AI Model, Feature, Weight, and Calibration Atlas](./ML_AI_Model_Feature_Calibration_Atlas_2026-07-09.md)
- [Reproducibility and Model Roadmap](./Reproducibility_Model_Roadmap_2026-07-09.md)
