# Biohub Cell Tracking - Official Source Map

> Search snapshot for Biohub - Cell Tracking During Development and the surrounding cell-tracking ecosystem. Captured after pulling `origin/master` on 2026-07-08. This page separates authoritative competition sources from supporting methods, datasets, tools, and public community leads.

## Authoritative Competition Sources

| Source | Why it matters | URL |
|---|---|---|
| Kaggle competition page | Official task, rules, deadline, prize, data, code, discussions, and leaderboard entry point. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development |
| Kaggle data tab | Official data description: detect cells at each timepoint, link cells across time, and output a tracking graph. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data |
| Kaggle code tab | Public notebooks and baselines using the competition data. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/code |
| Kaggle leaderboard | Live public leaderboard; volatile, so use only as a timestamped snapshot. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard |
| Biohub Kaggle organization | Host profile for Chan Zuckerberg Biohub competition activity. | https://www.kaggle.com/organizations/biohub |
| Official competition repo | Baseline implementation, data loading, training, prediction, evaluation, and visualization code. | https://github.com/royerlab/kaggle-cell-tracking-competition |
| Official metric spec | Source of truth for edge Jaccard, division Jaccard, micro-averaging, and final score formula. | https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md |

## Official Kaggle API And CLI Sources

| Source | What to use it for | URL |
|---|---|---|
| Kaggle API documentation | Authentication, official API access, and CLI/API setup. | https://www.kaggle.com/docs/api |
| Kaggle CLI repository | Official CLI source and release documentation. | https://github.com/Kaggle/kaggle-cli |
| Kaggle CLI docs README | Command-family map for competitions, datasets, forums, kernels, models, benchmarks, and configuration. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md |
| Competitions CLI docs | Competition metadata, files, downloads, submissions, leaderboard, team submissions, pages, and topics. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md |
| Kernels CLI docs | Public notebook discovery, notebook files, pulls, outputs, logs, status, and topics. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md |
| Forums CLI docs | Wider Kaggle forum and topic discovery routes. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/forums.md |
| Output format docs | Structured `json`/`csv` output and field projection for reproducible snapshots. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/output_format.md |

## Data Model And Metric Facts

| Topic | Confirmed detail | Evidence |
|---|---|---|
| Image format | OME-Zarr image volumes ordered as `(T, Z, Y, X)`. | Official repo README |
| Spatial scale | Official repo documents `Z, Y, X = 1.625, 0.40625, 0.40625` microns per pixel. | Official repo README |
| Track format | Tracks use GEFF / `tracksdata`: nodes are approximate cell centers with `(t, z, y, x)` and edges connect cells over time. | Official repo README and `tracksdata` repo |
| Sparse labels | Ground truth is sparse; only a subset of cells is annotated. | Official repo README and metric spec |
| Edge matching | Nodes match by centroid distance with a maximum distance of 7 micrometers; edge TP/FP/FN then form edge Jaccard. | Official metric spec |
| Division scoring | Divisions have a tolerance of plus or minus one timepoint and contribute through division Jaccard. | Official metric spec |
| Final score | `score = adjusted_edge_jaccard + 0.1 * division_jaccard`. | Official metric spec |
| Submission object | A predicted graph can be represented as a `tracksdata.InMemoryGraph`. | Official repo README |

## Biohub And Royer Lab Context

| Source | What to extract | URL |
|---|---|---|
| Royer Group at Biohub | Scientific context: time-resolved, multidimensional zebrafish atlas; light-sheet microscopy; deep-learning image analysis. | https://biohub.org/royer/ |
| Biohub Ultrack news | Plain-language explanation of why Ultrack matters for whole-embryo and large-scale 3D tracking. | https://biohub.org/blog/dynamic-duo-powerful-tools-show-cells-in-action/ |
| Zebrahub Biohub news | Developmental-atlas context for zebrafish embryo imaging and multimodal data. | https://biohub.org/blog/zebrahub-tracks-zebrafish-development/ |
| Royer Lab GitHub org | Related repos: `ultrack`, `inTRACKtive`, `tracksdata`, `ultrack-td`, `img-embed-td`, and `contrastive-td`. | https://github.com/royerlab |
| Public Ultrack datasets | Public OME-Zarr examples, including zebrafish embryo and zebrafish neuromast data. | https://public.czbiohub.org/royerlab/ultrack/ |

## Core Royer / Biohub Stack

| Tool | Competition relevance | URL |
|---|---|---|
| Ultrack | Host-associated multi-hypothesis segmentation and tracking; highly relevant baseline and ensemble member. | https://royerlab.github.io/ultrack/ |
| Ultrack GitHub | Source code, examples, plugins, and install details. | https://github.com/royerlab/ultrack |
| Ultrack Nature Methods 2025 | Peer-reviewed method paper; validates large-scale 2D/3D/multichannel timelapse tracking. | https://www.nature.com/articles/s41592-025-02778-0 |
| inTRACKtive | Browser-based interactive cell-track visualization and sharing; useful for inspecting lineage outputs. | https://github.com/royerlab/inTRACKtive |
| inTRACKtive Nature Methods 2025 | Companion visualization paper for large lineage datasets. | https://www.nature.com/articles/s41592-025-02777-1 |
| tracksdata | Graph data model for multi-object tracking, with in-memory and SQL backends and CTC compatibility. | https://github.com/royerlab/tracksdata |
| tracksdata docs | API docs for graph representation and operations. | https://royerlab.github.io/tracksdata/ |
| ultrack-td | C++ rewrite of Ultrack core segmentation with tracksdata compatibility. | https://github.com/royerlab/ultrack-td |

## High-Value Methods To Monitor

| Method | Why it matters for Biohub | URL |
|---|---|---|
| Trackastra | Transformer-based linker with pretrained models; won the 7th Cell Tracking Challenge at ISBI 2024. | https://github.com/weigertlab/trackastra |
| Trackastra paper | Directly learns association costs between segmented objects in temporal windows. | https://arxiv.org/abs/2405.15700 |
| Cellpose / Cellpose-SAM | Strong current generalist cell/nucleus segmenter with 3D use and recent Cellpose-SAM / Cellpose4 updates. | https://github.com/mouseland/cellpose |
| Cellpose-SAM preprint | SAM-backed Cellpose model for broad cell and nucleus segmentation generalization. | https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1 |
| StarDist | 2D/3D star-convex object detection for dense nuclei. | https://github.com/stardist/stardist |
| StarDist 3D paper | Star-convex polyhedra for 3D object detection and segmentation in microscopy. | https://openaccess.thecvf.com/content_WACV_2020/html/Weigert_Star-Convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.html |
| CellSAM | Foundation model for cell segmentation across imaging modalities. | https://github.com/vanvalenlab/cellsam |
| CellSAM Nature Methods 2025 | CellSAM paper and universal cell segmentation context. | https://www.nature.com/articles/s41592-025-02879-w |
| Cellpose3 | Restoration plus segmentation; relevant for noisy or anisotropic light-sheet data. | https://www.nature.com/articles/s41592-025-02595-5 |

## Datasets And Benchmarks

| Dataset or benchmark | Why it matters | URL |
|---|---|---|
| Zebrahub | Closest biological context: zebrafish developmental atlas. | https://zebrahub.sf.czbiohub.org/ |
| Zebrahub Cell paper | Published atlas paper for zebrafish development context. | https://www.cell.com/cell/fulltext/S0092-8674(24)01147-4 |
| Zebrahub Zenodo data | Downloadable data archive for Zebrahub resources. | https://zenodo.org/records/13761503 |
| Cell Tracking Challenge 3D datasets | Benchmark 3D+t tracking datasets for pretraining and validation ideas. | https://celltrackingchallenge.net/3d-datasets/ |
| Cell Tracking Challenge results | Current public benchmark results and method comparison. | https://celltrackingchallenge.net/latest-ctb-results/ |
| CTC 10-year paper | Benchmark history and objective tracking-evaluation context. | https://www.nature.com/articles/s41592-023-01879-y |
| BlastoSPIM | 3D nuclei ground truth for mouse embryo pretraining/robustness. | https://blastospim.flatironinstitute.org/html/ |
| DynamicNuclearNet | Large live-cell nuclei tracking dataset from DeepCell ecosystem. | https://www.biorxiv.org/content/10.1101/803205v4.full |
| LIVECell | Large manually annotated cell segmentation dataset for 2D detector pretraining. | https://www.nature.com/articles/s41592-021-01249-6 |
| IDR / OMERO | Public imaging studies and large bioimage repository context. | https://idr.openmicroscopy.org/ |

## Data Engineering And Viewer Stack

| Tool or spec | Role | URL |
|---|---|---|
| OME-Zarr / OME-NGFF | Cloud-friendly chunked n-dimensional bioimage storage, relevant to the competition data layout. | https://ngff.openmicroscopy.org/ |
| OME-Zarr paper | Technical background for the file format. | https://pmc.ncbi.nlm.nih.gov/articles/PMC9980008/ |
| zarr | Chunked array storage layer used by OME-Zarr workflows. | https://zarr.dev/ |
| dask | Out-of-core / parallel processing for large arrays. | https://www.dask.org/ |
| napari | Interactive n-dimensional image viewer and plugin hub. | https://napari.org/ |
| Fiji / ImageJ | Bioimage-analysis platform with TrackMate, Mastodon, and related tooling. | https://fiji.sc/ |
| ilastik | Interactive ML segmentation/tracking and pixel/object classification. | https://www.ilastik.org/ |
| BIII | Searchable registry of bioimage-analysis software. | https://biii.eu/ |

## Community And News Leads

| Source | Use with caution | URL |
|---|---|---|
| FEBS Network announcement | Good summary of competition purpose and June 2026 launch context; secondary source. | https://network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking |
| Kaggle LinkedIn announcement | Confirms public launch framing and prize messaging; secondary/social source. | https://www.linkedin.com/posts/kaggle_biohub-cell-tracking-during-development-activity-7477462443552817152-ljji |
| Kaggle discussion welcome thread | Host/community context and updates; volatile and should be checked live. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716062 |

## Search Queries Used

- `Biohub Cell Tracking During Development Kaggle competition official metric github`
- `site:github.com royerlab kaggle-cell-tracking-competition metrics.md Biohub Cell Tracking During Development`
- `Biohub Cell Tracking During Development Kaggle notebooks baseline`
- `Royer lab Ultrack Nature Methods 2025 cell tracking during development Biohub`
- `Biohub Cell Tracking During Development tracksdata GEFF OME-Zarr`
- `royerlab kaggle-cell-tracking-competition tracksdata OME-Zarr geff`
- `site:kaggle.com/code biohub-cell-tracking-during-development`
- `Trackastra cell tracking transformer Cell Tracking Challenge 2024 paper github`
- `Cellpose-SAM bioimage cell segmentation foundation model paper github`
- `StarDist 3D cell nuclei segmentation github paper`

## Repository Routing

- Official Kaggle API/CLI extraction plan: [Kaggle API and CLI Extraction Plan](./Kaggle_API_CLI_Extraction_Plan_2026-07-08.md)
- Competition strategy and live Kaggle tracking: [Current Competition Intelligence](./Current_Competition_Intelligence_2026-07-08.md)
- Direct country evidence for the exact competition: [Global Direct Competition Country Coverage](./Global_Direct_Competition_Country_Coverage/)
- Country-by-country bioimaging routes: [Global Country Bioimaging Coverage](./Global_Country_Bioimaging_Coverage/)
- Metric and official scoring: [Competition Overview and Scoring](./Competition_Overview_and_Scoring.md)
- Public notebooks and external data: [Datasets, Metrics and Notebooks](./Datasets_Metrics_and_Notebooks.md)
- Tools: [Segmentation and Tracking Tools](./Segmentation_and_Tracking_Tools.md)
- Broad methods: [Segmentation Methods Compendium](./Segmentation_Methods_Compendium.md) and [Tracking Methods Compendium](./Tracking_Methods_Compendium.md)
