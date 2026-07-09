# Devastating Double Check - Biohub Cell Tracking - 2026-07-09

> Four-agent, read-only verification pass for `Biohub - Cell Tracking During Development`. This file consolidates public Kaggle CLI/API signals, Hugging Face model metadata, Kaggle weight-pack file listings, official metric evidence, model-family routes, calibration constants, and caveats. It is intentionally strict about separating verified evidence from inference.

## Scope And Safety

| Item | Status |
|---|---|
| Competition | `biohub-cell-tracking-during-development` |
| Pass date | 2026-07-09 |
| Repository language | English |
| Kaggle operations | Read-only metadata, leaderboard, notebook listings, topic listings, notebook-code pulls, and dataset file listings. No submission. |
| Hugging Face operations | Public model metadata and space/model discovery. No private token was required. |
| Secret handling | `C:\Users\davis\Workspace\KG-GPT\CHAVE.txt` was not read or printed. Public tools were sufficient. |
| Large data handling | Raw competition data and large notebook outputs were not committed into AIForge. |
| Local organization | Findings are stored under `05_VERTICAL_APPLICATIONS/20_AI_Project_Showcases/Kaggle/Biohub_Cell_Tracking/`. |

## Agent Cross-Check

| Agent | Audit axis | Key result |
|---|---|---|
| Plato | Hugging Face, weights, model packs, model cards, spaces | Confirmed direct model/weight routes: Cellpose-SAM, Cellpose zebrafish fine-tunes, Cellpose/CellSAM ONNX, SAM2 cell segmentation, StarDist 3D Xenopus, MedSAM2, Trackastra mirrors, and Biohub Kaggle weight packs. |
| Faraday | Kaggle CLI/API, notebooks, leaderboard, discussion, output metadata | Confirmed current competition metadata, top public leaderboard score, public notebook ref inventory, output file families, and limits of public team/submission APIs. |
| Meitner | Algorithms, metrics, calibration, features, datasets | Confirmed the scoring bottleneck: graph nodes and temporal edges dominate; divisions are a smaller add-on. Recommended tracksdata/GEFF plus U-Net/Cellpose/StarDist proposals and LAP/ILP/Trackastra/Ultrack linking. |
| Ramanujan | Repository organization and consistency | Found and validated navigation issues: split deep-source and radar navigation rows, update Biohub directory `lastmod`, and add local links among long-form docs. |

## Latest Volatile Kaggle Snapshot

Verified by read-only Kaggle CLI on 2026-07-09.

| Field | Latest observed value | Evidence limit |
|---|---|---|
| Competition ref | `biohub-cell-tracking-during-development` | Exact CLI competition ref. |
| Category | Research | Public metadata. |
| Reward | 60,000 USD | Public metadata. |
| Deadline | 2026-09-29 23:59 UTC | Public metadata. |
| Team count | 951 | Volatile; a previous snapshot had 950. Treat as time-sensitive. |
| User entered | true | Local Kaggle account state. |
| Top visible public score | 0.910 | Public leaderboard page exposed by CLI. |
| Next visible scores | 0.906, 0.905, 0.905, 0.904 | Public top-page snapshot only. |
| Leaderboard fields exposed | `teamId`, `teamName`, `submissionDate`, `score` | No reliable country field. |

Important caveat: public leaderboard and public notebook metadata are not the private leaderboard, not a complete team submission history, and not proof of hidden-test behavior.

## Official Metric And Geometry Facts

| Fact | Verification | Competition impact |
|---|---|---|
| The task is sparse graph tracking, not only segmentation. | Official repo and data model use OME-Zarr volumes plus GEFF/tracksdata graph outputs. | A successful pipeline must output node rows and edge rows correctly. |
| Data geometry is `(T, Z, Y, X)`. | Official competition scaffold and notebook configs. | Axis mistakes can produce visually plausible but score-wrong predictions. |
| Official voxel scale is `Z,Y,X = 1.625, 0.40625, 0.40625` microns. | Official metric/scaffold and notebook configs. | All NMS, link distance, gap recovery, and division geometry should be in microns. |
| Node matching radius is `7 um`. | Official metric. | Detector threshold and NMS calibration dominate early score movement. |
| Score is edge-heavy: `adjusted_edge_jaccard + 0.1 * division_jaccard`. | Official metric. | Optimize node/edge precision and recall before chasing divisions. |
| Over-detection is penalized. | Official adjusted edge Jaccard. | Flooding the graph with nodes can lower score even if recall improves. |

Sources:

- Official competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- Official repo: https://github.com/royerlab/kaggle-cell-tracking-competition
- Official metric notes: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- tracksdata: https://github.com/royerlab/tracksdata
- GEFF: https://live-image-tracking-tools.github.io/geff/

## Verified Kaggle Weight Packs

These are not generic search hits. They were checked through `kaggle datasets files`, so the listed files are public dataset file-list evidence. Licenses were not always visible through the CLI; verify before reuse.

| Grade | Dataset | Verified files | Why it matters |
|---|---|---|---|
| A | `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` | `weights/full_frame_center/best.pt`, `checkpoint_last.pt`, `config.json`, gate metrics, source training scripts, manifests. | Direct full-frame 3D center detector pack for Biohub-style center proposals. |
| A | `subinium/biohub-v34-retrain-weights-mirror` | `unet_transformer/split_0/edge_predictor_best.pth`, `config.json`. | Direct edge predictor / U-Net-transformer weight mirror. |
| A | `subinium/biohub-trackastra-public-weights-mirror` | `ctc/model.pt`, `general_2d/model.pt`, train configs, Trackastra wheel, GEFF/Zarr dependency wheels. | Ready offline route for Trackastra-style learned association inside Kaggle notebooks. |
| A | `drkongvis/biohub-v4-3dunet-pretrained-weights` | `unet3d.pt`, `unet3d_pretrained.pt`. | Compact direct detector-pretraining pack for 3D U-Net proposals. |

Additional Kaggle model packs worth inspecting next:

- `justinkim1216/biohub-nnunet-center-support-v1`
- `justinkim1216/biohub-nnunet-flow-support-v1`
- `mdaliazad/biohub-cell-tracking-model`
- `addisonhoward/biohub-unet-weights`
- `drkongvis/biohub-v1-diffusion-refine-weights`

## Verified Hugging Face Model Leads

Checked with public Hugging Face repo metadata. No HF secret was needed.

| Grade | Repo | Type | License signal | Biohub use |
|---|---|---|---|---|
| A | https://huggingface.co/mouseland/cellpose-sam | Cellpose-SAM / CellposeDINO segmentation weights | BSD-3-Clause | Strong detector proposal route; tune 2D/3D tiling, diameter, and thresholds. |
| B | https://huggingface.co/SDu90/zebrafish-cellpose-finetunes | Zebrafish Cellpose-SAM fine-tunes | CC-BY-NC-4.0 | Closest biology hit; likely 2D/slice transfer, not direct lineage tracking. |
| B | https://huggingface.co/keejkrej/cellpose-cpsam-onnx | ONNX Cellpose-SAM | BSD-3-Clause | Deployment/parity testing route. |
| B | https://huggingface.co/DnaRnaProteins/sam2-cells-seg | SAM2 tiny cell segmentation fine-tune | Apache-2.0 | Prompt-oriented 2D cell segmentation route; not a tracker. |
| B | https://huggingface.co/keejkrej/cellsam-onnx | CellSAM ONNX | Other license | Segmenter route; license and official-weight provenance need verification. |
| B | https://huggingface.co/KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari | StarDist 3D nuclei | BSD-3-Clause | Embryo-like 3D nuclei proposal route with domain shift. |
| B | https://huggingface.co/KapoorLabs-Copenhagen/xenopus-stardist3d-membrane-mari | StarDist 3D membrane | BSD-3-Clause | 3D membrane/cell proposal route with domain shift. |
| C | https://huggingface.co/wanglab/MedSAM2 | Medical SAM2 segmentation | CC-BY-SA-4.0 | Architecture/volumetric prompting ideas, not direct Biohub inference. |

Related spaces:

- https://huggingface.co/spaces/mouseland/cellpose
- https://huggingface.co/spaces/MicroAtlas/microatlas

HF caveat: broad semantic searches can miss domain-specific repositories. Direct known-ID lookup found relevant repos even when generic model search returned sparse results.

## Algorithm Priority Stack

| Priority | Component | Best current route |
|---|---|---|
| P0 | Metric-faithful local evaluation | Official metric, `tracksdata`, `traccuracy`, `py-ctcmetrics`, GEFF validation. |
| P0 | Candidate centers | Official U-Net, Biohub 3D U-Net weight packs, DoG/LoG baselines, Cellpose-SAM/CellposeDINO, StarDist 3D. |
| P0 | Temporal linking | LAP/Hungarian baseline, ILP global graph, Trackastra learned association, Ultrack multi-hypothesis tracking. |
| P0 | Graph repair | Single-parent/child constraints, short-track filtering, one-frame gap recovery, conservative two-frame gap recovery, physical-motion relinking. |
| P1 | Division recovery | Conservative post-link second-child insertion; do after edge backbone is stable. |
| P1 | Ensembling | Blend preprocessing, detector consensus, edge score fusion, per-dataset threshold profiles. |
| P2 | Foundation models | CellSAM, micro-sam, SAM2, MedSAM2 for pseudo-labeling, QA, or slice proposals. |

Core public sources:

- Ultrack: https://github.com/royerlab/ultrack
- Ultrack paper: https://www.nature.com/articles/s41592-025-02778-0
- Trackastra: https://github.com/weigertlab/trackastra
- Trackastra paper: https://arxiv.org/abs/2405.15700
- Cellpose: https://github.com/mouseland/cellpose
- Cellpose models: https://cellpose.readthedocs.io/en/latest/models.html
- CellSAM: https://github.com/vanvalenlab/cellsam
- StarDist: https://github.com/stardist/stardist
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet
- micro-sam: https://github.com/computational-cell-analytics/micro-sam
- btrack: https://github.com/quantumjot/btrack

## Notebook-Pulled Calibration Constants

High-value public notebook code was pulled into a temporary audit directory, not committed as raw notebooks. The strongest inspected notebook family uses a U-Net/transformer graph route with ILP and postprocessing. Exact observed constants include:

| Calibration area | Observed values | Why to sweep |
|---|---|---|
| Detection threshold | `DET_THRESHOLD = 0.99` in inspected LB897 config; nearby variants mentioned 0.985-0.99. | Controls the node-flooding vs missed-node boundary. |
| Edge distance cap | `OUTPUT_EDGE_MAX_UM = 14.0` | Drops physically implausible edges before graph repair. |
| Motion relink | tight `6.0 um`, relaxed `10.0 um`, velocity weight `0.5`, learned bonus `0.75`. | Rebuilds one-to-one temporal graph from node positions and learned-edge hints. |
| Gap close | max gap `1`, `GAP_CLOSE_UM = 6.0`, reuse radius `3.2`, max added fraction `0.05`, max added absolute `2000`. | Repairs one-frame missed detections while capping node injection. |
| Gap2 recovery | default off in inspected config; if enabled: total `10.2 um`, step `4.4 um`, link fraction cap `0.0045`, absolute cap `180`. | Two-missing-frame repair is powerful but risky. |
| Short-track filtering | default `OUTPUT_MIN_TRACK_LEN = 6`; high-confidence variant set global `7` and one dataset-specific `6`. | Removes false positives but can delete real short tracks. |
| Safe divisions | parent-child `4.7 um`, sister `7.2 um`, existing-child `7.8 um`, frame cap `0.008`, global cap `0.004`. | Adds divisions only where geometry is tight. |
| Line-fit smoothing | enabled, weight `0.8`, window `2`. | Stabilizes coordinates without turning into free-form trajectory generation. |
| ILP weights | edge `-1.0`, appearance `0.1`, disappearance `0.1`, division `1.0`. | Cost tradeoffs decide graph topology. |
| Weights path | `weights/unet_transformer/split_0/edge_predictor_best.pth` | Confirms direct link between public weight pack and notebook graph model. |

The postprocessing formula in pulled notebook code uses physical distance:

```text
distance_um = sqrt((1.625 * dz)^2 + (0.40625 * dy)^2 + (0.40625 * dx)^2)
```

This confirms that calibration should be in microns, not raw voxel units.

## Kaggle Notebook And Discussion Signals

| Signal | Verified status | Action |
|---|---|---|
| Public notebook output listings expose `run_stats.csv`, `submission.csv`, `kaggle_test_splits_50ep.json`, and GEFF/Zarr graph outputs. | Verified through `kaggle kernels files` and agent API probes. | Parse `run_stats.csv` before downloading heavy outputs. |
| Public notebook discovery found 181 unique refs across sampled listings. | Agent API probe. | Build a source inventory, but treat it as sampled, not exhaustive. |
| `scoreDescending` notebook list highlights V4 U-Net ILP, LB897, learned graph gap recovery, blend preprocessing, short-track filters, and relink variants. | Verified through CLI metadata. | Prioritize graph/postprocess diffing. |
| Discussion topic about public example test/train overlap exists. | Topic title signal only. | Monitor host clarification; do not treat as hidden-test leakage. Kaggle data page states public example test samples can be copies from train and hidden test is swapped on notebook submission. |
| Public leaderboard exposes no country evidence. | Verified. | Keep country matrices strict; no inference from names or handles. |

## Dataset And Pretraining Routes

| Grade | Dataset or benchmark | Use |
|---|---|---|
| A | Official Biohub data | Source of truth for geometry, sparsity, metric, schema, and hidden-test behavior. |
| B+ | Zebrahub | Closest zebrafish developmental-biology context. |
| B | Cell Tracking Challenge | General tracking benchmark and CTC-compatible tooling. |
| B | BlastoSPIM | 3D embryo nuclei training/validation ideas. |
| B | DynamicNuclearNet | Nuclear segmentation/tracking and lineage data. |
| B | LIVECell | Large 2D cell segmentation pretraining. |
| B | BBBC032 / BBBC035 | Embryo and CTC-style synthetic validation sources. |
| C | TYC dataset | Motion/instance segmentation ideas with domain shift. |
| C | IDR/OMERO and BioImage.IO | Model/dataset discovery and transferable bioimage models. |

Sources:

- Zebrahub: https://zebrahub.sf.czbiohub.org/
- Cell Tracking Challenge datasets: https://celltrackingchallenge.net/datasets/
- Cell Tracking Challenge results: https://celltrackingchallenge.net/latest-ctb-results/
- CTC 10-year paper: https://www.nature.com/articles/s41592-023-01879-y
- BlastoSPIM: https://blastospim.flatironinstitute.org/html/
- DynamicNuclearNet: https://deepcell.readthedocs.io/en/master/data-gallery/dynamicnuclearnet.html
- LIVECell: https://www.nature.com/articles/s41592-021-01249-6
- BioImage.IO: https://bioimage.io/

## Mismatch And Correction Log

| Issue | Correction |
|---|---|
| Team count drifted from 950 to 951 between snapshots. | Current docs should say latest observed 951 and mark the value as volatile. |
| `NAVIGATION_GUIDE.md` used one row for both deep source atlas and radar. | Split rows so each file is discoverable. |
| Biohub directory `lastmod` lagged behind new 2026-07-09 child docs. | Update directory and Biohub README/doc entries in `sitemap.xml`. |
| Broad HF searches can miss relevant cell-segmentation repos. | Use known-ID lookup plus direct project sources. |
| Public discussion overlap title can be over-interpreted. | Keep it as risk-monitoring only unless host or data evidence confirms hidden-test impact. |
| Country coverage can be tempting to infer from usernames. | Continue strict D0/D1/D2 evidence rules; Kaggle public fields are insufficient. |
| Raw notebooks/outputs can clutter AIForge. | Keep raw pulls in temp or artifact manifests; commit curated English summaries only. |

## Query And Prompt Patterns To Repeat

These are useful search prompts for future agents and humans. They are intentionally precise rather than broad.

```text
site:kaggle.com/code biohub-cell-tracking-during-development run_stats geff submission
site:kaggle.com/code biohub-cell-tracking-during-development gap recovery edge_predictor_best
site:kaggle.com/datasets biohub cell tracking weights edge_predictor_best
site:huggingface.co cellpose sam zebrafish cell segmentation
site:huggingface.co Biohub cell tracking model weights
Trackastra pretrained models GEFF cell tracking divisions
Ultrack optimization division_weight max_distance gap closing embryo tracking
Cellpose-SAM cpsam_v2 cpdino microscopy nuclei segmentation
Royer Lab tracksdata geff zebrafish embryo cell tracking
```

LLM audit prompt:

```text
Given a Biohub public notebook, extract only verifiable implementation facts:
detector architecture, weight paths, input axes, micron scale, node threshold,
NMS radius, max link distance, gap-close rules, min track length, division
rules, output schema, package dependencies, and run_stats columns. Separate
verified constants from inferred strategy, and do not infer country or private
team information.
```

## Next Double-Check Queue

1. Pull and diff the top 20 public notebooks by score, votes, and latest run.
2. Query `GetKernel.bestPublicScore` for the full public notebook candidate set.
3. List files for all Biohub-related Kaggle datasets that contain `weight`, `model`, `checkpoint`, `nnunet`, `trackastra`, `unet`, `flow`, `center`, or `diffusion`.
4. Parse selected `run_stats.csv` outputs and build a method-score matrix.
5. Reproduce the official metric locally and check whether each public notebook's diagnostic columns map to edge Jaccard or division Jaccard.
6. Build a calibration sweep table for detection threshold, min track length, gap close, motion relink, and safe division caps.
7. Re-check HF for new Biohub/Cellpose/Trackastra/CellSAM repos weekly; the hub search index can lag or miss known IDs.
8. Monitor official discussions for scoring, data, and notebook-environment clarifications.

## Related Biohub Documents

- [README](./README.md)
- [Deep Source Atlas](./Deep_Source_Atlas_2026-07-09.md)
- [Kaggle Notebook and Discussion Radar](./Kaggle_Notebook_Discussion_Radar_2026-07-09.md)
- [ML and AI Model, Feature, Weight, and Calibration Atlas](./ML_AI_Model_Feature_Calibration_Atlas_2026-07-09.md)
- [Reproducibility and Model Roadmap](./Reproducibility_Model_Roadmap_2026-07-09.md)
- [Kaggle API and CLI Extraction Plan](./Kaggle_API_CLI_Extraction_Plan_2026-07-08.md)
