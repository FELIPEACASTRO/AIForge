# 🐟 Biohub — Cell Tracking During Development (Kaggle Playbook)

> A fact-checked playbook for the Kaggle **[Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)** competition — detect and track cells in 3D over time in real zebrafish light-sheet data from the **Chan Zuckerberg Biohub / Royer Group**. Covers the exact scoring metric, the best segmentation and tracking tools, pretraining datasets, local evaluation, public notebooks, and a concrete recipe for the top of the leaderboard. **Every link verified live (July 2026).**

## 📚 Pages

| Page | What's inside |
|---|---|
| [Competition Overview & Scoring](./Competition_Overview_and_Scoring.md) | The task, the OME-Zarr / `tracksdata` data model, and the **exact metric decoded** — `score = adjusted_edge_jaccard + 0.1·division_jaccard` (7 µm node matching, ±1-frame divisions, over-detection penalty) — plus a scoring strategy. |
| [Segmentation & Tracking Tools](./Segmentation_and_Tracking_Tools.md) | Best tools per pipeline stage: detection (**Cellpose-SAM**, **StarDist-3D**, **nnU-Net**) → linking (**Ultrack** — the host's own tool, **Trackastra** — CTC-2024 winner, **DeepCell**, **btrack**). |
| [Datasets, Metrics & Notebooks](./Datasets_Metrics_and_Notebooks.md) | Local eval (**traccuracy**, **py-ctcmetrics**), pretraining data (**Zebrahub**, **Cell Tracking Challenge**, **BlastoSPIM**, **DynamicNuclearNet**), public notebooks, and the winning recipe. |

### 📖 Field compendium (everything on the subject, beyond the competition)

| Page | What's inside |
|---|---|
| [Segmentation Methods — Full Compendium](./Segmentation_Methods_Compendium.md) | All detectors: Cellpose / **Cellpose3** / **Cellpose-SAM** / **Omnipose**, **StarDist-3D**, **InstanSeg**, **EmbedSeg**, **SplineDist**, **nnU-Net**, **CellSAM**, restoration (CARE/N2V), ilastik, Celldetective. |
| [Tracking Methods — Full Compendium](./Tracking_Methods_Compendium.md) | All trackers by paradigm: multi-hypothesis (**Ultrack**, Biological-Needs), classical LAP (**u-track**, **TrackMate**, trackpy, **LapTrack**, **btrack**), graph/flow (**EmbedTrack/KIT-GE**, **PyUAT**), DL/GNN/transformer (**Trackastra**, **GNN tracking**, **DeepCell**, **DeLTA**, **Cell-as-Point**, **ELEPHANT**), uncertainty & SAM2. |
| [Frameworks, Platforms & Extra Datasets](./Frameworks_Platforms_and_Extra_Datasets.md) | **napari**, Fiji/ImageJ, **ilastik**, **inTRACKtive**, **BIII**; no-code training (**ZeroCostDL4Mic**, **BiaPy**, **DeepImageJ**, **BioImage Model Zoo**); OME-Zarr/zarr/dask; extra datasets (**LIVECell**, **EVICAN**, **IDR/OMERO**, CTC). |
| [Surveys, Benchmarks & Key Papers](./Surveys_Benchmarks_and_Key_Papers.md) | Review articles (198-paper segmentation survey; CTC 10-years), the Cell Tracking Challenge benchmark, and the landmark methods papers with a suggested reading path. |
| [Official Source Map - 2026-07-08](./Official_Source_Map_2026-07-08.md) | Authoritative source index for Kaggle, Biohub/Royer Lab, official metric, data model, `tracksdata`, Ultrack, Trackastra, OME-Zarr, datasets, and benchmark context. |
| [Current Competition Intelligence - 2026-07-08](./Current_Competition_Intelligence_2026-07-08.md) | Kaggle CLI snapshot: competition metadata, public leaderboard frontier, top notebooks by votes, recent notebook activity, and strategy signals. |
| [Kaggle API and CLI Extraction Plan - 2026-07-08](./Kaggle_API_CLI_Extraction_Plan_2026-07-08.md) | Official Kaggle CLI/API audit: authentication handling, reproducible extraction commands, Python API method map, output-field limits, and country-evidence rules. |
| [Global Country Bioimaging Coverage](./Global_Country_Bioimaging_Coverage/) | All-country search matrix for Biohub-related cell tracking, bioimage analysis, developmental microscopy, OME-Zarr/image data, and microscopy infrastructure. |
| [Global Direct Competition Country Coverage](./Global_Direct_Competition_Country_Coverage/) | Strict all-country matrix for direct public evidence about the exact Biohub Kaggle competition; avoids country inference from names or handles. |

## ⚡ TL;DR — how to reach the best score

1. **Edges dominate** (weight 1.0 vs 0.1 for divisions): nail **detection within 7 µm + temporal linking**.
2. **Don't over-detect** — the penalty `max(0, jaccard·(1 − 0.1·(T_pred−T_true)/T_true))` punishes flooding the graph with nodes.
3. **Use the host's own tracker, Ultrack**, and/or the transformer **Trackastra**; **ensemble the edges** and add **gap recovery**.
4. **Evaluate locally with traccuracy** (mirror `edge_jaccard + 0.1·division_jaccard`) so you tune without spending submissions.
5. **Divisions are a small top-up** (±1 frame) — never trade edge precision for them.

## Related in AIForge

- Parent: [`../`](../) (Kaggle competitions & winning solutions) · [`../../`](../../) (AI Project Showcases)
- Science context — developmental-biology cell tracking research, atlases (Zebrahub), and tools (Ultrack/inTRACKtive/Trackastra): [`../../../15_Science_AI/`](../../../15_Science_AI/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Computer_Vision/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Computer_Vision/)

**Keywords:** Biohub cell tracking Kaggle, cell tracking during development, zebrafish light-sheet, 3D cell tracking competition, Ultrack, Trackastra, Cellpose-SAM, StarDist-3D, edge Jaccard, division Jaccard, tracksdata, OME-Zarr, traccuracy, Zebrahub, BlastoSPIM, Cell Tracking Challenge, CZ Biohub, Royer Group, Kaggle leaderboard strategy, embryo cell lineage.
