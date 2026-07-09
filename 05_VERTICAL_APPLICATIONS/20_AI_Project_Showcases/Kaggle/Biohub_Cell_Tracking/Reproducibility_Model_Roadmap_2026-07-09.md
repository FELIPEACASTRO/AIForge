# Reproducibility And Model Roadmap - Biohub Cell Tracking

> Practical research-to-submission roadmap derived from official competition sources, live Kaggle signals, and the broader cell-tracking ecosystem. The goal is to convert sources into a reproducible, auditable pipeline.

## Core Principle

The competition is not only a segmentation challenge. The official score is graph-based:

- detect cell centers in 3D+t;
- link cells over time;
- encode divisions as one parent with two outgoing edges;
- optimize edge correctness under sparse ground truth;
- avoid excessive node overprediction because adjusted edge Jaccard penalizes node inflation;
- add division quality after the edge/linking backbone is stable.

## Minimum Reproducible Baseline

| Layer | Baseline choice | Why |
|---|---|---|
| Data loading | Official Royer Lab repo | Aligns with OME-Zarr, GEFF, tracksdata, and official evaluation. |
| Detection | nearest-neighbor or DoG baseline from public notebooks | Quick sanity check and debugging baseline. |
| Segmentation | 3D U-Net or Cellpose-SAM/CellposeDINO screening | Stronger cell/nucleus candidates than hand thresholds. |
| Graph construction | tracksdata `InMemoryGraph` | Matches official code path. |
| Linking | LAP/ILP baseline, Trackastra, or Ultrack | Captures temporal association and division logic. |
| Evaluation | official metrics plus traccuracy/py-ctcmetrics side checks | Keeps local score aligned while monitoring lineage quality. |
| Visualization | napari and inTRACKtive | Required for failure-mode inspection. |

## Stage 1 - Official Baseline Lock

| Action | Source |
|---|---|
| Clone/read official competition repo and use its loader/evaluator. | https://github.com/royerlab/kaggle-cell-tracking-competition |
| Validate image shape `(T, Z, Y, X)` and voxel scale. | Official repo README |
| Represent predictions with tracksdata graph objects. | https://github.com/royerlab/tracksdata |
| Evaluate with official edge/division metric before leaderboard submission. | https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md |

## Stage 2 - Detector And Segmentation Sweep

| Method | Test purpose | Source |
|---|---|---|
| DoG / band-pass | Cheap candidate-center baseline. | Public Kaggle notebooks |
| 3D U-Net | Competition-specific volumetric detector/segmenter. | Public Kaggle notebooks and official repo training route |
| Cellpose-SAM / CellposeDINO | Generalist cell/nucleus segmentation candidate. | https://cellpose.readthedocs.io/en/latest/models.html |
| CellSAM | Universal cell segmentation candidate. | https://github.com/vanvalenlab/cellsam |
| StarDist 3D | Dense nuclei and star-convex 3D detection. | https://github.com/stardist/stardist |
| nnU-Net | Strong self-configuring 3D segmentation baseline. | https://github.com/MIC-DKFZ/nnUNet |
| u-Segment3D | Consensus 3D segmentation from 2D stack outputs. | https://github.com/DanuserLab/u-segment3D |

Decision rule: keep detectors that improve matched-node recall without inflating total predicted nodes enough to damage adjusted edge Jaccard.

## Stage 3 - Linking, Gap Recovery, And Graph Cleanup

| Method | Use case | Source |
|---|---|---|
| Nearest neighbor | Debug baseline and distance sanity. | Kaggle starter notebook |
| LAP / linear assignment | Fast baseline for frame-to-frame linking. | Public Kaggle LAP discussions and notebooks |
| Trackastra | Learned association with division-aware transformer tracking. | https://github.com/weigertlab/trackastra |
| Ultrack | Joint segmentation/tracking under uncertainty and large-scale 3D+t data. | https://github.com/royerlab/ultrack |
| btrack | Bayesian multi-object tracking for crowded fields. | https://github.com/quantumjot/btrack |
| motile | Candidate-graph optimization. | https://funkelab.github.io/motile/ |
| Rule-based filters | Remove short tracks, implausible jumps, duplicates, and isolated artifacts. | Public Kaggle rule-based notebooks |
| Gap recovery | Reconnect plausible broken lineages. | Public Kaggle gap-recovery notebooks |

Decision rule: prioritize edge Jaccard and adjusted edge Jaccard. Division candidates should be added only when they do not degrade edge precision.

## Stage 4 - Division Handling

| Signal | How to use |
|---|---|
| Official division tolerance | Match splits within plus/minus one timepoint. |
| Public division-support notebooks | Inspect image-support and slot/relink ideas. |
| Rule-based split checks | Enforce one parent to two daughters, plausible daughter distances, and lineage continuity. |
| Learned division cues | Train or fine-tune after stable edge linking exists. |

Do not overfit division at the expense of edge linking; division Jaccard has a smaller coefficient in the final score.

## Stage 5 - Pretraining And Validation Data

| Dataset | Use |
|---|---|
| Zebrahub | Closest zebrafish developmental context and lineage imaging reference. |
| Cell Tracking Challenge 3D datasets | Benchmark tracking and generalization. |
| BlastoSPIM | 3D nuclear segmentation/tracking in embryo context. |
| DynamicNuclearNet | Live-cell nuclear segmentation/tracking route through DeepCell. |
| LIVECell | Large 2D cell segmentation pretraining route. |
| BBBC032 / BBBC035 | Mouse embryo and synthetic CTC-style benchmark routes. |
| IDR / OMERO | Broader public microscopy discovery. |

Pretraining should be treated as domain adaptation, not direct transfer. The Biohub data is zebrafish light-sheet 3D+t with sparse graph annotations.

## Stage 6 - Artifact Discipline

| Artifact | Store | Commit? |
|---|---|---|
| Curated source docs | Biohub markdown directory | Yes |
| Kaggle command manifest | `artifacts/kaggle_biohub_cell_tracking_<date>/manifest.json` | Maybe, after review |
| Raw Kaggle notebook metadata | Artifact directory | Maybe |
| Pulled notebook source | Artifact directory or curated notebook index | Review first |
| Notebook outputs | Artifact directory | Usually no, unless tiny and public-safe |
| Competition data | Outside repo or ignored artifact storage | No |
| Secrets | Never in repo | No |

## Failure Modes To Watch

| Failure mode | Symptom | Mitigation |
|---|---|---|
| Node flooding | Adjusted edge Jaccard drops despite many detections. | Calibrate detector thresholds and remove isolated/short tracks. |
| Broken tracks | High local detection but poor edge score. | Gap recovery, Trackastra/Ultrack, relink passes. |
| False divisions | Division candidates reduce edge precision. | Add division only after edge backbone stabilizes. |
| Axis/scale errors | Good-looking predictions fail matching. | Validate `(T, Z, Y, X)` and micron scale early. |
| Schema mistakes | Submission accepted poorly or fails. | Test `tracksdata` graph construction and official evaluator locally. |
| Leaderboard chasing | Public score improves but method becomes brittle. | Keep local splits and notebook-output manifests. |
| Country inference error | Attributing team country from names. | Use only explicit public country evidence. |

## Next Devastating Search Pass

1. Pull and diff the top 10 public notebook sources.
2. Parse all `run_stats.csv` files from selected notebook outputs.
3. Build a method-feature matrix: detector, linker, gap recovery, division handling, filtering, blending, score.
4. Track discussion updates daily for host clarifications.
5. Search public GitHub forks of notebook refs for hidden helper code and configs.
6. Search model hubs for Cellpose-SAM/CellposeDINO, CellSAM, StarDist, and Trackastra model cards.
7. Add a reproducibility manifest with command hashes and source timestamps.

## Related Biohub Documents

- [README](./README.md)
- [Devastating Double Check - 2026-07-09](./Devastating_Double_Check_2026-07-09.md)
- [Deep Source Atlas](./Deep_Source_Atlas_2026-07-09.md)
- [Kaggle Notebook and Discussion Radar](./Kaggle_Notebook_Discussion_Radar_2026-07-09.md)
- [ML and AI Model, Feature, Weight, and Calibration Atlas](./ML_AI_Model_Feature_Calibration_Atlas_2026-07-09.md)
