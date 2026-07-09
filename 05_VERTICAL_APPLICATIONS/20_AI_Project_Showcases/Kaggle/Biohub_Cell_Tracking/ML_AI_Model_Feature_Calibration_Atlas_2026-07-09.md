# ML and AI Model, Feature, Weight, and Calibration Atlas - Biohub Cell Tracking

> Technical inventory of models, weights, algorithms, features, datasets, calibration knobs, and ML/AI tactics relevant to `Biohub - Cell Tracking During Development`. This is a competition-oriented map, not a generic bibliography.

## Relevance Grades

| Grade | Meaning |
|---|---|
| A | Directly actionable for Biohub cell tracking or official scoring. |
| B | Highly relevant after adaptation to 3D+t zebrafish microscopy. |
| C | Adjacent ML/AI idea worth monitoring or mining for tactics. |

## Model And Weight Routes

| Grade | Model or weight route | What it gives | Calibration notes | Source |
|---|---|---|---|---|
| A | Official Royer U-Net/transformer baseline | Competition-native training, prediction, evaluation, and visualization route. | Train longer than starter settings; tune detection threshold against adjusted edge Jaccard. | https://github.com/royerlab/kaggle-cell-tracking-competition |
| A | Kaggle public `V4 UNet ILP Reproduction` | Public U-Net plus ILP-style reproduction path. | Compare its output `run_stats.csv`, split JSON, and GEFF/Zarr graph outputs. | https://www.kaggle.com/code/yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction |
| A | Kaggle public `LB897 Baseline` | Strong public baseline around the 0.897 score region. | Mine thresholds, short-track filters, and submission schema. | https://www.kaggle.com/code/yusuketogashi/lb897-baseline |
| A | Kaggle public `Learned Graph w Gap Recovery` | Learned graph, gap recovery, and repair strategy. | Tune gap additions; watch for over-repair that creates false edges. | https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery |
| A | Trackastra pretrained models | Transformer-based association for segmented cells; includes general models and SAM2-feature variant. | Start with `general_2d`; test `general_2d_w_SAM2_features` where features help, but validate on Biohub geometry. | https://github.com/weigertlab/trackastra |
| A | Cellpose-SAM weights | Generalist cellular segmentation weights; HF repo includes `cpsam` and related weights. | Calibrate diameter, thresholding, tiling, and 3D handling; local and HF Space outputs may differ. | https://huggingface.co/mouseland/cellpose-sam |
| A | Cellpose built-in 2026 models | `cpsam_v2`, `cpdino`, `cpdino-vitb`, and `cpsam`. | `cpsam_v2` fixes low-contrast training; DINO models use 384x384 tile size by default. | https://cellpose.readthedocs.io/en/latest/models.html |
| B | Cellpose cpsam ONNX | ONNX export of Cellpose-SAM for deployment experiments. | Validate parity against official Cellpose Python output before using for submissions. | https://huggingface.co/keejkrej/cellpose-cpsam-onnx |
| B | Classpose | Fine-tuned Cellpose-SAM route for cell phenotyping/segmentation. | Treat as domain-adjacent; test only if Biohub cells resemble training distribution. | https://huggingface.co/classpose/classpose |
| B | CellSAM | Universal cell segmentation model with CellFinder detector and SAM prompting. | Check weight access and 3D limitations; may be useful for 2D slice proposals. | https://github.com/vanvalenlab/cellsam |
| B | CellSAM app | Deployed CellSAM inference route. | Use for qualitative sanity checks, not production-scale competition inference. | https://cellsam.deepcell.org/ |
| B | StarDist 3D | Star-convex 3D nuclei/cell object detection. | Best for compact nuclei-like objects; tune normalization and tile size. | https://github.com/stardist/stardist |
| B | nnU-Net | Self-configuring 3D segmentation baseline. | Requires careful patch size, anisotropic spacing, validation splits, and postprocessing. | https://github.com/MIC-DKFZ/nnUNet |
| B | u-Segment3D | 3D consensus segmentation from 2D stack outputs. | Useful for converting strong 2D slice predictions into 3D objects. | https://github.com/DanuserLab/u-segment3D |
| C | SAM2 / micro-sam | Interactive 2D/3D segmentation and tracking route using SAM-style models. | Use for annotation, pseudo-labeling, or QA, not blindly for final tracking. | https://computational-cell-analytics.github.io/micro-sam/micro_sam.html |
| C | MedSAM2 | 3D medical/video segmentation foundation-model route. | Adjacent; useful for architecture ideas and volumetric prompting, not directly trained for zebrafish cells. | https://github.com/bowang-lab/MedSAM |

## Verified Public Weight Packs - 2026-07-09

These entries were upgraded from search leads to file-list evidence with read-only Kaggle dataset-file queries.

| Grade | Source | Verified files | Practical interpretation |
|---|---|---|---|
| A | https://www.kaggle.com/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1 | `weights/full_frame_center/best.pt`, `checkpoint_last.pt`, `config.json`, manifests, gate metrics, and source training scripts. | Direct full-frame 3D center detector candidate. |
| A | https://www.kaggle.com/datasets/subinium/biohub-v34-retrain-weights-mirror | `unet_transformer/split_0/edge_predictor_best.pth`, `config.json`. | Direct edge-predictor / U-Net-transformer route. |
| A | https://www.kaggle.com/datasets/subinium/biohub-trackastra-public-weights-mirror | `ctc/model.pt`, `general_2d/model.pt`, train configs, Trackastra wheel, GEFF/Zarr dependency wheels. | Offline Trackastra-style learned association route for Kaggle notebooks. |
| A | https://www.kaggle.com/datasets/drkongvis/biohub-v4-3dunet-pretrained-weights | `unet3d.pt`, `unet3d_pretrained.pt`. | Compact 3D U-Net detector/pretraining pack. |
| B | https://www.kaggle.com/datasets/justinkim1216/biohub-nnunet-center-support-v1 | File-list inspection still pending. | Likely center-support nnU-Net candidate. |
| B | https://www.kaggle.com/datasets/justinkim1216/biohub-nnunet-flow-support-v1 | File-list inspection still pending. | Possible motion/flow-support candidate. |
| C/B | https://www.kaggle.com/datasets/mdaliazad/biohub-cell-tracking-model | File-list inspection still pending. | Direct but under-described model pack; inspect architecture first. |

## Verified Hugging Face Leads - 2026-07-09

These entries were checked through public Hugging Face metadata. No private HF token was required.

| Grade | Source | Metadata signal | Competition use |
|---|---|---|---|
| A | https://huggingface.co/mouseland/cellpose-sam | BSD-3-Clause; Cellpose-SAM model repo, updated 2026-05-17. | Main generalist cell detector route. |
| B | https://huggingface.co/SDu90/zebrafish-cellpose-finetunes | Cellpose library, zebrafish tags, Cellpose-SAM base, CC-BY-NC-4.0. | Closest zebrafish Cellpose transfer lead, but likely 2D/slice. |
| B | https://huggingface.co/keejkrej/cellpose-cpsam-onnx | ONNX, Cellpose, microscopy, BSD-3-Clause. | Deployment/parity test route. |
| B | https://huggingface.co/DnaRnaProteins/sam2-cells-seg | SAM2 tiny fine-tune, cell segmentation, Apache-2.0. | Prompt-oriented cell segmentation lead. |
| B | https://huggingface.co/keejkrej/cellsam-onnx | ONNX CellSAM route, license marked `other`. | Potential CellSAM deployment path; check license/provenance first. |
| B | https://huggingface.co/KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari | 3D StarDist-like Xenopus nuclei, BSD-3-Clause. | Embryo-ish 3D nuclei proposal route with domain shift. |
| B | https://huggingface.co/KapoorLabs-Copenhagen/xenopus-stardist3d-membrane-mari | 3D membrane model, BSD-3-Clause. | 3D membrane/cell proposal route with domain shift. |
| C | https://huggingface.co/wanglab/MedSAM2 | Medical SAM2, CC-BY-SA-4.0. | Volumetric prompting ideas only. |

## Algorithms And Techniques

| Grade | Technique | Why it matters | Tuning knobs |
|---|---|---|---|
| A | Tracking-by-detection | Core paradigm: detect cells per frame, then link. | Detector threshold, NMS radius, max displacement, edge cost. |
| A | Linear assignment problem (LAP) | Fast frame-to-frame linking baseline. | Distance cost, max-link distance, gap penalty, split penalty. |
| A | Integer linear programming (ILP) | Global graph optimization for tracks, gaps, and divisions. | Birth/death cost, transition cost, division cost, false positive/negative cost. |
| A | Gap recovery | Repairs missed detections and broken tracklets. | Max gap length, interpolation cost, intensity/position compatibility. |
| A | Short-track filtering | Removes isolated false positives that inflate node count. | Minimum track length, track confidence, edge support. |
| A | Division slot/relinking | Adds or repairs one-parent two-child events. | Division window, daughter distance, daughter angle, lineage continuity. |
| A | Blend preprocessing | Combines contrast normalization, denoising, band-pass, and detector variants. | Blend weights, threshold per preprocessing channel, per-embryo calibration. |
| A | Candidate-center detection | Reduces segmentation to centroid graph problem. | DoG/LoG sigma, peak threshold, min distance, z/y/x scale. |
| A | Multi-hypothesis segmentation | Keeps several candidate objects before temporal selection. | Segmentation hierarchy, candidate pruning, solver constraints. |
| B | Transformer association | Learns pairwise links in a temporal window. | Window size, embedding features, greedy vs ILP linking, division handling. |
| B | Siamese/re-identification features | Adds appearance similarity to linking. | Crop size, embedding model, feature normalization, temporal sampling. |
| B | Optical flow / motion priors | Helps propose likely next positions. | Flow model, smoothness, scale, confidence threshold. |
| B | Uncertainty-aware tracking | Estimates track/error confidence. | Calibrated probabilities, confidence thresholds, manual-review routing. |
| B | Pseudo-labeling | Uses public baselines or model consensus to train detectors. | Confidence threshold, augmentation, label cleanup, leakage controls. |
| C | Diffusion/virtual staining | Can enhance or synthesize channels/features. | Use only for feature engineering experiments; validate against metric. |

## Feature Engineering Targets

| Feature family | Examples | Where used |
|---|---|---|
| Spatial centroid features | `z`, `y`, `x`, micron-scaled distance, anisotropic distance | Node matching, edge cost, division geometry |
| Temporal features | time index, gap length, velocity, acceleration, displacement rank | Linking, gap recovery, short-track filtering |
| Intensity features | local mean, max, percentile, contrast, background-subtracted intensity | Detection scoring, link compatibility, division support |
| Shape features | volume, radius, elongation, eccentricity, bounding box, surface area | Candidate filtering, division sanity checks |
| Blob/filter features | DoG, LoG, Hessian, Laplacian, local maxima, distance transform | Candidate-center detection |
| Segmentation confidence | model probability, mask score, flow consistency, NMS score | Node score, false-positive pruning |
| Tracklet features | length, missing frames, average speed, edge confidence, start/end time | Short-track filters, relinking |
| Division features | parent size, daughter size ratio, daughter angle, split timing, daughter displacement | Division candidate scoring |
| Graph features | node degree, edge cost, connected component size, fork count, merge count | Graph cleanup and schema validation |
| Embedding features | Trackastra embeddings, CNN/Siamese crops, SAM/ViT features | Learned association and re-identification |

## Calibration Knobs

| Calibration knob | Practical range to sweep | Metric risk |
|---|---|---|
| Detection threshold | low/medium/high per preprocessing branch | Too low floods nodes; too high breaks edges. |
| Cell diameter/radius | match expected object scale in pixels and microns | Wrong scale causes duplicate detections or missed cells. |
| Z/Y/X anisotropic scale | use official `1.625, 0.40625, 0.40625` microns | Wrong scale breaks node matching and edge geometry. |
| NMS radius | radius in physical units, not raw voxels | Too small duplicates cells; too large merges neighbors. |
| Max link distance | per timepoint movement in microns | Too small fragments tracks; too large false-links. |
| Max gap length | 1, 2, 3+ frames | Too aggressive gap closing creates false edges. |
| Minimum track length | 2, 4, 6, 8+ nodes | Too high removes real short tracks; too low preserves noise. |
| Division score threshold | conservative first, then expand | False divisions can hurt edges despite small division weight. |
| Division time tolerance | align with official plus/minus one frame tolerance | Over-wide windows create fake forks. |
| ILP costs | transition, birth/death, division, FP/FN costs | Bad costs can globally reinforce wrong graph structure. |
| Blend weights | per detector/preprocessing branch | Overweighting noisy branch inflates nodes. |
| Per-embryo normalization | quantile, z-score, background subtraction | Bad normalization shifts detector calibration. |

## Notebook-Pulled Calibration Constants - 2026-07-09

High-value public notebook sources were pulled into a temporary audit directory and inspected locally. Raw notebooks and large outputs were not committed. The most concrete inspected path used a U-Net/transformer detector/linker with ILP and graph repair.

| Calibration area | Observed values | Use |
|---|---|---|
| Detector threshold | `DET_THRESHOLD = 0.99`; nearby public variants mention 0.985-0.99. | Sweep against node flooding and broken-track failures. |
| Edge cap | `OUTPUT_EDGE_MAX_UM = 14.0` | Removes physically implausible edges. |
| Motion relink | tight `6.0 um`, relaxed `10.0 um`, velocity weight `0.5`, learned bonus `0.75`. | Rebuilds a one-to-one graph from detected nodes and learned-edge hints. |
| Gap close | max gap `1`, gap distance `6.0 um`, reuse radius `3.2 um`, added-node fraction cap `0.05`, absolute cap `2000`. | Repairs one-frame misses while limiting synthetic nodes. |
| Gap2 recovery | default off; if enabled, total `10.2 um`, step `4.4 um`, link fraction cap `0.0045`, absolute cap `180`. | High-risk repair for two-missing-frame cases. |
| Short-track filtering | default `6`; high-confidence variant used global `7` with one dataset-specific `6`. | Removes isolated false positives; can also delete real short tracks. |
| Safe divisions | parent-child `4.7 um`, sister `7.2 um`, existing child `7.8 um`, frame cap `0.008`, global cap `0.004`. | Conservative second-child insertion after stable linking. |
| Line-fit smoothing | enabled, weight `0.8`, window `2`. | Reduces coordinate jitter without changing topology too aggressively. |
| ILP weights | edge `-1.0`, appearance `0.1`, disappearance `0.1`, division `1.0`. | Controls global graph topology. |
| Weight path | `weights/unet_transformer/split_0/edge_predictor_best.pth` | Connects notebook implementation to public weight-pack file listings. |

The inspected code computed physical distances as:

```text
distance_um = sqrt((1.625 * dz)^2 + (0.40625 * dy)^2 + (0.40625 * dx)^2)
```

This reinforces that every distance-like knob should be calibrated in microns, not raw voxel units.

## Dataset And Pretraining Matrix

| Grade | Dataset | Use | Source |
|---|---|---|---|
| A | Official Biohub train/test | Competition source of truth. | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data |
| A | Official Royer repo dense/local data path | Local training/evaluation scaffold. | https://github.com/royerlab/kaggle-cell-tracking-competition |
| B | Zebrahub | Closest zebrafish developmental context. | https://zebrahub.sf.czbiohub.org/ |
| B | Cell Tracking Challenge 3D datasets | Tracking benchmark and generalization validation. | https://celltrackingchallenge.net/datasets/ |
| B | BlastoSPIM | 3D nuclear embryo segmentation/tracking. | https://blastospim.flatironinstitute.org/html/ |
| B | DynamicNuclearNet | DeepCell nuclear segmentation/tracking. | https://datasets.deepcell.org/ |
| B | LIVECell | Large 2D cell segmentation pretraining. | https://www.nature.com/articles/s41592-021-01249-6 |
| B | BBBC032 | Mouse embryo blastocyst 3D nuclei. | https://bbbc.broadinstitute.org/BBBC032 |
| B | BBBC035 | Synthetic HL60 CTC-style data. | https://bbbc.broadinstitute.org/BBBC035 |
| B | TYC dataset | Yeast cell segmentation/tracking in microstructures. | https://huggingface.co/papers/2308.12116 |
| C | Multi-modality Cell Segmentation Challenge | Multi-modality segmentation and transformer baseline ideas. | https://huggingface.co/papers/2308.05864 |
| C | Low-frame-rate cell tracking dataset | Sparse temporal tracking and mitosis stress test. | https://openaccess.thecvf.com/content/CVPR2025W/CVMI/papers/Gachloo_Low-Frame-Rate_Cell_Tracking_Unmet_Needs_and_Future_Directions_CVPRW_2025_paper.pdf |
| C | IDR / OMERO | Broad microscopy source discovery. | https://idr.openmicroscopy.org/ |
| C | BioImage Model Zoo | Reusable pretrained bioimage models. | https://bioimage.io/ |

## Kaggle Feature Signals To Mine

| Public signal | Likely ML/AI component |
|---|---|
| `learned-graph-w-gap-recovery` | learned edge scoring, graph repair, gap closure |
| `lb897-baseline` | strong baseline thresholds and postprocessing |
| `v4-unet-ilp-reproduction` | 3D U-Net detector plus ILP graph optimization |
| `rule-based-v14` | handcrafted graph filters and competition-specific heuristics |
| `blend-preprocessings` | preprocessing ensemble and detector blending |
| `deepcenter-blend` | learned center proposal blending |
| `mintrack-recall` | minimum-track-length calibration |
| `div-image-support` | division support from image evidence |
| `gap-image-anchor-reuse` | image-anchored gap recovery |
| `relink-division-slot` | post-hoc division relinking |

## Experiment Queue

| Priority | Experiment | Expected insight |
|---|---|---|
| 1 | Parse top notebook `run_stats.csv` files. | Identify which methods improve edge score vs division score. |
| 1 | Reproduce official baseline locally. | Establish trusted evaluator and submission schema. |
| 1 | Sweep detection threshold vs adjusted edge Jaccard. | Find node-flooding boundary. |
| 1 | Sweep min track length and max gap length. | Quantify postprocessing tradeoff. |
| 2 | Compare DoG/LoG, Cellpose-SAM, CellposeDINO, StarDist, and 3D U-Net proposals. | Determine best detection recall/precision mix. |
| 2 | Add Trackastra linking over multiple proposal sets. | Test learned association vs LAP/ILP. |
| 2 | Add Ultrack multi-hypothesis selection. | Test robust tracking under segmentation uncertainty. |
| 2 | Add conservative division candidate scorer. | Improve division term without harming edges. |
| 3 | Test CellSAM or SAM/micro-sam slice proposals. | Evaluate foundation-model transfer. |
| 3 | Use CTC/BlastoSPIM/DynamicNuclearNet for pretraining or sanity validation. | Improve robustness outside official train split. |

## What Not To Do

- Do not infer country or affiliation from Kaggle usernames.
- Do not treat a model hub demo as equivalent to local reproducible inference.
- Do not optimize division first; the score is dominated by edge quality.
- Do not ignore anisotropic voxel scaling.
- Do not commit Kaggle credentials or raw competition data.
- Do not download huge notebook outputs into the curated markdown tree.
- Do not trust public leaderboard gains without local split and output-manifest checks.

## Related Biohub Documents

- [README](./README.md)
- [Devastating Double Check - 2026-07-09](./Devastating_Double_Check_2026-07-09.md)
- [Deep Source Atlas](./Deep_Source_Atlas_2026-07-09.md)
- [Kaggle Notebook and Discussion Radar](./Kaggle_Notebook_Discussion_Radar_2026-07-09.md)
- [Reproducibility and Model Roadmap](./Reproducibility_Model_Roadmap_2026-07-09.md)
