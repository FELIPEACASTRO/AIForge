# Biohub Cell Tracking — Competition Overview & Scoring

> Everything you need to understand the Kaggle **[Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)** competition: the task, the real zebrafish light-sheet data, the exact evaluation metric, and a scoring strategy derived directly from that metric. All links verified live (July 2026).

## The competition

Hosted by the **Chan Zuckerberg Biohub San Francisco** ([Biohub Kaggle profile](https://www.kaggle.com/organizations/biohub)), the challenge asks you to **detect and track cells in 3D over time** in real light-sheet microscopy recordings of **zebrafish embryos** produced by the **Royer Group**. It builds directly on the group's **Ultrack** algorithm ([Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0)) and aims to establish community standards for evaluating cell tracking.

- **Competition page:** https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- **Official code & metric repository:** [github.com/royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition)

> Always confirm the current **timeline, prize, and rules on the official competition page** — those change and the page is authoritative.

## The data

- **Images:** stored as **OME-Zarr**, dimensions ordered **(T, Z, Y, X)**. Anisotropic voxel size — Z is coarser than X/Y (repository-documented spacing on the order of ~1.6 µm axial vs ~0.4 µm lateral). Treat Z anisotropy explicitly in any 3D model.
- **Ground truth / submission:** a **sparse graph** (GEFF / `tracksdata` format): **nodes** = cell-center detections (t, z, y, x), **edges** = temporal links between consecutive frames, and **cell divisions** encoded as one parent node connecting to two daughter nodes.
- **Submission object:** a `tracksdata.InMemoryGraph` (nodes + temporal edges), per the official repo.

## The evaluation metric (decoded)

Source of truth: **[royerlab/kaggle-cell-tracking-competition/metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)**. The competition uses a **custom, Jaccard-based** score:

```
score = adjusted_edge_jaccard  +  0.1 · division_jaccard
```

**1. Edge Jaccard (weight 1.0 — dominates the score).**
- Predicted nodes are matched to ground-truth nodes within a **maximum distance of 7 µm**.
- **Edges** (parent→child temporal links) are then scored: `TP / (TP + FP + FN)`.
- **Over-detection penalty (adjusted variant):**
  `max(0, jaccard · (1 − 0.1 · (T_pred − T_true) / T_true))`
  where `T_pred` / `T_true` are predicted vs. estimated true node counts — predicting far **too many nodes cuts your score**.

**2. Division Jaccard (weight 0.1 — small bonus).**
- Divisions are scored with a **±1 timepoint tolerance**. A division is a true positive when both daughter lineages are matched within a connected component.

**3. Aggregation.** **Micro-averaged** — counts are summed across all videos before computing the Jaccard, so **longer/denser videos weigh more**.

### Key thresholds

| Parameter | Value |
|---|---|
| Node matching distance | **7 µm** maximum |
| Division timepoint tolerance | **±1 timepoint** |
| Node over-prediction penalty coefficient | **0.1** |
| Division weight in final score | **0.1** |

## Scoring strategy (what actually moves the leaderboard)

1. **Edges are 10× more valuable than divisions.** Spend your effort on **accurate detection (within 7 µm) + correct temporal linking**. Do not over-invest in divisions.
2. **Control false-positive detections.** The over-detection penalty means precision matters — calibrate your detector threshold so `T_pred` stays close to `T_true`; don't flood the graph with spurious nodes.
3. **Robust linking wins.** Marginal gains come from better edge linking — multi-hypothesis tracking and **gap recovery** (re-connecting broken tracks across frames within 7 µm).
4. **Exploit the 7 µm tolerance.** Sub-voxel localization beyond ~7 µm precision yields no reward; focus compute on recall/linking instead.
5. **Divisions are a cheap top-up.** A light division detector (±1 frame) only needs to scrape the +0.1 term — never at the expense of edge precision.
6. **Evaluate locally.** Replicate the metric offline with [traccuracy](https://traccuracy.readthedocs.io/) so you iterate without burning daily submissions.

## Related

- [Segmentation & Tracking Tools](./Segmentation_and_Tracking_Tools.md) · [Datasets, Metrics & Notebooks](./Datasets_Metrics_and_Notebooks.md) · [Playbook README](./README.md)
- Parent: [`../`](../) (Kaggle) · Science context: [`../../../../15_Science_AI/`](../../../../15_Science_AI/)

**Sources:** [Kaggle competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) · [Biohub Kaggle profile](https://www.kaggle.com/organizations/biohub) · [official repo](https://github.com/royerlab/kaggle-cell-tracking-competition) · [metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md) · [Ultrack — Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0)

**Keywords:** Biohub cell tracking Kaggle, cell tracking during development, zebrafish light-sheet, 3D cell tracking competition, edge Jaccard, division Jaccard, tracksdata, OME-Zarr, Ultrack, cell tracking metric, leaderboard strategy, CZ Biohub, Royer Group, embryo cell tracking, 7 micron matching.
