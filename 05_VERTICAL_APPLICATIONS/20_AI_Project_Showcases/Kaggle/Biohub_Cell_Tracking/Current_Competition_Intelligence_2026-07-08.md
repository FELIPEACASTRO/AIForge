# Biohub Cell Tracking - Current Competition Intelligence

> Live search and Kaggle CLI snapshot for Biohub - Cell Tracking During Development. Captured on 2026-07-08 after pulling `origin/master`. Because Kaggle competitions change quickly, treat scores, team counts, and public notebooks as timestamped intelligence, not permanent facts.

## Kaggle CLI Competition Snapshot

Command used:

```powershell
kaggle competitions list -s "Biohub Cell Tracking"
```

| Field | Snapshot value |
|---|---|
| Competition ref | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development |
| Deadline | 2026-09-29 23:59:00 |
| Category | Research |
| Reward | 60,000 USD |
| Team count | 950 |
| Local account entered | True |

## Leaderboard Snapshot

Command used:

```powershell
kaggle competitions leaderboard biohub-cell-tracking-during-development --show --csv
```

| Rank in snapshot | Team | Submission date | Public score |
|---:|---|---|---:|
| 1 | Rahul Parmeshwar | 2026-07-07 23:11:12.513000 | 0.910 |
| 2 | doheon114 | 2026-07-06 16:13:07.240000 | 0.906 |
| 3 | Suuuiiii4 | 2026-07-07 05:44:45.383000 | 0.905 |
| 4 | Avner Gomes | 2026-07-08 13:11:43.113000 | 0.905 |
| 5 | Gavedu AI | 2026-07-08 15:58:42.690000 | 0.904 |
| 6 | Nikolenko_Sergei | 2026-07-08 00:55:53.483000 | 0.902 |
| 7 | Weatherhead | 2026-07-08 22:53:26.320000 | 0.901 |
| 8 | TWEAK | 2026-07-08 18:18:17.523000 | 0.901 |
| 9 | hilbate | 2026-07-08 12:29:55.366000 | 0.901 |
| 10 | Vibes & Cells Trade-Off | 2026-07-08 16:34:13.146000 | 0.900 |
| 11 | ChenWenSheng | 2026-07-08 07:01:39.853000 | 0.900 |

Practical read: the public frontier is already around 0.90 to 0.91. Any local pipeline below that needs a specific reason to be competitive, such as a stronger detection threshold policy, edge ensemble, gap repair, division recovery, or leakage-safe public-notebook adaptation.

## Top Public Notebooks By Votes

Command used:

```powershell
kaggle kernels list --competition biohub-cell-tracking-during-development --sort-by voteCount --page-size 20
```

| Ref | Title | Author | Last run | Votes | Notes |
|---|---|---|---|---:|---|
| `inversion/cell-tracking-getting-started-w-nearest-neighbor` | Cell Tracking Getting Started w/ Nearest Neighbor | inversion | 2026-06-23 21:23:55 | 230 | Official-style starter; useful for format sanity checks. |
| `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` | Biohub Cell Tracking: Learned Graph w Gap Recovery | Pilkwang Kim | 2026-07-06 20:57:16 | 146 | Important public signal: learned linking plus one-frame gap repair. |
| `yusuketogashi/lb897-baseline` | LB897 BaseLine | Yusuke Togashi | 2026-07-08 02:16:49 | 136 | High-scoring public baseline; opened from search as redirected notebook. |
| `pilkwang/biohub-cell-tracking-data-model-eda-baseline` | Biohub Cell Tracking: Data Model, EDA, Baseline | Pilkwang Kim | 2026-07-01 09:15:49 | 129 | Strong for understanding OME-Zarr plus `tracksdata`. |
| `yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction` | Biohub Cell Tracking V4 UNet ILP Reproduction | Yaroslav kholmirzayev | 2026-07-07 13:26:11 | 108 | U-Net plus ILP reproduction lead. |
| `seshurajup/lb-0-857-rule-based-v14` | LB 0.857 Rule-Based V14 | SeshuRaju | 2026-07-03 23:09:15 | 91 | Strong non-heavy public baseline signal. |
| `romanrozen/strong-start-dog-band-pass-lb-0-73` | STRONG START: DoG BAND-PASS | Roman Rozen | 2026-06-30 18:58:02 | 78 | Classical detection baseline family. |
| `lucifer19/biohub-cell-lineage-tracker` | Biohub Cell Lineage Tracker | Krizso Gergely | 2026-07-05 07:27:55 | 76 | Interactive lineage-tracker approach. |
| `boristown/agi-biohub-cell-tracking` | AGI Biohub Cell Tracking | AGI | 2026-07-05 09:36:52 | 73 | Public score title/search snippet reported near 0.861. |
| `xiaoleilian/biohub-cell-tracking-classical-baseline` | Biohub Cell Tracking - Classical Baseline | Xiaolei Lian | 2026-07-02 07:26:31 | 63 | Single-file/no-GPU/no-internet style baseline. |
| `isakatsuyoshi/biohub-rule-based-baseline` | Biohub Rule-Based Baseline | ISAKA Tsuyoshi | 2026-07-01 07:41:30 | 63 | Plain nearest-neighbor/rule-based reference. |
| `xiaoleilian/biohub-cell-tracking-3d-u-net` | Biohub Cell Tracking - 3D U-Net | Xiaolei Lian | 2026-07-03 06:54:08 | 47 | Detector-model lead. |
| `pilkwang/biohub-cell-tracking-blend-preprocessings` | Biohub Cell Tracking: Blend Preprocessings | Pilkwang Kim | 2026-07-08 22:44:01 | 43 | Very recent preprocessing/blend lead. |
| `thibautgoldsborough/unet-baseline-inference-submission` | UNet baseline - inference & submission | Thibaut Goldsborough | 2026-06-30 02:53:23 | 25 | Related to official repo's baseline note. |
| `amanatar/biohub-0-855-ema-intensity-cost-tracking` | Biohub 0.855: EMA + Intensity Cost Tracking | Aman Atar | 2026-07-05 19:27:04 | 16 | Intensity-cost tracking lead. |

## Most Recent Public Notebook Activity

Command used:

```powershell
kaggle kernels list --competition biohub-cell-tracking-during-development --sort-by dateRun --page-size 30
```

| Ref | Title | Author | Last run | Votes | Why watch |
|---|---|---|---|---:|---|
| `pilkwang/biohub-cell-tracking-blend-preprocessings` | Biohub Cell Tracking: Blend Preprocessings | Pilkwang Kim | 2026-07-08 22:44:01 | 43 | High-signal author, current preprocessing blend. |
| `nikitagajbhiye30/biohub-00` | Biohub 00 | Nikita | 2026-07-08 20:03:27 | 8 | Recent candidate. |
| `pedroapalaciosz/biohub-cell-tracking-complete-journey` | Biohub Cell Tracking Complete Journey | Pedro A. Palacios Z. | 2026-07-08 19:04:36 | 2 | Broad explanatory notebook. |
| `tamerlanomralinov/biohub-cell-tracking-learned-graph-w-gap-recovery` | Learned Graph w Gap Recovery | Tamerlan Omralinov | 2026-07-08 11:22:37 | 11 | Fork/variant of important gap-recovery notebook. |
| `beicicc/biohub-exp043-yusuke-dataset-mintrack-recall` | Biohub Exp043 Yusuke Dataset Mintrack Recall | Kun Zhang | 2026-07-08 10:06:19 | 2 | Experiments around Yusuke baseline and minimum-track recall. |
| `beicicc/biohub-exp042-vmerckle-div-image-support` | Biohub Exp042 Vmerckle Div Image Support | Kun Zhang | 2026-07-08 09:33:46 | 1 | Division-support signal. |
| `llccqq624/biohub-cell-tracking-deepcenter-blend` | biohub-cell-tracking-deepcenter-blend | Jiachen Li | 2026-07-08 05:28:16 | 2 | Deep-center blending lead. |
| `vmerckle/biohub-cand-gap-image-anchor-reuse-0708002923` | gap image anchor reuse | Victor Merckle | 2026-07-08 00:29:29 | 1 | Gap-recovery candidate. |
| `amanatar/biohub-cell-tracking-ensemble` | Biohub Cell Tracking Ensemble | Aman Atar | 2026-07-07 20:05:34 | 6 | Ensemble lead. |
| `yaroslavkholmirzayev/high-upside-min7-short-track-filter-risk-a-b` | High-Upside Min7 Short-Track Filter | Yaroslav kholmirzayev | 2026-07-07 16:11:46 | 10 | Short-track filtering risk/reward lead. |

## Public Strategy Signals

| Signal | Evidence | Implication |
|---|---|---|
| Edge quality dominates. | Official metric weights edge Jaccard at 1.0 and division Jaccard at 0.1. | Prioritize detection thresholding, linking precision, gap recovery, and edge pruning before division engineering. |
| Over-detection is dangerous. | Official metric applies an adjusted edge Jaccard penalty when predicted node count exceeds estimated true count. | Threshold calibration and short-track pruning can matter as much as detector recall. |
| Gap recovery is a public baseline theme. | Pilkwang's high-vote learned-graph notebook and many recent notebooks mention gap repair/relinking. | Build a local ablation harness for one-frame and short-gap reconnects under the 7 micrometer tolerance. |
| U-Net plus ILP is a strong public direction. | Official repo baseline is a temporal 3D U-Net plus cross-attention transformer; public notebooks reference U-Net/ILP and LB897. | Compare pure geometric linking, learned graph linking, Trackastra, and ILP variants on a common local evaluator. |
| Classical baselines are still meaningful. | Rule-based, DoG band-pass, nearest-neighbor, and classical baseline notebooks have strong vote counts and usable scores. | Keep a transparent classical baseline as a debugging oracle and ensemble member. |
| Ensembling is emerging quickly. | Recent notebooks include preprocessing blends, deep-center blends, and ensemble variants. | Store edge lists and confidence scores in an intermediate format so multiple linkers can be merged or voted. |

## Recommended Next Searches

1. Pull notebook metadata repeatedly with the Kaggle CLI and watch deltas in `dateRun`, title score hints, and votes.
2. Download the top public notebooks into a separate audited package before adapting any code.
3. Inspect official repo updates for changes to `metrics.md`, `src/tracking_cellmot/metrics`, and data spec files.
4. Track GitHub issues/discussions in `royerlab/kaggle-cell-tracking-competition` for scoring or format clarifications.
5. Compare public notebook claims against the official leaderboard whenever possible; notebook titles can lag or overstate scores.

## Useful Commands

```powershell
kaggle competitions list -s "Biohub Cell Tracking"
kaggle competitions leaderboard biohub-cell-tracking-during-development --show --csv
kaggle kernels list --competition biohub-cell-tracking-during-development --sort-by voteCount --page-size 20
kaggle kernels list --competition biohub-cell-tracking-during-development --sort-by dateRun --page-size 30
```

## Related Files

- [Official Source Map](./Official_Source_Map_2026-07-08.md)
- [Competition Overview and Scoring](./Competition_Overview_and_Scoring.md)
- [Datasets, Metrics and Notebooks](./Datasets_Metrics_and_Notebooks.md)
- [Segmentation and Tracking Tools](./Segmentation_and_Tracking_Tools.md)
- [Tracking Methods Compendium](./Tracking_Methods_Compendium.md)
