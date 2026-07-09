# Kaggle Notebook And Discussion Radar - 2026-07-09

> Live Kaggle CLI snapshot for `biohub-cell-tracking-during-development`. This radar tracks public notebooks, outputs, discussion risks, and reproducibility signals without inferring private attributes such as country or identity.

## Read-Only Commands Used

```powershell
$env:PYTHONIOENCODING='utf-8'
kaggle competitions list -s "Biohub Cell Tracking" --format json
kaggle competitions leaderboard biohub-cell-tracking-during-development --show --format json
kaggle kernels list --competition biohub-cell-tracking-during-development --page-size 25 --sort-by voteCount --format json
kaggle kernels list --competition biohub-cell-tracking-during-development --page-size 25 --sort-by dateRun --format json
kaggle kernels list --competition biohub-cell-tracking-during-development --page-size 25 --sort-by scoreDescending --format json
kaggle competitions topics list biohub-cell-tracking-during-development --sort-by recent --format json
kaggle competitions topics show biohub-cell-tracking-during-development/723921 --format json
kaggle competitions topics show biohub-cell-tracking-during-development/717109 --format json
kaggle kernels files pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery --format json
kaggle kernels files yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction --format json
kaggle kernels files yusuketogashi/lb897-baseline --format json
kaggle kernels pull pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery -m
kaggle kernels pull yusuketogashi/lb897-baseline -m
kaggle kernels pull yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction -m
kaggle datasets files pilkwang/biohub-deepcenter-unet3d-center-prior-v1
kaggle datasets files subinium/biohub-v34-retrain-weights-mirror
kaggle datasets files subinium/biohub-trackastra-public-weights-mirror
kaggle datasets files drkongvis/biohub-v4-3dunet-pretrained-weights
```

## Competition Snapshot

| Field | Value |
|---|---|
| Competition | Biohub - Cell Tracking During Development |
| Ref | `biohub-cell-tracking-during-development` |
| Deadline | 2026-09-29 23:59 UTC |
| Category | Research |
| Reward | 60,000 USD |
| Teams | 951 |
| User entered | true |
| Snapshot date | 2026-07-09 |

## Public Leaderboard Frontier

| Rank in visible page | Team | Score | Last submission observed |
|---|---|---|---|
| 1 | Rahul Parmeshwar | 0.910 | 2026-07-07 23:11 UTC |
| 2 | doheon114 | 0.906 | 2026-07-06 16:13 UTC |
| 3 | Suuuiiii4 | 0.905 | 2026-07-07 05:44 UTC |
| 4 | Avner Gomes | 0.905 | 2026-07-08 13:11 UTC |
| 5 | Gavedu AI | 0.904 | 2026-07-08 15:58 UTC |
| 6 | Nikolenko_Sergei | 0.902 | 2026-07-08 00:55 UTC |
| 7 | Weatherhead | 0.901 | 2026-07-08 22:53 UTC |
| 8 | TWEAK | 0.901 | 2026-07-08 18:18 UTC |
| 9 | hilbate | 0.901 | 2026-07-08 12:29 UTC |
| 10 | Vibes and Cells Trade-Off | 0.900 | 2026-07-08 16:34 UTC |

Interpretation: the visible top page is compressed. A one-point strategy based on a single detector is unlikely to be enough; graph cleanup, gap recovery, split handling, and postprocessing matter.

## Top Notebook Signals By Votes

| Notebook ref | Votes | Signal |
|---|---:|---|
| `inversion/cell-tracking-getting-started-w-nearest-neighbor` | 231 | Starter and sanity baseline. |
| `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` | 146 | Learned graph and gap recovery. |
| `yusuketogashi/lb897-baseline` | 136 | Strong public baseline around 0.897. |
| `pilkwang/biohub-cell-tracking-data-model-eda-baseline` | 129 | Data model, EDA, and baseline route. |
| `yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction` | 109 | U-Net plus ILP reproduction. |
| `seshurajup/lb-0-857-rule-based-v14` | 91 | Competitive rule-based graph logic. |
| `romanrozen/strong-start-dog-band-pass-lb-0-73` | 78 | DoG band-pass detector baseline. |
| `lucifer19/biohub-cell-lineage-tracker` | 76 | Lineage tracker framing. |
| `boristown/agi-biohub-cell-tracking` | 73 | Public fork/candidate with high engagement. |
| `xiaoleilian/biohub-cell-tracking-classical-baseline` | 63 | Classical baseline. |
| `xiaoleilian/biohub-cell-tracking-3d-u-net` | 47 | 3D U-Net path. |
| `pilkwang/biohub-cell-tracking-blend-preprocessings` | 43 | Active blend-preprocessing path. |

## Recent Notebook Signals

| Notebook ref | Last run | Signal |
|---|---|---|
| `pyossdev/biohub-cell-tracking-baseline` | 2026-07-09 02:04 UTC | Fresh baseline; monitor for useful simplification. |
| `pilkwang/biohub-cell-tracking-blend-preprocessings` | 2026-07-08 22:44 UTC | Active preprocessing blend. |
| `pedroapalaciosz/biohub-cell-tracking-complete-journey` | 2026-07-08 19:04 UTC | End-to-end explanatory notebook. |
| `beicicc/biohub-exp043-yusuke-dataset-mintrack-recall` | 2026-07-08 10:06 UTC | Min-track recall experiment. |
| `beicicc/biohub-exp042-vmerckle-div-image-support` | 2026-07-08 09:33 UTC | Division image-support experiment. |
| `llccqq624/biohub-cell-tracking-deepcenter-blend` | 2026-07-08 05:28 UTC | Deep-center blend. |
| `vmerckle/biohub-cand-gap-image-anchor-reuse-0708002923` | 2026-07-08 00:29 UTC | Gap anchor reuse. |
| `dnicholson/biohub-exp041-vmerckle-relink-slot-v1` | 2026-07-07 21:32 UTC | Relink slot candidate. |
| `amanatar/biohub-cell-tracking-ensemble` | 2026-07-07 20:05 UTC | Ensemble candidate. |

## Notebook Output Structure

For the three inspected high-value notebooks, the public file list showed a consistent output pattern:

| Output family | Meaning |
|---|---|
| `run_stats.csv` | Run-level summary and diagnostics. |
| `submission.csv` | Kaggle submission artifact. |
| `kaggle_test_splits_50ep.json` | Split metadata used by the notebook. |
| GEFF/Zarr paths under output | Predicted tracking graphs and node/edge properties. |

Reproducibility implication: parse `run_stats.csv`, notebook metadata, and graph output paths before downloading full outputs. This can reveal method families and postprocessing without pulling large artifact trees.

## Notebook Code Pull Audit

Selected public notebooks were pulled into a temporary local audit directory for source inspection. Raw notebooks and large outputs were not committed into AIForge.

| Notebook ref | Verified implementation signal |
|---|---|
| `yusuketogashi/lb897-baseline` | U-Net/transformer graph route, ILP option, physical-distance postprocessing, motion relink, gap close, short-track filtering, safe divisions, line-fit smoothing. |
| `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` | Learned graph, gap recovery, and repair pattern; worth diffing against LB897-style configs. |
| `yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction` | 3D U-Net plus ILP reproduction route. |
| `pilkwang/biohub-cell-tracking-blend-preprocessings` | Preprocessing blend path for detector/graph improvements. |

Concrete constants extracted from the inspected LB897-family code include `DET_THRESHOLD=0.99`, `OUTPUT_EDGE_MAX_UM=14.0`, motion relink gates of `6.0` and `10.0` microns, one-frame gap close at `6.0` microns, short-track filtering around lengths `6` to `7`, safe-division parent/daughter gates near `4.7` to `7.8` microns, and line-fit smoothing weight `0.8`.

## Discussion Radar

| Topic id | Title | Votes | Comments | Interpretation |
|---|---|---:|---:|---|
| 714101 | How to get started plus Competition Official Discord | 11 | 1 | Official/community onboarding route. |
| 716062 | Welcome to the Biohub - Cell Tracking During Development Challenge | 43 | 3 | Host welcome thread; monitor for rules and clarifications. |
| 717109 | Resource sharing for cell tracking challenge | 19 | 2 | Resource-sharing thread; CLI body includes a comment about division learning signal. |
| 716952 | Rule-based is surprisingly strong? | 31 | 7 | Important evidence that graph logic and heuristics can score well. |
| 722668 | good visualisation of the task | 17 | 1 | Visualization route for understanding failure modes. |
| 723921 | Ground truth for all 4 test clips appears to be present in the train split | 1 | 1 | Critical risk-monitoring title; CLI did not return useful body text. Needs host/community verification. |
| 723696 | what does the node_id in submission file represent? | 0 | 1 | Submission-schema clarification route. |
| 723898 | Scaling LAP Tracking Baseline | 0 | 0 | LAP baseline scaling route. |

## Strategic Interpretation

1. The frontier is postprocessing-heavy. Public notebook titles point to min-track filtering, gap closure, relinking, division slots, image-anchor reuse, and blend preprocessing.
2. Strong baseline work should be organized as a graph pipeline, not just a detector pipeline.
3. Division detection is still worth monitoring, but the metric weight makes edge/link quality the first-order objective.
4. Official host clarifications and public discussion titles must be monitored because the competition appears to be evolving quickly.
5. The Kaggle CLI/API does not expose reliable country evidence in leaderboard or kernel list output.

## Next Kaggle Automation Steps

| Step | Command family | Output to capture |
|---|---|---|
| Pull selected notebook code | `kaggle kernels pull <ref> -m` | Notebook source plus metadata. |
| Inspect notebook outputs | `kaggle kernels output <ref> --page-size 200` | `run_stats.csv`, graph outputs, split JSON. |
| Download leaderboard CSV | `kaggle competitions leaderboard --download` | Full leaderboard snapshot if fields justify it. |
| Export discussions | `kaggle competitions topics show` | Host clarifications and resource links. |
| Create manifest | Local script | command, timestamp, Kaggle CLI version, output path, source hash. |

## Related Biohub Documents

- [README](./README.md)
- [Devastating Double Check - 2026-07-09](./Devastating_Double_Check_2026-07-09.md)
- [Deep Source Atlas](./Deep_Source_Atlas_2026-07-09.md)
- [ML and AI Model, Feature, Weight, and Calibration Atlas](./ML_AI_Model_Feature_Calibration_Atlas_2026-07-09.md)
- [Reproducibility and Model Roadmap](./Reproducibility_Model_Roadmap_2026-07-09.md)
