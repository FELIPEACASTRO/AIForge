# Kaggle Public Signal Snapshot

> Snapshot date: 2026-07-08. Source: Kaggle CLI and official Kaggle pages for Biohub - Cell Tracking During Development.

## Competition Metadata

Command:

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

## Public Leaderboard Signal

Command:

```powershell
kaggle competitions leaderboard biohub-cell-tracking-during-development --show --csv
```

The public leaderboard exposes `teamId`, `teamName`, `submissionDate`, and `score`. It does not expose a reliable country field. Therefore, leaderboard rows are useful for competitive intelligence but not sufficient for country attribution.

## Public Notebook Signal

Commands:

```powershell
$env:PYTHONIOENCODING='utf-8'
kaggle kernels list --competition biohub-cell-tracking-during-development --sort-by voteCount --page-size 200 --format csv
kaggle kernels list --competition biohub-cell-tracking-during-development --sort-by dateRun --page-size 200 --format csv
kaggle kernels list --competition biohub-cell-tracking-during-development --sort-by scoreDescending --page-size 200 --format csv
```

The public notebook list exposes `ref`, `title`, `author`, `lastRunTime`, and `totalVotes`. It does not expose a reliable country field. Names and usernames are not used for country inference.

## High-Signal Notebook Refs Without Country Assignment

| Ref | Signal |
|---|---|
| `inversion/cell-tracking-getting-started-w-nearest-neighbor` | Official-style starter and format sanity check. |
| `pilkwang/biohub-cell-tracking-data-model-eda-baseline` | OME-Zarr and `tracksdata` data-model walkthrough. |
| `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` | Learned graph plus gap recovery signal. |
| `yusuketogashi/lb897-baseline` | High-scoring public baseline signal. |
| `yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction` | U-Net plus ILP reproduction signal. |
| `seshurajup/lb-0-857-rule-based-v14` | Strong public rule-based baseline signal. |
| `xiaoleilian/biohub-cell-tracking-classical-baseline` | Classical baseline signal. |
| `xiaoleilian/biohub-cell-tracking-3d-u-net` | 3D U-Net detector-model signal. |
| `pilkwang/biohub-cell-tracking-blend-preprocessings` | Recent preprocessing/blend signal. |
| `amanatar/biohub-cell-tracking-ensemble` | Public ensemble signal. |

## Country Attribution Decision

No country claim is assigned from this Kaggle CLI data alone. Future D2 assignments require public profile/location or institution-level corroboration.
