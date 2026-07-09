# Kaggle API and CLI Extraction Plan - Biohub Cell Tracking

> Official-documentation audit and reproducible extraction route for the Kaggle competition `biohub-cell-tracking-during-development`. This page documents what the Kaggle CLI/API can extract, what it cannot extract reliably, and how to handle credentials without leaking secrets into the repository.

## Scope

This plan covers only public or authenticated Kaggle metadata that can be accessed through official Kaggle tooling:

- competition metadata, files, pages, discussions, leaderboard, submissions, and team submissions;
- public competition notebooks, notebook files, notebook outputs, logs, and notebook discussion topics;
- official output formatting through `--format json`, `--format csv`, or projected fields;
- Python API methods exposed by the installed `kaggle` package.

It does not treat Kaggle usernames, team names, notebook author names, display names, languages, or personal names as country evidence. Country attribution must require an explicit public location or an official institutional/country source.

## Official Documentation Audited

| Source | Coverage used | URL |
|---|---|---|
| Kaggle API documentation | Official API setup and command entry point. | https://www.kaggle.com/docs/api |
| Kaggle CLI repository | Official CLI source, package identity, and docs root. | https://github.com/Kaggle/kaggle-cli |
| Kaggle CLI docs README | Command families: competitions, datasets, forums, kernels, models, benchmarks, and configuration. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md |
| Competitions CLI docs | `list`, `files`, `download`, `submit`, `submissions`, `leaderboard`, `team-submissions`, pages, topics, and topic messages. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md |
| Kernels CLI docs | `list`, `files`, `pull`, `output`, `status`, `logs`, `delete`, and kernel topics. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md |
| Forums CLI docs | Forum/topic discovery and topic inspection commands. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/forums.md |
| Output format docs | `--format json`, `--format csv`, field projections, and the relationship to legacy `--csv`. | https://github.com/Kaggle/kaggle-cli/blob/main/docs/output_format.md |

## Local Toolchain Verified

| Item | Observed value |
|---|---|
| Kaggle CLI | `Kaggle CLI 2.2.3` |
| Python package | `kaggle 2.2.3` |
| Package location | `C:\Users\davis\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages` |
| Credential file supplied by user | `C:\Users\davis\Workspace\KG-GPT\kaggle.json` |
| Secret handling | Existence and metadata can be checked, but file contents must not be printed, copied into repo docs, committed, or logged. |

## Authentication Handling

Official Kaggle authentication supports browser login, token environment variables, and token files. For this workspace, the user supplied a legacy `kaggle.json` credential file outside the AIForge repository. Safe usage rules:

1. Do not print the file contents.
2. Do not move or copy the file into this repository.
3. Do not commit raw Kaggle tokens, access tokens, cookies, or generated auth files.
4. Prefer read-only Kaggle commands until a download or submission is explicitly required.
5. When a command must use the supplied file directly, set `KAGGLE_CONFIG_DIR=C:\Users\davis\Workspace\KG-GPT` for that shell session instead of duplicating the credential.

## Confirmed Competition Snapshot

Read-only CLI command:

```powershell
$env:PYTHONIOENCODING='utf-8'
kaggle competitions list -s "Biohub Cell Tracking" --format json
```

Observed fields:

| Field | Observed value |
|---|---|
| `ref` | `https://www.kaggle.com/competitions/biohub-cell-tracking-during-development` |
| `deadline` | `2026-09-29T23:59:00` |
| `category` | `Research` |
| `reward` | `60,000 Usd` |
| `teamCount` | `951` on 2026-07-09 recheck; `950` in the earlier 2026-07-08 snapshot |
| `userHasEntered` | `true` |

Important limitation: this command does not expose host country, participant country, team country, or notebook author country.

## Extraction Routes

### 1. Competition Metadata

Use the official competition search endpoint through the CLI:

```powershell
kaggle competitions list -s "Biohub Cell Tracking" --format json
```

Use this for deadline, category, reward, team count, and entry status. Treat it as volatile and snapshot it with a timestamp.

### 2. Competition Files

Use:

```powershell
kaggle competitions files biohub-cell-tracking-during-development --page-size 200 --format json
```

Observed fields include `name`, `size`, and `creationDate`, plus a next-page token in CLI output when more files exist. This is useful for mapping the Zarr hierarchy before downloading large data.

Full data download should be a separate, explicit step because competition data can be large:

```powershell
kaggle competitions download biohub-cell-tracking-during-development -p <local-artifact-dir>
```

### 3. Competition Pages

Use:

```powershell
kaggle competitions pages biohub-cell-tracking-during-development --format json
```

Then inspect individual pages when available:

```powershell
kaggle competitions pages biohub-cell-tracking-during-development --page-name <page-name> --format json
```

Use this to capture official page text, rules, data notes, evaluation descriptions, and any host updates exposed by the CLI/API.

### 4. Leaderboard

Use:

```powershell
kaggle competitions leaderboard biohub-cell-tracking-during-development --show --format json
```

For a full local CSV snapshot:

```powershell
kaggle competitions leaderboard biohub-cell-tracking-during-development --download -p <local-artifact-dir>
```

Observed CLI leaderboard display fields are team-oriented, not country-oriented. The leaderboard should be used for scores and teams only unless a downloaded official file includes explicit country fields.

### 5. Submissions And Team Submissions

User-authenticated submission history:

```powershell
kaggle competitions submissions biohub-cell-tracking-during-development --format json
```

Team submission history when a team id is known:

```powershell
kaggle competitions team-submissions <team-id> --format json
```

These routes are useful for score chronology and reproducibility audits. They should not be treated as country evidence unless returned fields explicitly contain a country or location field.

### 6. Public Notebooks

Use multiple sorts because Kaggle notebook discovery is not one-dimensional:

```powershell
kaggle kernels list --competition biohub-cell-tracking-during-development --page-size 100 --sort-by voteCount --format json
kaggle kernels list --competition biohub-cell-tracking-during-development --page-size 100 --sort-by dateRun --format json
kaggle kernels list --competition biohub-cell-tracking-during-development --page-size 100 --sort-by scoreDescending --format json
```

Observed fields for notebook list output include `ref`, `title`, `author`, `lastRunTime`, and `totalVotes`. No reliable country field was observed.

For a selected notebook:

```powershell
kaggle kernels files <owner-or-author>/<kernel-slug> --format json
kaggle kernels pull <owner-or-author>/<kernel-slug> -p <local-artifact-dir> -m
kaggle kernels output <owner-or-author>/<kernel-slug> -p <local-artifact-dir>\output --page-size 200
kaggle kernels logs <owner-or-author>/<kernel-slug>
kaggle kernels status <owner-or-author>/<kernel-slug>
```

Use pulled notebooks for reproducibility, method extraction, dependency lists, and prompt/code patterns. Do not infer country from author display names.

### 7. Discussions And Topics

Competition topics:

```powershell
kaggle competitions topics list biohub-cell-tracking-during-development --sort-by recent --format json
kaggle competitions topics show biohub-cell-tracking-during-development/<topic-id> --format json
```

Observed topic-list fields include `id`, `title`, `authorName`, `commentCount`, `votes`, and `postDate`. In the installed CLI, the public help for `competitions topics list` showed fewer options than the current GitHub docs/source indicate, so commands with `--search`, `--page-size`, or `--page-token` should be verified locally before automation.

Global forum search can be tested separately when the installed CLI help confirms the supported filters:

```powershell
kaggle forums topics list --search "Biohub Cell Tracking" --format json
```

## Python API Methods To Use

Local introspection of `kaggle.api.kaggle_api_extended.KaggleApi` found these relevant methods:

| Method | Use |
|---|---|
| `competitions_list(...)` | Search/list competitions with `search`, `page_size`, and `page_token` support. |
| `competition_list_files(...)` | Enumerate official competition files. |
| `competition_download_files(...)` | Download official competition data. |
| `competition_leaderboard_view(...)` | View leaderboard rows. |
| `competition_leaderboard_download(...)` | Download leaderboard CSV. |
| `competition_submissions(...)` | Read authenticated user submission history. |
| `competition_team_submissions(...)` | Read team submission history by team id. |
| `competition_list_pages(...)` | Enumerate or retrieve competition pages. |
| `competition_list_topics(...)` | List competition discussions. |
| `competition_list_topics_cli(...)` | CLI-oriented topic listing with extra pagination/search arguments in source. |
| `competition_list_topic_messages(...)` | Read topic messages. |
| `kernels_list(...)` | List public notebooks with competition, user, language, kernel type, output type, sort, and page filters. |
| `kernels_list_files(...)` | List files for a public notebook. |
| `kernels_pull(...)` | Pull notebook code and metadata. |
| `kernels_output(...)` | Download notebook outputs. |
| `kernels_logs(...)` | Read notebook logs. |
| `kernels_status(...)` | Read notebook execution status. |
| `kernel_list_topics(...)` | Read notebook discussion topics. |
| `forums_list(...)`, `forums_list_topics(...)`, `forums_topic_show(...)` | Explore wider Kaggle forum sources. |

The Python API is better for pagination, token control, and structured automation. The CLI is better for quick, auditable shell snapshots.

## Country-Coverage Conclusion

The official Kaggle CLI/API can support a rigorous global evidence workflow, but it cannot by itself prove country coverage for all participants or notebooks unless explicit country/location fields are returned.

Use this standard:

| Grade | Meaning | Allowed evidence |
|---|---|---|
| D3 | Official host/source country | Official Kaggle host, Biohub, or institution source. |
| D2 | Public country-attributed competition evidence | A public profile, institution page, notebook, discussion, or source explicitly states the country and links to the exact competition. |
| D1 | Global competition route only | Country has no direct public evidence, but the competition is globally accessible through Kaggle. |
| D0 | No usable route | Neither direct evidence nor a reliable global access route is available. |

Do not upgrade a country because of names, handles, languages, leaderboard positions, time zones, or guessed affiliations.

## Recommended Artifact Layout

Keep raw extraction artifacts outside the curated markdown docs until reviewed:

```text
05_VERTICAL_APPLICATIONS/20_AI_Project_Showcases/Kaggle/Biohub_Cell_Tracking/
  Kaggle_API_CLI_Extraction_Plan_2026-07-08.md
  Current_Competition_Intelligence_2026-07-08.md
  Global_Direct_Competition_Country_Coverage/
  Global_Country_Bioimaging_Coverage/

artifacts/kaggle_biohub_cell_tracking_2026-07-08/
  competition_metadata.json
  competition_files_page_*.json
  competition_pages/
  leaderboard/
  topics/
  kernels/
  extraction_manifest.json
```

Before committing raw artifacts, review them for secrets, private user data, excessive file size, and Kaggle terms/rules compatibility.

## Next Extraction Checklist

1. Snapshot competition metadata, files, pages, leaderboard display, topics, and notebook lists with `--format json`.
2. Save a manifest with command, timestamp, CLI version, and output path for each artifact.
3. Pull only selected high-value notebooks first: official starter, highest-vote baseline, strongest public LB baseline, graph/linking baseline, and visualization notebooks.
4. Compare notebook methods against the official metric and data model.
5. Only then update the curated markdown summaries.
6. Keep the country matrix strict: upgrade D1 to D2 only with explicit public country evidence tied to the exact competition.

## Related Biohub Documents

- [README](./README.md)
- [Devastating Double Check - 2026-07-09](./Devastating_Double_Check_2026-07-09.md)
- [Deep Source Atlas](./Deep_Source_Atlas_2026-07-09.md)
- [Kaggle Notebook and Discussion Radar](./Kaggle_Notebook_Discussion_Radar_2026-07-09.md)
- [ML and AI Model, Feature, Weight, and Calibration Atlas](./ML_AI_Model_Feature_Calibration_Atlas_2026-07-09.md)
