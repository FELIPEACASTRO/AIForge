# Direct Competition Country Methodology

This folder is stricter than the broader bioimaging country matrix. It covers only direct public evidence for the specific Kaggle competition **Biohub - Cell Tracking During Development** by country.

## Core Rule

Do not assign a country from a Kaggle username, author name, personal name, language, or team name. A country assignment requires one of these evidence types:

- official competition or host source with an explicit country/location;
- public Kaggle profile or notebook metadata with explicit country/location;
- public institutional article, university page, lab page, company page, or government/research-infrastructure page linking the country to this exact competition;
- public discussion/post that clearly links the exact competition to an identified country or institution.

## Direct Evidence Tiers

| Tier | Meaning |
|---|---|
| D3 official host/source country | Official competition/host/source country evidence. |
| D2 public country-attributed competition evidence | A public source links this exact Biohub competition to a country-specific person, team, institution, notebook, or article. |
| D1 global competition route only | The competition is globally available through Kaggle, but no country-specific public source was captured in this pass. |
| D0 no usable direct source | No direct competition evidence and no reliable global route was captured. |

## Important Limitation

The Kaggle CLI outputs used in this pass expose team names, notebook refs, authors, dates, votes, and public scores. They do not provide a reliable country field. Therefore, the leaderboard and notebook lists are not enough to assign most countries.

## Source References

| Source ID | Source | URL | Use in this direct-country pass |
|---|---|---|---|
| BIOHUB-KAGGLE | Kaggle Biohub - Cell Tracking During Development | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development | Official competition page, task, rules, data and global access point. |
| BIOHUB-ORG | Biohub Kaggle organization profile | https://www.kaggle.com/organizations/biohub | Official Kaggle organization hosting the competition. |
| BIOHUB-ROYER | Royer Group at Chan Zuckerberg Biohub | https://biohub.org/royer/ | Host-lab scientific context for zebrafish developmental imaging and computational microscopy. |
| ROYER-OFFICIAL-REPO | Royer Lab official competition repository | https://github.com/royerlab/kaggle-cell-tracking-competition | Official code, data model, metric and baseline implementation. |
| ROYER-METRICS | Official metric specification | https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md | Direct source for edge Jaccard, division Jaccard and final score. |
| KAGGLE-CODE | Kaggle public code tab | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/code | Public notebooks; useful but no reliable country field in CLI output. |
| KAGGLE-LEADERBOARD | Kaggle public leaderboard | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard | Public team score snapshot; useful but no reliable country field in CLI output. |
| BACKLOG | Country-specific search backlog | ./Country_Specific_Search_Backlog_2026-07-08.md | Follow-up searches for public country-attributed evidence. |
