# Prediction Benchmarks & Open Challenges

> Reproducible **benchmarks, open datasets and public prediction challenges** for football (soccer / *futebol*) match forecasting — the rigorous-science backbone that lets you check whether a model is genuinely good or just overfit. Covers the **2017 & 2023 Soccer Prediction Challenges** (*Machine Learning* journal), the **Open International Soccer Database**, **Ranked Probability Score (RPS)** as the field's standard metric, the "**can you beat the bookmaker?**" literature, Kaggle competitions, and public forecasting track records (FiveThirtyEight SPI, ClubElo, Opta). Research & education only — current 2024–2026, Brazil-aware.

> ⚠️ **RESPONSIBLE GAMBLING — READ FIRST. This is a research & education page, NOT betting advice (não é aconselhamento de apostas).** The entire point of a *benchmark* is honesty: the strongest published models on these open challenges score only **marginally** better than the **de-vigged bookmaker closing line**, and none has demonstrated a large, persistent, transferable edge. Betting markets are **highly efficient**; the built-in margin (*overround / margem*) means the **average bettor loses money**, and public "beat-the-bookie" results routinely evaporate once realistic limits, closing-line movement and stake restrictions are applied. A better RPS on a challenge test set is an **academic** result, not a licence to bet. If gambling stops being fun, get help — see [Responsible-gambling resources](#responsible-gambling-resources--jogo-responsável) at the bottom. 🇧🇷 CVV **188** · [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel).

---

## 0. Why benchmarks & challenges exist (the reproducibility angle)

Most football-prediction claims are **irreproducible**: authors pick their own leagues, own train/test split, own metric, and report accuracy against no baseline. A *benchmark* fixes all of that so results are **comparable across teams and years**. Three ingredients define a rigorous soccer-prediction benchmark:

| Ingredient | What it pins down | Community standard |
|---|---|---|
| **Frozen dataset** | Everyone trains on the *same* history | Open International Soccer DB; football-data.co.uk |
| **Held-out future matches** | No leakage — you predict games **not yet played** | 2017 Challenge (206 matches) · 2023 Challenge (736) |
| **Single scoring rule** | One number ranks all entries | **Ranked Probability Score (RPS)**, lower = better |
| **Named baselines** | "Better than what?" | de-vigged bookmaker odds, Elo/pi-ratings, Dixon–Coles |

The gold standard is a **blind, forward-looking competition**: participants submit probabilities for real matches before kickoff, and a neutral organiser scores them. That is exactly what the *Machine Learning* journal Soccer Prediction Challenges provide.

---

## 1. Flagship: the 2017 Soccer Prediction Challenge (*Machine Learning* journal)

Organised by **Daniel Berrar, Philippe Lopes, Jesse Davis & Werner Dubitzky** as a special issue of Springer's *Machine Learning* journal ("Machine Learning for Soccer"). Participants got the **Open International Soccer Database v1.0** and had to predict **206 future matches** (26 leagues, 31 Mar–9 Apr 2017), scored by **average RPS**. Guest editorial: [Springer, *ML* 108:1–7 (2019)](https://link.springer.com/article/10.1007/s10994-018-5763-8).

| Rank / team | Method | Paper (peer-reviewed) | Access |
|---|---|---|---|
| **1st — Team OH** (Hubáček, Šourek, Železný) | Gradient-boosted trees on engineered pi-ratings & PageRank features | [Learning to predict soccer results…GBTs, *ML* 108:29–47 (2019)](https://link.springer.com/article/10.1007/s10994-018-5704-6) | 🆓 [IDA lab copy](https://ida.fel.cvut.cz/papers/hubacek2019learning.html) |
| **2nd — Team ACC** (Constantinou) | **Dolores**: dynamic ratings + hybrid Bayesian networks; predicts leagues a team never played in | [Dolores, *ML* 108:49–75 (2019)](https://link.springer.com/article/10.1007/s10994-018-5703-7) | 🆓 [QMRO PDF](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/38063/Constantinou%20Dolores%20A%20model%202018%20Accepted.pdf?sequence=1&isAllowed=y) |
| **3rd — Team FK** (Tsokos, Narayanan, Kosmidis, Baio et al.) | Bradley–Terry extensions + hierarchical Poisson log-linear | [Modeling outcomes of soccer matches, *ML* 108:77–95 (2019)](https://link.springer.com/article/10.1007/s10994-018-5741-1) | Springer · 🆓 [arXiv:1807.01623](https://arxiv.org/abs/1807.01623) |
| Organisers' method | **Recency feature extraction + rating feature learning** (kNN); best over full DB | [Incorporating domain knowledge, *ML* 108:97–126 (2019)](https://link.springer.com/article/10.1007/s10994-018-5747-8) | Springer |

> **Correction worth flagging:** it is often mis-stated that "Dolores won." It did **not** — the gradient-boosted-tree entry (Team OH) had the best test-set RPS; Dolores placed **2nd**, only marginally behind (≈0.94% higher error than the winner), and the winner's features were themselves built on the **pi-ratings** of Constantinou & Fenton. The margins between the top entries were **tiny**, which is itself the headline finding: on goals-only data, very different methods converge to almost the same RPS.

**Design lesson:** the database deliberately contains **only** dates, teams, league and final goals — no xG, no lineups — to maximise coverage (52 leagues) and force models to generalise from minimal signal.

---

## 2. The 2023 Soccer Prediction Challenge (the follow-up)

A second edition ran in the *Machine Learning* "Machine Learning in Soccer" special issue. Official site: [sites.google.com/view/2023soccerpredictionchallenge](https://sites.google.com/view/2023soccerpredictionchallenge). Training data spanned **51 leagues, 2001 → Apr 2023 (~300k matches)**; the test set was **736 matches from 44 leagues (14–26 Apr 2023)**. Two tasks:

| Task | Target | Metric | Notes |
|---|---|---|---|
| **Task 1** | Exact score (home & away goals) | **RMSE** | Harder; few models beat a naïve Poisson |
| **Task 2** | P(home win / draw / away win) | **average RPS** | The 1X2 headline metric |

Key peer-reviewed outputs (all *Machine Learning*, 2024):

| Paper | Contribution | Reported result | Link |
|---|---|---|---|
| Berrar et al. — **data- & knowledge-driven framework** | kNN, ANN, naïve Bayes, ordinal forests on curated ratings | **Berrar ratings best: RPS ≈ 0.2101, acc ≈ 0.4854** (> Bivariate/Double Poisson, pi-ratings) | [Springer 10.1007/s10994-024-06625-9](https://link.springer.com/article/10.1007/s10994-024-06625-9) · 🆓 [ORO](https://oro.open.ac.uk/100167/) |
| Yeung, Bunker, Umemoto & Fujii — **DL vs gradient-boosted trees** | CatBoost + pi-ratings vs deep nets; feature optimisation | CatBoost "strong & stable" vs 2017-Challenge models | [Springer 10.1007/s10994-024-06608-w](https://link.springer.com/article/10.1007/s10994-024-06608-w) · 🆓 [arXiv:2309.14807](https://arxiv.org/abs/2309.14807) |
| Reproducible code | Reference implementation of Berrar/Poisson/pi-rating baselines & deep nets | Python | 🆓 [github.com/calvinyeungck/Soccer-Prediction-Challenge-2023](https://github.com/calvinyeungck/Soccer-Prediction-Challenge-2023) |

**Takeaway:** across both editions the winning RPS sits in a narrow **~0.19–0.21** band — right around what de-vigged market odds achieve. Six years of new methods (incl. deep learning) moved the needle only slightly.

---

## 3. Benchmark datasets (the frozen ground truth)

| Dataset | Content | Size / coverage | Cost | Link |
|---|---|---|---|---|
| **Open International Soccer Database** | Date, teams, league, final goals only — the Challenge benchmark | 216,743 matches · 52 leagues · 35 countries · 2000–2017 (v1.0) | 🆓 | [ML paper 10.1007/s10994-018-5726-0](https://link.springer.com/article/10.1007/s10994-018-5726-0) · [OSF mirror](https://osf.io/kqcye/) |
| **football-data.co.uk** | Results **+ multi-book odds** (Bet365, Pinnacle, closing lines), O/U, Asian handicap | 25+ leagues · 2000 → today, weekly CSV | 🆓 | [football-data.co.uk/data.php](https://www.football-data.co.uk/data.php) |
| **StatsBomb Open Data** | Full **event data** + StatsBomb 360 freeze-frames for selected comps (WC, WSL, La Liga-Messi) | JSON, dozens of competitions | 🆓 (attribution required) | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) · [statsbombpy](https://github.com/statsbomb/statsbombpy) |
| **Kaggle European Soccer Database** (H. Mathien) | Matches, odds (≤10 books), EA FIFA player/team attributes, events | 25k+ matches · 11 countries · 2008–2016 | 🆓 | [kaggle.com/datasets/hugomathien/soccer](https://www.kaggle.com/datasets/hugomathien/soccer) |
| **SoccerNet** | **Computer-vision** benchmark: action spotting, tracking, re-ID, game-state reconstruction | 500+ broadcast games, ~800h video | 🆓 (research) | [soccer-net.org](https://www.soccer-net.org/) · [github.com/SoccerNet](https://github.com/SoccerNet) · [SoccerNet 2024 results](https://arxiv.org/abs/2409.10587) |

**How to choose:** use **Open International Soccer DB** to compare a rating/1X2 model against Challenge leaderboards; **football-data.co.uk** as the *de-facto* benchmark whenever you need **odds** (for CLV / value studies); **StatsBomb open** for event-level / xG modelling; **SoccerNet** for the vision pipeline that *produces* tracking data upstream of prediction.

---

## 4. The standard metric: Ranked Probability Score (RPS) — and the live debate

Football outcomes are **ordinal** (away < draw < home), so a good scoring rule should penalise a confident "home" more when the result is "away" than when it is "draw". The **RPS** does exactly this and has become the field's default since **Constantinou & Fenton (2012)** argued the older rules (Brier, accuracy, geometric mean) are inadequate for ordinal football forecasts.

| Metric | Type | Property | Reference |
|---|---|---|---|
| **RPS** (Ranked Probability Score) | proper, **sensitive to distance**, ∈ [0,1] | Standard for 1X2; lower = better | [Constantinou & Fenton, *JQAS* 8(1) 2012](https://www.degruyter.com/document/doi/10.1515/1559-0410.1418/html) · 🆓 [PDF](http://constantinou.info/downloads/papers/solvingtheproblem.pdf) |
| **Ignorance / log score** | proper, **local** (only the realised outcome) | Argued to *out-perform* RPS & Brier | [Wheatcroft, "The case against the RPS" 2019](https://arxiv.org/abs/1908.08980) |
| **Brier score** | proper, insensitive to distance | Simpler; ignores ordinality | classic |
| **RMSE** (exact score) | for goal counts | Used in 2023 Challenge Task 1 | see §2 |

> **Open challenge #1 — which metric?** The RPS is *not* uncontested. Wheatcroft (2019) shows the "distance sensitivity" that motivated RPS "adds nothing" to the actual goal of scoring rules and that the **ignorance (log) score** discriminates models better. Reporting **both RPS and log-loss** — plus **calibration** curves — is now best practice.

**Orientation scale (typical out-of-sample 1X2 RPS):** random/uniform ≈ 0.23; a decent Elo/Dixon–Coles ≈ 0.21; state-of-the-art & de-vigged **market closing odds** ≈ 0.19–0.20. The entire competitive range is a few thousandths of RPS — which is why **rigorous, matched baselines matter more than the model**.

---

## 5. Baselines & the "can you beat the bookmaker?" benchmark studies

The most important baseline is the **bookmaker's own de-vigged closing line**. Beating it out-of-sample, after margin, is the real bar — and the honest literature says it is **very hard**.

| Study | Claim | Sober reading | Link |
|---|---|---|---|
| **Hubáček, Šourek & Železný (2019)** — *Exploiting sports-betting market using ML* | CNN on player stats + **de-correlating** from the bookmaker's odds → simulated profit | Profit hinges on finding books slower than the market; fragile to limits | [*Int. J. Forecasting* 35(2):783–796](https://www.sciencedirect.com/science/article/abs/pii/S016920701930007X) · 🆓 [PDF](http://ida.felk.cvut.cz/zelezny/pubs/ijf.2019.pdf) |
| **Kaunitz, Zhong & Kreiner (2017)** — *Beating the bookies with their own numbers* | Bet mispriced odds vs a market-average estimate; profitable incl. **real-money** trial | Bookmakers **limited/closed** the winning accounts — "the market is rigged" | [arXiv:1710.02824](https://arxiv.org/abs/1710.02824) · 🆓 [code](https://github.com/Lisandro79/BeatTheBookie) |
| **Walsh & Joshi (2024)** — accuracy vs calibration (**NBA study**) | Model selection by **accuracy** ≠ by **calibration**; the lesson transfers to football betting | Calibrate, don't just classify | [arXiv:2303.06021](https://arxiv.org/abs/2303.06021) · [MLwA 16:100539](https://doi.org/10.1016/j.mlwa.2024.100539) |
| **Hubáček & Šír — "Beating the market with a bad predictive model"** | You can profit with a *worse* model if it's decorrelated from the market | Edge = *disagreement*, not accuracy — but disagreement is usually just noise | [arXiv:2010.12508](https://arxiv.org/abs/2010.12508) |
| **Systematic review — ML in sports betting (2024)** | Surveys techniques, data quality and evaluation challenges | Most "profitable" papers have methodological holes | [arXiv:2410.21484](https://arxiv.org/abs/2410.21484) |

> **Open challenge #2 — realistic evaluation.** Nearly every profitable claim uses **pre-closing** odds, ignores **stake limits / account closure**, or leaks information. A credible benchmark must bet at the **closing line** (or worse), cap stakes, and report **CLV** — see the section's [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) page. The consensus of the rigorous literature: **a durable, sizeable edge over efficient markets has not been convincingly demonstrated.**

---

## 6. Kaggle competitions (open leaderboards)

Public leaderboards are a lower-barrier complement to the journal challenges — many entries, shared notebooks, but weaker leakage controls.

| Competition / dataset | Task | Data | Metric | Link |
|---|---|---|---|---|
| **Football Match Probability Prediction** (Octosport × Sportmonks, 2022) | P(home/draw/away) from each team's last **10 matches** | 150k+ matches · 860+ leagues · 9,500 teams · 2019–2021 | **RPS**; bookmaker odds included as benchmark; 382 competing teams (LSTM winners) | [Kaggle competition](https://www.kaggle.com/competitions/football-match-probability-prediction) · [host write-up](https://www.sportmonks.com/blogs/football-predictions-kaggle-competition/) |
| **Prediction of Results in Soccer Matches** | Predict 1X2 **and** show profit vs historical odds | league match history + odds | profit / accuracy | [Kaggle competition](https://www.kaggle.com/competitions/prediction-of-results-in-soccer-matches) |
| **FIFA World Cup 2022 / 2026 notebooks** | Tournament simulation from FIFA rankings + results | intl. results 1872→now | varies | [example notebook](https://www.kaggle.com/code/sslp23/predicting-fifa-2022-world-cup-with-ml) |

**Caveat:** Kaggle public leaderboards reward **leaderboard-fitting**; the private split and the bookmaker benchmark are what matter. Treat a Kaggle rank as a *sanity check*, not proof of a betting edge.

---

## 7. Public forecasting track records & leaderboards (audit them yourself)

Long-running public forecasters let you compute **your own** RPS/log-loss over thousands of past predictions — a real-world benchmark that outlives any single paper.

| Forecaster | What it publishes | Status 2024–2026 | Cost | Link |
|---|---|---|---|---|
| **FiveThirtyEight SPI** (Soccer Power Index) | `spi_matches.csv`: pre-match win/draw/loss probs + xG proj., back to 2016 | ⚠️ **Frozen 21 Jun 2023**; 538 fully shut by Disney/ABC Mar 2025 — but CSVs remain **archived & downloadable** | 🆓 | [github.com/fivethirtyeight/data/tree/master/soccer-spi](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) · [archived project](https://projects.fivethirtyeight.com/soccer-predictions/) |
| **ClubElo** | Daily Elo since **1939**; match probabilities incl. per-goal-difference distribution | ✅ Active | 🆓 public API | [clubelo.com](http://clubelo.com/) · [API docs](http://clubelo.com/API) |
| **Opta Analyst "supercomputer"** | 10,000-run season simulations (Opta Power Rankings + market odds); per-match 1X2 | ✅ Active | 🆓 articles (data = paid) | [theanalyst.com football predictions](https://theanalyst.com/articles/opta-football-predictions) |
| **Superbru Predictor** | Community score-prediction game (2.9M+ players) — a human-forecaster baseline | ✅ Active | 🆓 | [superbru.com](https://www.superbru.com/) |

> **Open challenge #3 — a living leaderboard.** Unlike the one-off journal challenges, no continuously-updated, neutral, **odds-anchored** public leaderboard survives (538's death left a gap). Rebuilding one — ClubElo / Opta / a Poisson baseline vs the closing line, scored weekly on RPS **and** log-loss — is an obvious, high-value open-science project.

---

## 8. Reproducibility checklist (use before you believe any result)

1. **Forward split only** — test on matches *after* every training match; never shuffle across time (avoids leakage / *vazamento*).
2. **Name the baseline** — de-vigged **closing** odds + Dixon–Coles + Elo/pi-ratings, minimum.
3. **Report ≥2 proper scores** — RPS **and** log-loss (ignorance), plus a **calibration** plot.
4. **Bet at closing (or worse)** and report **CLV**, stake caps, and account-limit assumptions before any "profit."
5. **Publish code + the frozen dataset id** (Open International Soccer DB version, football-data.co.uk snapshot date).
6. **Beware tiny margins** — top methods differ by thousandths of RPS; quote **confidence intervals** over many matches.

Survey to orient: **Bunker, Yeung & Fujii — *Machine Learning for Soccer Match Result Prediction*** ([arXiv:2403.07669](https://arxiv.org/abs/2403.07669)) reviews methods, datasets and the Challenge results, and recommends **gradient-boosted trees (CatBoost) + soccer-specific ratings** as the practical benchmark to beat.

---

## Responsible-gambling resources · Jogo Responsável

Benchmarks measure *forecast quality*, not a path to profit. Markets are efficient, **most bettors lose**, and nothing here is advice. If betting is causing harm, help is free and confidential:

| Resource | Region | Link |
|---|---|---|
| **BeGambleAware** | UK / global | [begambleaware.org](https://www.begambleaware.org/) |
| **GamCare** (National Gambling Helpline) | UK | [gamcare.org.uk](https://www.gamcare.org.uk/) |
| **Gambling Therapy** | Global (multilingual) | [gamblingtherapy.org](https://www.gamblingtherapy.org/) |
| 🇧🇷 **Jogo Responsável / SPA-MF** | Brazil (*apostas de quota fixa*) | [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) |
| 🇧🇷 **CVV — apoio emocional** | Brazil, 24h | ligue **188** · [cvv.org.br](https://www.cvv.org.br/) |

Set limits (*defina limites*), never chase losses (*nunca persiga perdas*), and treat this material as **science, not tips** (*ciência, não palpites*).

**Keywords:** soccer prediction benchmark, football match prediction challenge, Open International Soccer Database, 2017 Soccer Prediction Challenge, 2023 Soccer Prediction Challenge, Ranked Probability Score, RPS, ignorance score, Dixon–Coles baseline, pi-ratings, Dolores model, beat the bookmaker, closing line value, FiveThirtyEight SPI, ClubElo, Opta supercomputer, StatsBomb open data, SoccerNet, Kaggle football, reproducibility · *benchmark de previsão de futebol, desafio de previsão de partidas, base de dados aberta de futebol, pontuação de probabilidade ordenada, linha de fechamento, apostas esportivas, jogo responsável, ciência reprodutível*
