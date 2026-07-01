# Features & Feature Engineering for Football Prediction

> Authoritative, source-backed catalogue of the **features (variáveis / atributos)** that actually drive football (soccer / futebol) match-outcome and betting models — strength ratings, form, the expected-goals family, context, team news, style, referee and market signals — plus the engineering craft (encoding, normalization, **point-in-time / anti-leakage**, feature selection, feature stores). Research & education, current 2024–2026.

> ⚠️ **Research & education only — NOT betting advice (não é dica de aposta).** Football betting markets are **highly efficient**: across 397,935 Pinnacle games the sharp closing line tracked realized outcomes at **r² ≈ 0.997** ([Trademate Sports, citing Pinnacle](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458); see also [football-data.co.uk efficiency analysis](https://www.football-data.co.uk/blog/pinnacle_efficiency.php)). **Most bettors lose money over time**, and a genuine, sustained edge is rare and hard to keep (winning accounts get limited/closed). Great features do **not** guarantee profit — treat everything below as modeling practice. Help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [SPA / Ministério da Fazenda (regulador de apostas)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/apostas-de-quota-fixa) · 🇧🇷 CVV **188**.

**Data sources for every feature below** live in the sibling pages: [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets](./Kaggle_Football_Datasets_and_Competitions.md).

---

## 0) The one rule that matters most: point-in-time (only pre-match info!)

Every feature for a fixture must be computable **using only information available before kickoff (apenas informação pré-jogo)**. The single biggest cause of "amazing" models that die in production is **target/temporal leakage (vazamento de dados)** — accidentally feeding post-match or same-match info (final xG, result-derived form, the closing line you couldn't have had, a lineup published after your bet). Build features as **lagged, as-of-date snapshots** and validate **chronologically** (walk-forward / expanding window, never shuffled k-fold). See [§10](#10-feature-engineering-craft-encoding-normalization-leakage-selection-stores).

**Feature families at a glance:**

| Family | Example features | Signal strength | Primary sources |
|---|---|---|---|
| Strength / ratings | Elo, pi-ratings, SPI, squad value, market-implied | ⭐⭐⭐⭐ | ClubElo, Transfermarkt, odds |
| Form & momentum | rolling pts/goals, xG rolling, streaks | ⭐⭐⭐ (regresses to mean) | FBref, Understat |
| Expected goals family | xG, npxG, xGD, xT, VAEP, xPts | ⭐⭐⭐⭐ | Understat, FBref, StatsBomb |
| Context | home adv., rest, congestion, travel, altitude | ⭐⭐–⭐⭐⭐ | fixtures, APIs |
| Team news | lineups, absences, injuries, manager change | ⭐⭐⭐ (if pre-match) | API-Football, press |
| H2H & style | PPDA, possession, set-piece reliance | ⭐⭐ | FBref, StatsBomb |
| Referee | cards / penalty tendencies | ⭐ (niche markets) | FBref, FootyMetrics |
| Market | opening/closing odds, movement, volume | ⭐⭐⭐⭐⭐ (hard to beat) | football-data, OddsPortal |

---

## 1) Strength & ratings (força / rating do time)

The backbone of most models: a compact number for "how good is each side right now", ideally split into attack and defence. Their **difference** (e.g. Elo diff) is a strong single predictor.

| Feature | Definition (PT) | Why it matters | Data source |
|---|---|---|---|
| **Elo rating / Elo diff** | Zero-sum rating updated by result vs expectation (rating de Elo, diferença de Elo) | Cheap, robust baseline; the home−away Elo gap maps almost directly to win probability | [ClubElo API](http://clubelo.com/API) (clubs, daily since 1939) · [eloratings.net](https://eloratings.net/) (national) |
| **pi-ratings** | Dynamic home/away ratings from **relative score discrepancies**, recency-weighted (Constantinou & Fenton) | Beat Elo in tests and returned a profit vs published EPL odds over 5 seasons | [paper (JQAS 2013)](http://constantinou.info/downloads/papers/pi-ratings.pdf) |
| **SPI (Soccer Power Index)** | Offence/defence goal-expectation vs an average team → overall strength estimate | Combines results + margin + opponent quality into one interpretable index | Method (offence/defence ratings) + data: [fivethirtyeight/data soccer-spi](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) ⚠️ FiveThirtyEight closed in 2023; ratings are archived and **no longer updated** |
| **Market-implied strength** | De-vigged probabilities from odds → implied team strength (força implícita do mercado) | The market's consensus is the hardest-to-beat "rating"; excellent feature and benchmark | [football-data.co.uk](https://www.football-data.co.uk/data.php) · de-vig: [penaltyblog](https://penaltyblog.readthedocs.io/en/latest/implied/implied.html) |
| **Squad market value** | Sum/mean of player values (valor de elenco, Transfermarkt) | Peeters (2018): value-based forecasts rival betting odds and **beat Elo/FIFA** rankings | [Transfermarkt](https://www.transfermarkt.com/) · dataset: [dcaribou](https://github.com/dcaribou/transfermarkt-datasets) · [Peeters 2018](https://www.researchgate.net/publication/322175868_Testing_the_Wisdom_of_Crowds_in_the_field_Transfermarkt_valuations_and_international_soccer_results) |
| **Poisson attack/defence strength** | Team scoring/conceding rates → Dixon–Coles λ parameters (força de ataque/defesa) | Classic generative model; the ρ low-score correction + time-decay is a whole feature set | [Dixon & Coles 1997](https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9876.00065) · [tutorial](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/) |

> Note: ratings encode strength **as of a date** — always join the rating snapshot dated **before** the match, never the post-match update.

---

## 2) Form & momentum (forma & momento)

Recent-performance features. Useful but **noisy and mean-reverting** — raw "last-5 results" overfits; xG-based form is more stable than goal/points-based form.

| Feature | Definition (PT) | Why it matters | Data source |
|---|---|---|---|
| **Rolling points / PPG** | Points per game over last N matches (média de pontos por jogo) | Simple momentum proxy; window N (3–10) is a tunable hyperparameter | Any results feed ([football-data.co.uk](https://www.football-data.co.uk/data.php)) |
| **Rolling goals for / against** | GF/GA moving averages (gols marcados/sofridos) | Captures scoring trend; blend home-only vs away-only splits | football-data.co.uk, [OpenFootball](https://github.com/openfootball) |
| **Rolling xG / xGA** | Moving average of expected goals for/against | **More predictive of future goals than past goals** — less noisy signal of underlying level | [Understat](https://understat.com/) · [FBref](https://fbref.com/) |
| **Shot-based form** | Rolling shots, shots on target, big chances (finalizações) | Volume proxy that leads goals; robust in small samples | FBref, [StatsBomb open data](https://github.com/statsbomb/open-data) |
| **Weighted / decayed form** | Exponential time-decay so recent games count more (forma ponderada) | Same idea as Dixon–Coles ξ half-life; reduces staleness | [time-weighting](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/) |
| **Streaks** | Consecutive W/D/L, unbeaten/scoring runs (sequências / invencibilidade) | Weak marginal signal once ratings are included; watch for leakage & overfitting | derived from results |

---

## 3) The Expected Goals family (família de Expected Goals — xG)

The most valuable modern feature class. xG assigns each shot a scoring probability from its characteristics; aggregating and extending it yields a rich family. (FBref/StatsBomb value a penalty at **xG ≈ 0.76**; Opta uses **0.79**.)

| Feature | Definition (PT) | Why it matters | Data source |
|---|---|---|---|
| **xG (Expected Goals)** | Sum of shot scoring probabilities (gols esperados) | Best single "deserved goals" signal; smooths finishing luck | [FBref xG explained](https://fbref.com/en/expected-goals-model-explained/) · [Understat](https://understat.com/) |
| **xGA (xG Against)** | xG conceded (gols esperados sofridos) | Defensive quality proxy, less noisy than goals against | FBref, Understat |
| **xGD (xG Difference)** | xG − xGA, often per 90 (saldo de gols esperados) | Strong league-table and future-results predictor | FBref, Understat |
| **npxG (non-penalty xG)** | xG excluding penalties (xG sem pênaltis) | Removes penalty distortion → cleaner open-play quality | [FBref](https://fbref.com/en/expected-goals-model-explained/) |
| **xG overperformance** | Goals − xG (goals scored above/below expected) | Flags unsustainable hot/cold finishing → **fade regression to the mean** | derived (FBref/Understat) |
| **Set-piece xG (xGSP)** vs **open-play xG (xGOP)** | xG split by phase (xG de bola parada vs bola rolando) | Set plays convert ~2× open play; reveals set-piece-reliant sides (e.g. Arsenal) | [StatsBomb](https://github.com/statsbomb/open-data), FBref |
| **xT (Expected Threat)** | Pitch split into a **16×12 grid**; value = danger added by moving the ball (ameaça esperada) | Credits build-up/progression, not just shots; team-style signal | [Karun Singh (2018)](https://karun.in/blog/expected-threat.html) · [socceraction](https://github.com/ML-KULeuven/socceraction) |
| **VAEP / xPossession value** | ML value of each action via score/concede probability change | Broadest action-value framework; context-aware | [VAEP (KU Leuven)](https://dtai.cs.kuleuven.be/sports/vaep/) · [socceraction (Python)](https://github.com/ML-KULeuven/socceraction) |
| **xPoints (xPts)** | Simulate match from both sides' shot xG → expected points (pontos esperados) | Detects lucky/unlucky results → better strength estimate than actual points | [Understat](https://understat.com/) (xPTS column) · [method](https://mckayjohns.substack.com/p/how-to-calculate-expected-points) |

---

## 4) Contextual features (contexto do jogo)

Conditions around the match. Individually small, collectively meaningful — and easy to get **for free** from schedules/APIs.

| Feature | Definition (PT) | Why it matters | Data source |
|---|---|---|---|
| **Home advantage** | Baseline edge of playing at home (fator casa / mando de campo) | Historically ~0.3–0.6 goals; **declined post-COVID** — COVID "ghost games" (torcida ausente) nearly **halved** the home points advantage (home-win rate ~18% lower, away ~30% higher) via weaker referee/crowd effects | [systematic review (Springer)](https://link.springer.com/article/10.1007/s11301-021-00254-5) · [ScienceDaily](https://www.sciencedaily.com/releases/2021/08/210813100323.htm) |
| **Rest days & fixture congestion** | Days since last match; matches in a window (descanso, calendário congestionado) | Injury risk rises under congested schedules (≤4 days recovery), which also degrades output | [systematic review (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9758680/) · [meta-analysis](https://link.springer.com/article/10.1007/s40279-020-01359-9) |
| **European midweek fatigue** | Played UCL/UEL midweek before a league game (desgaste de jogo europeu) | Rotation + fatigue lowers weekend performance; interacts with congestion | fixtures via [API-Football](https://www.api-football.com/) |
| **Travel distance / time zones** | Km travelled, TZ crossings (distância de viagem) | Long trips can reduce away performance; the travel-distance ↔ home-advantage link is measurable over decades (Bundesliga) | [Springer (57-yr study)](https://link.springer.com/article/10.1007/s12662-021-00787-7) |
| **Altitude** | Venue elevation & altitude *difference* (altitude) | BMJ: each **+1000 m difference ≈ +0.5 goal**; home-win prob 0.54→0.83 for Bolivia-type gaps (La Paz) | [McSharry, BMJ 2007](https://www.bmj.com/content/335/7633/1278) |
| **Weather** | Rain, wind, temperature (clima: chuva, vento, temperatura) | Secondary signal: extremes dampen tempo/goals; weak vs form/strength — use cautiously | [Forebet insight](https://www.forebet.com/en/insight/24315-how-weather-affects-football-and-match-predictions) · [Open-Meteo](https://open-meteo.com/) |
| **Kick-off time / day** | Slot & weekday (horário do jogo) | Proxies for congestion, TV picks, crowd; minor main effect | fixtures / APIs |
| **Derby / rivalry** | Local/historic rivalry flag (clássico / dérbi) | Motivation & pressure override form → **higher variance, harder to predict** | [Derbyist](https://derby.ist/all), manual flags |
| **Motivation / stakes** | Title race, relegation fight, dead rubber (rebaixamento, jogo sem valor) | End-of-season "dead rubbers" invite rotation/effort drops → outcome shifts | standings-derived |

---

## 5) Team news (notícias do time — lineups, absences, manager)

High-value but **time-sensitive**: confirmed lineups drop ~1 hour pre-match. Use only what a bettor could actually have at decision time (predicted vs confirmed XI).

| Feature | Definition (PT) | Why it matters | Data source |
|---|---|---|---|
| **Confirmed / predicted lineups** | Starting XI & formation (escalação) | Big swing vs a "full-strength" prior; rotation detection | [API-Football](https://www.api-football.com/) (lineups), [SportMonks](https://www.sportmonks.com/football-api/) |
| **Key-player absence** | Star/high-VAEP player missing (desfalque de titular) | Value-weighted absence (e.g. lost squad value %) beats a raw count | Transfermarkt values × lineups |
| **Injuries & suspensions** | Injured/banned players (lesões e suspensões) | Aggregate lost minutes/value; suspensions are known days ahead (less leakage risk) | API-Football injuries, [SportMonks](https://www.sportmonks.com/) |
| **Manager change** | New head coach recently appointed (troca de técnico) | "New-manager bounce" is real short-term (~+0.3 PPG over 5 games) but **largely regression to the mean** — model with skepticism | [Premier League](https://www.premierleague.com/en/news/4593686/what-is-a-new-manager-bounce-and-is-it-a-myth) · [analysis](https://socceranalytics.substack.com/p/is-the-new-manager-bounce-really) |

---

## 6) Head-to-head & playing style (confronto direto & estilo)

Matchup-specific signals. H2H is popular but **weak once ratings are included** (small samples, changing squads). Style features are richer.

| Feature | Definition (PT) | Why it matters | Data source |
|---|---|---|---|
| **Head-to-head (H2H)** | Prior meetings' results (confronto direto) | Intuitive but low signal beyond strength; prone to overfit on tiny N | results feeds |
| **Possession %** | Share of the ball (posse de bola) | Style descriptor; weak alone, useful in interactions | FBref, [Sofascore](https://www.sofascore.com/) |
| **PPDA (Passes per Defensive Action)** | Opp passes ÷ defensive actions in the pressing zone (~front 60%) — pressing intensity | Low PPDA (~4–8) = high press; mismatches (press vs build-up) shift outcomes | [Premier League def.](https://www.premierleague.com/en/news/4250153/passes-per-defensive-action-explained) · [StatsBomb/Hudl glossary](https://support.hudl.com/s/article/passes-defensive-action) |
| **Style / matchup vectors** | Directness, width, build-up type, aerial reliance (estilo de jogo) | Style clashes (e.g. press vs long-ball) inform tactical priors | FBref, StatsBomb, [socceraction](https://github.com/ML-KULeuven/socceraction) |
| **Set-piece reliance** | Share of xG/goals from set plays (dependência de bola parada) | Identifies teams whose output is corner/free-kick driven | StatsBomb, FBref (xGSP) |

---

## 7) Referee & market features (árbitro & mercado)

| Feature | Definition (PT) | Why it matters | Data source |
|---|---|---|---|
| **Referee card/penalty tendency** | Per-ref averages: yellows, reds, penalties, fouls, added time (tendência de cartões/pênaltis do árbitro) | Referee identity drives **cards & booking-points markets** (less so 1X2); studies also document referee bias in decisions | [FootyMetrics referees](https://www.footymetrics.com/referees) · FBref match refs · [bias study (favoritism)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7739605/) |
| **Opening & closing odds** | Bookmaker prices at open and at kickoff (odds de abertura e fechamento) | De-vigged closing odds ≈ best public probability estimate; a top-tier feature/benchmark | [football-data.co.uk](https://www.football-data.co.uk/data.php) (open+close since 2019/20) |
| **Odds movement / drift** | Change from open→close, direction & speed (variação das odds) | "Steam"/reverse line moves proxy sharp money & fresh info (news, lineups) | [OddsPortal](https://www.oddsportal.com/football/results/) (history), scrape ⚠️ ToS |
| **Market overround / margin** | Sum of implied probs − 100% (vig / overround) | Must be **removed** before using odds as probabilities; also a liquidity proxy | de-vig via [penaltyblog](https://penaltyblog.readthedocs.io/en/latest/implied/implied.html) |
| **Volume / limits (liquidity)** | Money matched / max stake (liquidez) | Sharper, higher-limit markets (Pinnacle, exchanges) are more efficient | exchange APIs (Betfair) |

> **Reality check:** market features are the most powerful **and** the reason edges are rare — if your model only re-learns the closing line, you have no edge, just the vig to pay.

---

## 8) Encoding & normalization (codificação & normalização)

| Task | Technique | Notes (PT) |
|---|---|---|
| Team / league IDs | Target/mean encoding, embeddings, or ratings-as-features | Prefer **ratings** (Elo/value) over one-hot: hundreds of teams → sparse; embeddings need lots of data (evitar one-hot esparso) |
| Home/away, derby flags | Binary / one-hot | Cheap and interpretable |
| Skewed counts (xG, values) | log1p / winsorize | Squad values & odds are heavy-tailed (log ajuda) |
| Scale for linear/NN models | Standardize (z-score) or min-max | Tree models (XGBoost/LightGBM) are scale-invariant — skip for them |
| Rates not totals | Per-90 / per-match normalization | Compare across differing minutes/samples (métricas por 90) |
| Relative, not absolute | Use **differences** (home − away) | A single "strength diff" feature often beats two raw features |

---

## 9) Feature selection (seleção de variáveis)

| Method | What it does | Tool |
|---|---|---|
| Correlation / VIF pruning | Drop redundant, collinear features (multicolinearidade) | pandas, statsmodels |
| Model importance | Gain / permutation importance from GBMs | XGBoost, LightGBM |
| **SHAP** | Consistent per-feature contributions & interactions | [shap](https://github.com/shap/shap) |
| Recursive elimination | RFE / boruta backward selection | scikit-learn, boruta |
| Regularization | L1 (lasso) shrinks weak features to zero | scikit-learn |

Guiding principle: **fewer, causal, leak-free features beat a wide leaky matrix.** Prefer signals with a plausible mechanism (strength, xG, rest) over spurious ones (raw H2H, kickoff minute).

---

## 10) Feature engineering craft: leakage, point-in-time, stores {#10-feature-engineering-craft-encoding-normalization-leakage-selection-stores}

**Anti-leakage checklist (checklist anti-vazamento):**
- ✅ Every feature uses **only pre-kickoff** data (lagged / as-of snapshots). No final xG, no full-time result-derived form, no closing line you couldn't hold at bet time.
- ✅ Rolling windows computed with `shift()` so the current match is **excluded** from its own features.
- ✅ Rating/value/lineup joins use the **as-of date**, not the latest value.
- ✅ Fit encoders/scalers on **train only**, inside the temporal split (fold-safe pipelines).
- ✅ Validate **chronologically**: walk-forward / expanding window; **never** shuffled k-fold ([why](https://towardsdatascience.com/putting-your-forecasting-model-to-the-test-a-guide-to-backtesting-24567d377fb5/) · [leakage](https://www.ibm.com/think/topics/data-leakage-machine-learning)).
- ✅ Backtest bets at prices you could have taken; grade skill with **CLV** (closing line value), not just ROI on a lucky sample.

**Feature stores & automation:**

| Tool | Role | Free/OSS | Link |
|---|---|---|---|
| **Feast** | Open-source feature store; **point-in-time correct** joins for train/serve consistency | ✅ Apache-2.0 | [github](https://github.com/feast-dev/feast) |
| **Featuretools** | Automated feature synthesis (deep feature synthesis) with time cutoffs | ✅ BSD | [github](https://github.com/alteryx/featuretools) |
| **Hopsworks / Tecton** | Managed feature platforms (online+offline) | Freemium / Paid | [hopsworks.ai](https://www.hopsworks.ai/) · [tecton.ai](https://www.tecton.ai/) |
| **soccerdata** | Pulls FBref/Understat/ClubElo/Sofascore into tidy frames to build features | ✅ | [github](https://github.com/probberechts/soccerdata) |
| **worldfootballR** | R access to FBref/Transfermarkt/Understat | ✅ | [github](https://github.com/JaseZiv/worldfootballR) |
| **socceraction / kloppy** | Event→SPADL, xT/VAEP features; standardize event/tracking | ✅ | [socceraction](https://github.com/ML-KULeuven/socceraction) · [kloppy](https://github.com/PySport/kloppy) |

> A feature store isn't required for a personal project, but its core idea — **compute a feature once, serve the exact same point-in-time value to training and to live prediction** — is exactly what stops train/serve skew and leakage.

---

## 11) 🇧🇷 Brazil-specific notes (notas para o Brasil)

- **Brasileirão data with the same features:** results/odds via [football-data.co.uk (Brazil)](https://www.football-data.co.uk/all_new_data.php); xG & style via [FBref (Série A)](https://fbref.com/); free fixtures via [football-data.org](https://www.football-data.org/coverage); squad values via [Transfermarkt/dcaribou](https://github.com/dcaribou/transfermarkt-datasets).
- **Cartões (cards) markets** are popular locally — referee-tendency features matter there, but samples are small and markets still efficient.
- **Altitude & travel** are first-class features in South America (e.g. La Paz ~3,600 m; long CONMEBOL trips) — see [McSharry BMJ](https://www.bmj.com/content/335/7633/1278).
- Brazilian datasets (results, gols, cartões): [adaoduque](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol) · [ricardomattos05](https://www.kaggle.com/datasets/ricardomattos05/jogos-do-campeonato-brasileiro).

---

## 12) Reality check — features are necessary, not sufficient

- The closing line already **prices in** almost every feature here (news, lineups, weather, congestion). Re-deriving it is not an edge.
- Most "edges" in backtests are **leakage or overfitting**; they vanish out-of-sample and after margin. Kaunitz et al. beat *published* odds in test, then **bookmakers limited/closed the winning accounts** ([arXiv 1710.02824](https://arxiv.org/abs/1710.02824)).
- Judge yourself by **CLV** and proper scoring (log-loss / Brier), not by a hot streak. Assume you will **not** beat sharp markets; build models to learn, not to fund a habit.

---

## 13) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious harm. This page exists for **data-science / ML research and education only**. Set limits, never chase losses (nunca persiga prejuízos), and seek help if betting stops being fun.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brazil | [SPA / Ministério da Fazenda — apostas de quota fixa (regulador)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/apostas-de-quota-fixa) | Apoio emocional **CVV 188** |

---

## Related in AIForge
- [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) — where every feature's data comes from.
- [Kaggle Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) · Parent vertical: [`../`](../) (Sports Analytics AI).

**Sources:** clubelo.com/API · eloratings.net · constantinou.info (pi-ratings, JQAS 2013) · github.com/fivethirtyeight/data (SPI; FiveThirtyEight discontinued 2023) · transfermarkt.com + github.com/dcaribou/transfermarkt-datasets · researchgate.net (Peeters 2018) · rss.onlinelibrary.wiley.com (Dixon–Coles 1997) · dashee87.github.io · fbref.com/en/expected-goals-model-explained · understat.com · karun.in/blog/expected-threat · dtai.cs.kuleuven.be/sports/vaep · github.com/ML-KULeuven/socceraction · mckayjohns.substack.com · link.springer.com (home-advantage review; congestion meta-analysis; travel study) · pmc.ncbi.nlm.nih.gov/PMC9758680 · sciencedaily.com · bmj.com/content/335/7633/1278 (McSharry 2007) · forebet.com · premierleague.com (PPDA; manager bounce) · socceranalytics.substack.com · footymetrics.com · pmc.ncbi.nlm.nih.gov/PMC7739605 (referee bias) · football-data.co.uk · oddsportal.com · penaltyblog.readthedocs.io · tradematesports.medium.com (Pinnacle closing-line efficiency) · arxiv.org/abs/1710.02824 · github.com/feast-dev/feast · github.com/alteryx/featuretools · hopsworks.ai · tecton.ai · github.com/probberechts/soccerdata · github.com/JaseZiv/worldfootballR · github.com/PySport/kloppy · ibm.com (data leakage) · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · gov.br/fazenda SPA/MF (apostas)

**Keywords:** football feature engineering, soccer match prediction features, Elo rating, pi-ratings, Soccer Power Index SPI, squad market value Transfermarkt, expected goals xG, npxG, xGD, expected threat xT, VAEP, expected points xPoints, rolling form, home advantage decline COVID, fixture congestion, rest days, altitude effect, PPDA pressing, set-piece xG, referee bias cards, closing line value, odds movement, market implied probability, data leakage, point-in-time correctness, walk-forward validation, feature store Feast, SHAP feature selection — engenharia de atributos futebol, previsão de partidas, diferença de Elo, valor de elenco, gols esperados, ameaça esperada, pontos esperados, fator casa, calendário congestionado, altitude, tendência de cartões do árbitro, variação das odds, probabilidade implícita, vazamento de dados, validação temporal, seleção de variáveis, jogo responsável.
