# In-Play, Advanced & ML/AI Approaches for Football Prediction

> Authoritative, densely-sourced index of the **cutting edge of football (soccer / futebol) prediction** — live/in-play win-probability, live xG & momentum, player-level & props valuation (VAEP/xT), Monte-Carlo season/tournament **simulation**, public "supercomputer" forecasters, deep/graph learning on event streams, and model+market ensembling — with real paper/repo/tool links, free-vs-paid marks, feature-engineering ideas, and honest benchmarking. **Research & education only**, Brazil-aware, current 2024–2026.

> ⚠️ **RESPONSIBLE GAMBLING — READ FIRST. This page is for research & education, NOT betting advice.** Football markets are **highly efficient**, and **in-play markets are even faster**: exchange odds re-price within seconds of a goal, and the built-in margin (*overround / vig*) plus book limits mean the *average* punter loses. Academic work on live markets finds only small, sporadic mispricings driven by longshot bias — not a repeatable edge ([Angelini, De Angelis & Singleton, *Int. J. Forecasting* 2022](https://www.sciencedirect.com/science/article/abs/pii/S0169207021000996)). **Most bettors lose money. Beating the closing line is extremely hard; beating it live is harder.** Nothing here is a tip or a system. If gambling stops being fun, get help — see the [Responsible-gambling resources](#responsible-gambling-resources-obrigatório) at the bottom. 🇧🇷 CVV **188**.

**Scope & how this page fits.** This is the *specialized / in-play* companion to the two sibling pages in this folder: model families and DL/GNN theory live in [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md); where to get data lives in [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) and [Kaggle Football Datasets](./Kaggle_Football_Datasets_and_Competitions.md). Here we focus on **what happens during the 90 minutes**, on **players and props**, on **simulation**, and on the **public forecasters** everyone benchmarks against.

---

## 1. In-play / live prediction (win-probability that updates on events)

An **in-game win-probability (WP)** model outputs P(home win / draw / away win) at every second, conditioning on the *game state*: score, minute, red cards, and (in richer models) live xG, shots, possession and field position. Football is the hardest major sport for this because it is **low-scoring and draw-heavy**, so a single goal swings WP violently and the model must be well-calibrated to be useful.

| Approach | Core idea | Example paper / tool | Link |
|---|---|---|---|
| **Bayesian in-game WP** ⭐ | Live win/draw/loss from score + time + red cards + context; tuned for football's low-scoring, draw-heavy nature | Robberechts, Van Haaren & Davis, *KDD* 2021 ("A Bayesian Approach to In-Game Win Probability in Soccer") | [arXiv:1906.05029](https://arxiv.org/abs/1906.05029) · [DTAI blog](https://dtai.cs.kuleuven.be/static/sports/blog/a-bayesian-approach-to-in-game-win-probability/) · [MLSA19 PDF](https://dtai.cs.kuleuven.be/events/MLSA19/papers/robberechts_MLSA19.pdf) |
| **Axial-transformer live forecaster** ⭐ | Recurrent transformer forecasting 13 player-action totals at match/team/player level; **~75,000 low-latency live predictions per game**, updating on each event | Horton & Lucey (Stats Perform), 2025 | [arXiv:2511.18730](https://arxiv.org/abs/2511.18730) |
| **Opta live win probability** | Commercial in-broadcast WP curve driven by live event feed + pre-match model | Stats Perform / The Analyst | [theanalyst.com](https://theanalyst.com/articles/opta-football-predictions) |
| **Interpretable WP (sport-agnostic)** | Simple, well-calibrated logistic/GBM WP framed for interpretability (NFL origin, portable pattern) | Pelechrinis, *iWinRNFL* | [arXiv:1704.00197](https://arxiv.org/pdf/1704.00197) |
| **Survival / hazard for next goal** | Model time-to-next-goal as a hazard process → live scoreline distribution | in-play goal-timing models | see socceraction / Dixon–Robinson lineage |

**Live xG & momentum (the two live features punters obsess over).** Sportsbooks increasingly surface **live xG** (accumulated shot quality, refreshed every couple of minutes) and a **momentum / pressure index** (rolling ~10-minute attacking intensity). These are *descriptive dashboards*, not proven edges.

| Live signal | What it measures | Source / tool | Link |
|---|---|---|---|
| **Live xG (xG ao vivo)** | Running sum of shot probabilities; "who deserves to be ahead" | Stats Perform / Opta; Infogol; Sportmonks xG API | [StatsPerform xG](https://www.statsperform.com/insights/expected-goals-xg-the-football-metric-changing-analysis-betting-and-fan-engagement/) · [Sportmonks xG](https://www.sportmonks.com/football-api/xg-data/) |
| **Attacking momentum** | Rolling attacking-pressure curve (last ~10 min) | Sofascore "Attack Momentum"; InPlayGuru Momentum | [Sofascore](https://www.sofascore.com/) · [InPlayGuru](https://inplayguru.com/guide/in-play-scanner/momentum) |
| **Live xGOT / xT** | Shot-on-target quality; live possession threat | Opta metrics (xGOT), socceraction xT | [socceraction xT](https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/xT.html) |

**Does momentum actually exist?** Evidence is **mixed and weak in soccer.** A widely-cited study found streaks in win-probability are non-random in **American football** ([Roebber et al., *PLOS ONE* 2022](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0269604)), but soccer-specific work is far less convincing, and betting *on* momentum is itself scrutinised ([*Gambling on Momentum*, Ötting et al., arXiv:2211.06052](https://arxiv.org/pdf/2211.06052)). Treat "momentum" as a **narrative/UX feature**, not a validated predictor.

**Honest in-play reality check.** Betting-exchange studies show live prices absorb the biggest news — the **first goal** — almost immediately; residual mispricing is small and explained by a **reverse favourite-longshot bias** (markets slightly underrate longshots after surprise goals), not a strategy you can farm ([Angelini, De Angelis & Singleton, *IJF* 38(1):282–299, 2022](https://centaur.reading.ac.uk/98329/1/information_efficiency_angelini_de_angelis_singleton.pdf)). Live margins are typically **wider** than pre-match, and winning accounts get limited — the edge, if any, is *smaller, faster, and more fragile* than pre-match.

---

## 2. Player-level valuation & props (xG/xA, VAEP, xT, ratings)

Player models rarely predict 1X2 directly; they are **feature factories** and the basis of **player-prop markets** (anytime goalscorer / *marcador a qualquer momento*, shots, shots-on-target, cards). Prop prices are largely a function of a player's **xG per 90**, shot volume, set-piece role, expected minutes, and opponent defensive quality.

| Model / metric | Idea | Example paper / tool | Link |
|---|---|---|---|
| **xG / xA (gols/assist. esperados)** | P(goal) per shot from location, angle, body part, pressure; xA = xG of the pass's resulting shot | Understat, Infogol, StatsBomb; improving-xG study (Mead et al.) | [understat.com](https://understat.com/) · [PLOS ONE 2023](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282295) |
| **Player/position-adjusted xG** | Adds shooter & positional skill to the base xG model | Hewitt & Karakuş, *Franklin Open* 2023 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2773186323000282) · [arXiv:2301.13052](https://arxiv.org/abs/2301.13052) |
| **VAEP / Atomic-VAEP** ⭐ | Value every on-ball action by Δ(scoring − conceding) probability; SPADL format | ML-KULeuven/socceraction | [GitHub](https://github.com/ML-KULeuven/socceraction) |
| **Expected Threat (xT)** ⭐ | Grid-based possession value: how much each ball location raises scoring chance | Karun Singh (2019) via socceraction | [xT docs](https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/xT.html) |
| **xT vs VAEP (critical comparison)** | Design choices change top-player rankings & action values — pick deliberately | Van Roy et al., *AAAI-20 AISA* workshop | [DTAI blog](https://dtai.cs.kuleuven.be/sports/blog/valuing-on-the-ball-actions-in-soccer-a-critical-comparison-of-xt-and-vaep/) |
| **Player ratings (Sofascore/WhoScored)** | ML rating per action, live; every player starts at 6.5, scale ~3–10 | Sofascore rating | [about/rating](https://corporate.sofascore.com/about/rating) |
| **Rating-system validity** | Peer-reviewed comparison of WhoScored/FotMob/Sofascore as performance metrics | Ball, Huynh & Varley, *J. Sports Sciences* 43(7), 2025 | [T&F](https://www.tandfonline.com/doi/full/10.1080/02640414.2025.2471208) |

**Props tooling (public examples, education only):**

| Tool | Market | Method | Link |
|---|---|---|---|
| **Squawka Goalscorer predictions** | Anytime scorer | xG-per-90 + role heuristics | [squawka.com](https://www.squawka.com/us/soccer-predictions/anytime-goalscorer/) |
| **ScoutingStats** | Goalscorer / shots / SOT | Player-prop probability model | [scoutingstats.ai](https://scoutingstats.ai/predictions/player-props) |
| **Dimers-style sims** | Multi-prop | 10,000-run match simulation → prop probabilities | [dimers.com](https://www.dimers.com/) |

**Injury / lineup impact.** Team-news is one of the few genuinely *pre-market* information sources — but it is priced fast. ML injury-risk research (workload, GPS, recovery) is mainly for **medical/tactical** use, not betting; the match-prediction value is via **availability features** (is the key player fit?).

| Angle | Idea | Example paper | Link |
|---|---|---|---|
| **GPS/workload injury risk** | Acute:chronic load, decelerations, minutes → injury probability | Saberisani et al., *Frontiers Sports & Active Living* 2025 (Iranian pros) | [frontiersin.org](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1425180/full) |
| **Interpretable (SHAP) risk** | Feature-attributed injury-risk model (university players) | *Sci. Reports* 2025 | [nature.com](https://www.nature.com/articles/s41598-025-24144-y) |
| **Decision-theoretic injury prediction** | Couples ML risk with decision utility for club use | medRxiv 2025 (FC Barcelona women) | [medrxiv](https://www.medrxiv.org/content/10.1101/2025.04.23.25326218v1) |

---

## 3. Simulation: season tables & tournament brackets (Monte Carlo)

To turn a **per-match** model into **season/tournament** probabilities (title, top-4, relegation, advance, win-it-all), you **Monte-Carlo** the remaining fixtures thousands of times and count finishing positions. This is exactly how the famous public "supercomputers" work.

| Tool / model | Idea | Free/Paid | Link |
|---|---|---|---|
| **Opta Supercomputer** ⭐ | Combines **Opta Power Rankings + market odds** → per-match probs → simulate remaining season 10,000×+ for final-table odds | **Free** (published) | [theanalyst.com](https://theanalyst.com/articles/who-will-win-the-premier-league-opta-supercomputer) |
| **FiveThirtyEight SPI** (historical) | Off-/def- ratings → match probs → season/WC sims; **models stopped updating June 2023**, data archived | **Free** (archive) | [github/soccer-spi](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) |
| **DTAI KU Leuven WC forecasts** | Academic Bayesian match model → tournament sim (2022, **2026**) | **Free** | [WC2026](https://dtai.cs.kuleuven.be/sports/worldcup2026/) |
| **Hicruben WC-2026 model** | Elo → Dixon–Coles bivariate-Poisson → 50,000-run Monte-Carlo of all 104 matches | **Free** (OSS, MIT) | [GitHub](https://github.com/Hicruben/world-cup-2026-prediction-model) |
| **lbenz730/world_cup_2026** | Bayesian bivariate-Poisson (intl results since 2016), 10,000 sims | **Free** (OSS) | [GitHub](https://github.com/lbenz730/world_cup_2026) |
| **goaliqlab/world-cup-2026-predictor** | Elo + XGBoost on 50,000+ internationals | **Free** (OSS) | [GitHub](https://github.com/goaliqlab/world-cup-2026-predictor) |
| **Probabilistic PL simulator** (tutorial) | Hybrid market–model probs → Monte-Carlo remaining season | **Free** | [Medium](https://medium.com/@vickyfrissdekereki/building-a-probabilistic-premier-league-simulator-in-python-34b5248f81b9) |
| **TacticsBadger / timseed** | Minimal MC match/season sims (PL, relegation-points) | **Free** (OSS) | [TacticsBadger](https://github.com/TacticsBadger/MonteCarloFootballMatchSim) · [timseed](https://github.com/timseed/Monte_Carlo_Football) |

**Reality:** back-tested MC tournament models cluster around **multiclass Brier ≈ 0.57–0.58, ~55–58% match accuracy** — i.e. *bookmaker-grade, no free lunch.* Simulations are best for **narrative/expectation** (xPTS, title odds), not for beating the market.

---

## 4. Public forecasters & rating systems (what to benchmark against)

Before building anything, compare to these **free public models**. If you can't beat a good Elo or the Opta table, your model has analytical value only.

| Forecaster | What it gives | Free/Paid | Link |
|---|---|---|---|
| **ClubElo** ⭐ | Daily club Elo + implied win/draw/loss probs; European clubs since 1939; CSV **API** | **Free** | [clubelo.com](http://clubelo.com/) · [API](http://clubelo.com/API) |
| **World Football Elo Ratings** | National-team Elo (margin, importance, home adv.) | **Free** | [eloratings.net](https://eloratings.net/) |
| **Opta Power Rankings / Analyst** | Global club power ranking + match & season predictions | **Free** (published) | [theanalyst.com](https://theanalyst.com/articles/opta-football-predictions) |
| **Infogol** | Opta-powered **xG** model, match predictions & O/U tips (incl. 🇧🇷 Brasileirão) | **Free**/premium | [via Timeform](https://www.timeform.com/football) |
| **Experimental 361** (Ben Mayhew) | Free viz/analysis, PL & EFL | **Free** | [experimental361.com](https://www.experimental361.com/) |
| **FiveThirtyEight SPI** (archive) | Historical SPI ratings + match/season forecasts (retired 2023) | **Free** | [GitHub](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) |
| **tonyelhabr/club-rankings** | Tidy historical **Opta + ClubElo** daily rankings (reproducible) | **Free** (OSS) | [GitHub](https://github.com/tonyelhabr/club-rankings) |

---

## 5. Deep & graph learning on the live stream (frontier — see sibling for depth)

The heavy DL/GNN/transformer theory is catalogued in the [sibling deep-learning page](./Innovative_Models_and_Deep_Learning.md); here are the angles that matter **for in-play & valuation** specifically.

| Approach | In-play/valuation angle | Example | Link |
|---|---|---|---|
| **Axial / recurrent transformers** | Live, multi-level (match/team/player) forecasts updating per event | Horton & Lucey 2025 | [arXiv:2511.18730](https://arxiv.org/abs/2511.18730) |
| **Sequence "language of soccer"** | Next-event prediction → simulate remainder from any game state | Seq2Event (Simpson et al., *KDD* 2022) | [ACM Seq2Event](https://dl.acm.org/doi/10.1145/3534678.3539138) |
| **GNN on player-interaction graphs** | Embed *how* a team plays as live features | see sibling §4 | [Innovative Models](./Innovative_Models_and_Deep_Learning.md) |
| **TacticAI** (DeepMind × Liverpool) | Geometric GNN for **set-piece** (corner) receiver/shot prediction & suggestions | Wang et al., *Nat. Commun.* 2024 | [Nature](https://www.nature.com/articles/s41467-024-45965-x) |

---

## 6. Market & ensemble: blending your model with the odds

The most reliable real-world gains come **not** from an exotic model but from **blending model probabilities with de-vigged market odds**, then **calibrating** — the market supplies calibration, your model supplies residual signal. This is precisely the **hybrid market–model** design the Opta supercomputer and public simulators use.

| Technique | Idea | Link |
|---|---|---|
| **Model + market blend** | Weighted avg of your probs and de-vigged closing odds; markets dominate, model nudges | [PL simulator (hybrid)](https://medium.com/@vickyfrissdekereki/building-a-probabilistic-premier-league-simulator-in-python-34b5248f81b9) |
| **Stacking / meta-learner** | Base models (Poisson, Elo, XGBoost, LSTM) → meta-model → final 1X2 (see sibling §9) | [Innovative Models](./Innovative_Models_and_Deep_Learning.md) |
| **Calibration (Platt / isotonic)** | Fix probability distortion before any staking decision | see sibling §9 |
| **Closing Line Value (CLV)** | The only honest skill test: do your prices repeatedly beat the close? | [Pinnacle: CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting) |

---

## 7. Feature-engineering ideas (what actually moves the needle)

Most predictive power in tabular models comes from a **handful of well-built features**, not model exotica. Ratings + form + rest dominate; weather/referee are **small, noisy** effects.

| Feature | Idea / signal | Evidence / tool | Link |
|---|---|---|---|
| **Elo / rating difference** ⭐ | Dynamic team strength; the single strongest tabular feature | Hvattum & Arntzen, *IJF* 26(3):460–470, 2010 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001708) |
| **pi-ratings** | Football-specific, goal-difference-aware, zero-centred ratings | Constantinou & Fenton; penaltyblog | [pi docs](https://penaltyblog.readthedocs.io/en/latest/ratings/pi.html) |
| **Form / rolling xG** | Recent results & underlying xG over last N matches | Understat / FBref | [understat.com](https://understat.com/) |
| **Home advantage** | Real but shrinking; empty stadia (COVID) reduced it | Wang & Qin, COVID crowd-effect review (*PLOS ONE* 2023) | [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0289899) |
| **Referee bias** | Individual refs give measurably different home bias (cards/penalties), partly crowd-linked | Boyko et al., Premiership referee study (*J. Sports Sciences* 2007) | [T&F](https://www.tandfonline.com/doi/full/10.1080/02640410601038576) |
| **Rest days / fixture congestion** | Congested schedules have **limited** effect on total distance; alter pacing/high-intensity output (this meta-analysis excludes injury) | Julian, Page & Harper, *Sports Medicine* meta-analysis 2020 | [PMC7846542](https://pmc.ncbi.nlm.nih.gov/articles/PMC7846542/) |
| **Travel distance** | Long/cross-continental travel + congestion strain squads | FIFPRO workload context | [ISSPF](https://www.isspf.com/articles/fixture-congestion-in-football/) |
| **Weather** | Temperature/humidity/wind measurably shift *technical* output (e.g. fewer counter-attack shots in heat; more cards per foul) — effects small but real | Zhong et al., *Biology of Sport* 2024 (UCL matches) | [PMC11474995](https://pmc.ncbi.nlm.nih.gov/articles/PMC11474995/) |

---

## 8. How to evaluate honestly (leia isto / read this)

1. **Calibration over accuracy** — for live WP and props, plot reliability curves / ECE; a confident-but-miscalibrated model loses money even at "high accuracy."
2. **RPS + log-loss, time-ordered** — rank-aware **Ranked Probability Score** is the football standard; always test train-past → predict-future (no season shuffling / leakage).
3. **Beat the *live* close, not the open** — in-play, the honest metric is CLV against the **next** price after your bet; live overrounds are wider than pre-match.
4. **First-goal shock** — most live "edges" are just slow reaction to a goal that the exchange has already priced; verify against tick data, not intuition.
5. **Expect tiny gains** — peer-reviewed work says advanced/in-play models beat a good Elo/Dixon–Coles + market by **little or nothing in profit terms**. The clear DL wins are **analytical** (tactics, valuation, scouting), not betting.

---

## 🇧🇷 Brazil & regional notes

- **xG ao vivo do Brasileirão:** free live/aggregated xG for **Série A/B** at [FootyStats](https://footystats.org/brazil/serie-a/xg), [xGscore](https://xgscore.io/xg-statistics/brazil-serie-a), [OddAlerts](https://www.oddalerts.com/xg/serie-a-brazil) and [FBref](https://fbref.com/en/comps/24/Serie-A-Stats); **Infogol** covers Brasil Série A with the same Opta xG engine used for Europe.
- **Simulação:** the World-Cup 2026 simulators above (Hicruben, lbenz730, goaliqlab, DTAI) all rate 🇧🇷 Brazil as a title contender; the code is open for re-running with your own priors.
- **Regulação (regulation):** Brazil's fixed-odds market (incl. live/*ao vivo* betting) is regulated under **Lei 14.790/2023** (in force since 1 Jan 2025) via the **Secretaria de Prêmios e Apostas (SPA/MF)** — mandatory self-limits, CPF/face verification, no credit-card funding, no sign-up bonuses. In-play products face the same duty-of-care rules.

---

## Responsible-gambling resources (OBRIGATÓRIO)

**Gambling is not an investment strategy. In-play is faster, more immersive, and higher-risk. The house edge is mathematical and permanent. If you or someone you know is struggling, reach out now — help is free and confidential.**

| Resource | Region | Contact |
|---|---|---|
| **GamCare** — National Gambling Helpline | 🇬🇧 UK (global info) | [gamcare.org.uk](https://www.gamcare.org.uk/) · **0808 8020 133** (24/7) |
| **BeGambleAware / National Gambling Support Network** | 🇬🇧 UK | [begambleaware.org](https://www.begambleaware.org/) |
| **Gambling Therapy** | 🌍 Worldwide (multilingual) | [gamblingtherapy.org](https://www.gamblingtherapy.org/) |
| **🇧🇷 Jogo Responsável — SPA/MF** | 🇧🇷 Brazil | [gov.br/fazenda (SPA)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) |
| **🇧🇷 IBJR — Instituto Brasileiro de Jogo Responsável** | 🇧🇷 Brazil | [ibjr.org.br](https://ibjr.org.br/en/responsible-gaming/) |
| **🇧🇷 CVV — apoio emocional** | 🇧🇷 Brazil | **188** · [cvv.org.br](https://www.cvv.org.br/) |

---

## Related in AIForge
- [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** arxiv.org/abs/1906.05029 (Robberechts et al., Bayesian in-game WP, KDD 2021) · dtai.cs.kuleuven.be (in-game WP blog, WC2026, xT vs VAEP) · arxiv.org/abs/2511.18730 (Horton & Lucey axial transformer) · arxiv.org/abs/1704.00197 (Pelechrinis iWinRNFL) · theanalyst.com (Opta predictions & supercomputer) · sciencedirect.com/S0169207021000996 & centaur.reading.ac.uk/98329 (Angelini/De Angelis/Singleton, IJF 38(1):282–299, 2022) · journals.plos.org/pone.0269604 (Roebber et al., momentum in NFL) · arxiv.org/abs/2211.06052 (Ötting et al., Gambling on Momentum) · github.com/ML-KULeuven/socceraction · socceraction.readthedocs.io (xT) · journals.plos.org/pone.0282295 (Mead et al., xG) & sciencedirect.com/S2773186323000282 + arxiv.org/abs/2301.13052 (Hewitt & Karakuş, player/position xG) · corporate.sofascore.com/about/rating · tandfonline.com/10.1080/02640414.2025.2471208 (Ball et al., rating systems, JSS 43(7) 2025) · squawka.com · scoutingstats.ai · dimers.com · frontiersin.org/fspor.2025.1425180 · nature.com/s41598-025-24144-y · medrxiv 2025.04.23.25326218 (injury) · github.com/fivethirtyeight/data/soccer-spi · github.com/Hicruben/world-cup-2026-prediction-model · github.com/lbenz730/world_cup_2026 · github.com/goaliqlab/world-cup-2026-predictor · github.com/TacticsBadger/MonteCarloFootballMatchSim · github.com/timseed/Monte_Carlo_Football · medium.com (Friss de Kereki, probabilistic PL simulator) · clubelo.com & clubelo.com/API · eloratings.net · experimental361.com · timeform.com/football (Infogol) · github.com/tonyelhabr/club-rankings · dl.acm.org/10.1145/3534678.3539138 (Simpson et al., Seq2Event) · nature.com/s41467-024-45965-x (Wang et al., TacticAI) · pinnacle.com (CLV) · sciencedirect.com/S0169207009001708 (Hvattum & Arntzen Elo, IJF 26(3) 2010) · penaltyblog.readthedocs.io (pi-ratings) · journals.plos.org/pone.0289899 (Wang & Qin, crowd/home adv) · tandfonline.com/10.1080/02640410601038576 (Boyko et al., referee bias) · pmc.ncbi.nlm.nih.gov/PMC7846542 (Julian et al., fixture congestion) · isspf.com · pmc.ncbi.nlm.nih.gov/PMC11474995 (Zhong et al., weather) · statsperform.com · sportmonks.com · sofascore.com · inplayguru.com · footystats.org · xgscore.io · oddalerts.com · fbref.com · gov.br/fazenda (SPA) · gamcare.org.uk · begambleaware.org · gamblingtherapy.org · ibjr.org.br · cvv.org.br

**Keywords:** in-play football prediction, live win probability probabilidade de vitória ao vivo, live xG gols esperados ao vivo, attacking momentum momentum, VAEP expected threat xT socceraction, player rating Sofascore, anytime goalscorer props marcador a qualquer momento, injury prediction lesão previsão, Monte Carlo season simulation simulação de temporada, Opta supercomputer FiveThirtyEight SPI ClubElo Elo pi-ratings, World Cup 2026 simulation Copa do Mundo 2026, axial transformer sequence model Seq2Event, TacticAI GNN set pieces bola parada, model market blend closing line value CLV eficiência de mercado, home advantage referee bias fixture congestion weather clima, ranked probability score Brier calibração, Brasileirão xG Infogol, responsible gambling jogo responsável.
