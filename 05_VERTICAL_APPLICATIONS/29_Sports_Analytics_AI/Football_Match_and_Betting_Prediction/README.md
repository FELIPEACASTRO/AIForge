# ⚽ Football (Soccer) Match & Betting Prediction

> The most complete open index of **prediction for football matches & betting** — models, features, statistics/probability, datasets & APIs (worldwide), odds & market theory, tools, research, and evaluation. **31 fact-checked pages**: every dataset, repo, paper, and API was verified to exist with a working link.

> ⚠️ **Research & education only — not betting advice.** Betting markets are highly efficient; **most bettors lose money** over time and beating the closing line is very hard. Nothing here recommends betting. If gambling is a problem, seek help: [BeGambleAware](https://www.begambleaware.org/), [GamCare](https://www.gamcare.org.uk/), [Gambling Therapy](https://www.gamblingtherapy.org/) (Gordon Moody); 🇧🇷 [Jogo Responsável](https://www.gov.br/fazenda/pt-br/assuntos/loterias/jogo-responsavel), CVV 188.

## 🧠 Models, features & theory
| Page | What's inside |
|---|---|
| [Match Prediction Models & Techniques](./Match_Prediction_Models_and_Techniques.md) | Poisson / **Dixon-Coles** / Maher, bivariate Poisson, Bayesian hierarchical, Elo / **pi-ratings** / SPI, xG-based, ML (XGBoost/CatBoost), deep learning; targets (1X2, O/U, BTTS, correct score, Asian handicap). |
| [Features & Feature Engineering](./Features_and_Feature_Engineering.md) | Elo diff, form/momentum, xG/xT rolling, home advantage, rest/congestion, team news, H2H, referee, market features — with no-leakage discipline. |
| [Statistics & Probability Foundations](./Statistics_and_Probability_Foundations.md) | Poisson/bivariate, Dixon-Coles τ & time-decay, Elo/pi math, Bayesian/MCMC, odds↔probability (overround/Shin/power), RPS/Brier/log-loss, Kelly & risk of ruin. |
| [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) | GBMs, Bayesian networks (dolores), GNNs, transformers, **DeepMind TacticAI**, Google Research Football, LLMs, ensembles & calibration. |
| [Model Evaluation, Calibration & Betting Validation](./Model_Evaluation_Calibration_and_Betting_Validation.md) | RPS/Brier/log-loss, reliability diagrams, Platt/isotonic, **closing line value (CLV)** as the edge benchmark, walk-forward, backtest pitfalls. |
| [Simulation & Tournament Forecasting Engines](./Simulation_and_Tournament_Forecasting_Engines.md) | Monte Carlo season/tournament sims, SPI (archived), Opta supercomputer, ClubElo, World-Cup/Euro simulators. |

## 🌍 Data, odds & markets
| Page | What's inside |
|---|---|
| [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) | football-data.co.uk, **StatsBomb Open Data**, Understat/FBref, API-Football, Football-Data.org, SofaScore, Transfermarkt, OpenFootball. |
| [Kaggle — Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) | European Soccer DB, Beat the Bookie (odds), Transfermarkt, Brasileirão, top-5 leagues, World Cup 1930–2026 (pulled live via API). |
| [Data Providers, Tracking & Computer Vision](./Data_Providers_Tracking_and_Computer_Vision.md) | Opta/Stats Perform, StatsBomb (Hudl), Wyscout, SkillCorner, PFF, Metrica open data, **SoccerNet** (CV benchmark). |
| [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) | Implied probability/overround, market efficiency & **CLV**, The Odds API/Betfair/Pinnacle, Kelly/bankroll — honest framing. |
| [Odds Aggregators & Historical Odds Archives](./Odds_Aggregators_and_Historical_Odds_Archives.md) | OddsPortal, BetExplorer, The Odds API, **football-data.co.uk historical odds** (for backtesting), Betfair historical, scraping repos. |
| [News, Sentiment & Alternative Data](./News_Sentiment_and_Alternative_Data.md) | Google Trends/pytrends, "fade the public", weather (Meteostat/NOAA), altitude effects, GDELT — noisy & largely priced-in (honest). |

## 🌏 Regional deep-dives (all countries)
| Page | What's inside |
|---|---|
| [Europe — Data, Leagues & Markets](./Europe_Football_Data_and_Markets.md) | Top-5 + lower divisions + all UEFA; football-data.co.uk, ClubElo, Understat/FBref, Opta. |
| [Americas — Data, Leagues & Markets](./Americas_Football_Data_and_Markets.md) | 🇧🇷 Brasileirão/CONMEBOL, MLS, Liga MX, Argentina; American Soccer Analysis, FBref, Lei das Bets. |
| [Asia & India — Data, Leagues & Markets](./Asia_and_India_Football_Data_and_Markets.md) | J-League, K-League, CSL, **ISL/India**, Saudi Pro League, AFC + origin of the **Asian handicap**. |
| [Africa, Middle East & Oceania](./Africa_MiddleEast_and_Oceania_Football_Data.md) | PSL, Botola, NPFL, CAF; Saudi/UAE/Iran; A-League — data sources (scarcity noted). |
| [Women's & Lower-Tier Football](./Womens_and_Lower_Tier_Football_Data.md) | WSL/NWSL/UWCL (StatsBomb free women's data), lower divisions — less-efficient markets & honest tradeoffs. |

## 🎯 Betting strategy, markets & specialised
| Page | What's inside |
|---|---|
| [Bet Selection, Staking & High-Odds ("Apostas Bomba")](./Bet_Selection_Staking_and_High_Odds_Analysis.md) | Value/edge, CLV, devigging, Kelly & fractional Kelly; **honest -EV debunk** of accumulators/longshots. |
| [Betting Exchanges, Trading & Microstructure](./Betting_Exchanges_Trading_and_Microstructure.md) | Betfair Exchange (back/lay), betfairlightweight/flumine, Smarkets; in-play trading, liquidity, matched betting (educational). |
| [Corners, Cards & Secondary-Markets Modeling](./Corners_Cards_and_Secondary_Markets_Modeling.md) | Over/under corners, cards, fouls; BTTS & total goals; player props. |
| [Referee Analytics & Discipline Data](./Referee_Analytics_and_Discipline_Data.md) | Referee card/penalty tendencies, home bias, VAR effect — for booking markets. |
| [Injury, Lineup & Availability Data](./Injury_Lineup_and_Availability_Data.md) | Predicted/confirmed lineups, injury feeds, suspensions; key-player-absence impact (pre-kickoff). |
| [Fantasy Football (FPL) Analytics](./Fantasy_Football_and_FPL_Analytics.md) | Official FPL API, vaastav historical dataset, xP models, LP optimizers; Dream11/Sorare. |

## 🛠️ Tools, community & research
| Page | What's inside |
|---|---|
| [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) | **penaltyblog** (Dixon-Coles), **socceraction** (VAEP), kloppy, mplsoccer, soccerdata, statsbombpy + snippets. |
| [Community, Forums, Newsletters & Open Projects](./Community_Forums_Newsletters_and_Open_Projects.md) | r/algobetting, r/footballanalytics, Soccermatics, StatsBomb blog, ASA, Buchdahl; verified open repos + books. |
| [In-Play, Advanced & ML/AI Approaches](./In_Play_Advanced_and_ML_Approaches.md) | Live win-probability, xT/VAEP, GNNs, TacticAI, LLMs. |
| [Key Papers & Research](./Key_Papers_and_Research.md) | Maher 1982, **Dixon-Coles 1997**, Rue-Salvesen, Karlis-Ntzoufras, Baio-Blangiardo, Constantinou (pi-football/dolores) — verified DOIs. |
| [Prediction Benchmarks & Open Challenges](./Prediction_Benchmarks_and_Open_Challenges.md) | **2017 Soccer Prediction Challenge** (Berrar/Lasek/Dubitzky), Open International Soccer DB, "beat the bookmaker" studies. |
| [Academic Preprints & Working Papers](./Academic_Preprints_and_Working_Papers.md) | arXiv (stat.AP/cs.LG), **SportRxiv**, SSRN, OSF, PsyArXiv, RePEc/NBER/MPRA, 🇧🇷 SciELO Preprints — how to search each. |
| [Scholarly Search Engines & Aggregators](./Scholarly_Search_Engines_and_Aggregators.md) | Semantic Scholar/OpenAlex (+APIs), CORE, BASE, DBLP, Papers with Code, Crossref/Unpaywall — literature pipeline. |
| [Betting-Market Efficiency Literature](./Betting_Market_Efficiency_Literature.md) | Favourite-longshot bias, Pope & Peel, Forrest-Goddard-Simmons, **Levitt (2004)**, Constantinou-Fenton — verified papers. |

## Related in AIForge
- Parent vertical: [`../`](../) (Sports Analytics AI)
- Fundamentals: [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/) · [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Bayesian_and_Probabilistic_ML/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Bayesian_and_Probabilistic_ML/) · [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/)

**Keywords:** football prediction, soccer prediction, betting model, Dixon-Coles, expected goals xG, Poisson model, Elo pi-ratings, value betting, closing line value, RPS, football datasets, Asian handicap, FPL, betting market efficiency, previsão de futebol, modelo de apostas, aposta de valor, mercado de apostas.
