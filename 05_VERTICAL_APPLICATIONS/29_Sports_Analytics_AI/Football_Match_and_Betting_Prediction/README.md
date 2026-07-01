# ⚽ Football (Soccer) Match & Betting Prediction

> The most complete open index of **prediction for football matches & betting** — models, features, statistics/probability, datasets & APIs (worldwide), odds & value theory, tools, and research. Every page is fact-checked (each dataset/repo/paper/API verified to exist with a working link).

> ⚠️ **Research & education only — not betting advice.** Betting markets are highly efficient; **most bettors lose money** over time and beating the closing line is very hard. Nothing here is a recommendation to bet. If gambling is a problem, seek help: [BeGambleAware](https://www.begambleaware.org/), [GamCare](https://www.gamcare.org.uk/), [Gambling Therapy](https://www.gamblingtherapy.org/) (Gordon Moody); 🇧🇷 [Jogo Responsável](https://www.gov.br/fazenda/pt-br/assuntos/loterias/jogo-responsavel), CVV 188.

## 🧠 Models, features & theory
| Page | What's inside |
|---|---|
| [Match Prediction Models & Techniques](./Match_Prediction_Models_and_Techniques.md) | Poisson / **Dixon-Coles** / Maher, bivariate Poisson, Bayesian hierarchical, Elo / **pi-ratings** / SPI, xG-based, ML (XGBoost/CatBoost), deep learning; targets (1X2, O/U, BTTS, correct score, Asian handicap); evaluation (RPS/Brier/log-loss). |
| [Features & Feature Engineering](./Features_and_Feature_Engineering.md) | Elo diff, form/momentum, xG/xT rolling, home advantage, rest/congestion, team news, H2H, referee, market features — with point-in-time / no-leakage discipline. |
| [Statistics & Probability Foundations](./Statistics_and_Probability_Foundations.md) | Poisson/bivariate, Dixon-Coles τ & time-decay, Elo/pi math, Bayesian/MCMC, odds↔probability (overround, Shin, power), RPS/Brier/log-loss, Kelly & risk of ruin. |
| [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) | GBMs, Bayesian networks (dolores), GNNs, transformers, **DeepMind TacticAI**, Google Research Football, LLMs, ensembles & calibration. |

## 🌍 Data & odds
| Page | What's inside |
|---|---|
| [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) | football-data.co.uk, **StatsBomb Open Data**, Understat/FBref, API-Football, Football-Data.org, SofaScore, Transfermarkt, OpenFootball + regional coverage. |
| [Kaggle — Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) | European Soccer DB, Beat the Bookie (odds), Transfermarkt, Brasileirão, top-5 leagues, World Cup 1930–2026 (pulled live via API). |
| [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) | Implied probability/overround, market efficiency & **closing line value (CLV)**, The Odds API/Betfair/Pinnacle, Kelly/bankroll — honest, research-only framing. |
| [Africa, Middle East & Oceania](./Africa_MiddleEast_and_Oceania_Football_Data.md) | PSL, Botola, NPFL, CAF; Saudi Pro League, UAE, Iran; A-League — data sources (data-scarcity noted). |

## 🛠️ Tools & research
| Page | What's inside |
|---|---|
| [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) | **penaltyblog** (Dixon-Coles), **socceraction** (VAEP), kloppy, mplsoccer, soccerdata, statsbombpy + code snippets. |
| [In-Play, Advanced & ML/AI Approaches](./In_Play_Advanced_and_ML_Approaches.md) | Live win-probability, xT/VAEP, Monte Carlo sims, SPI/ClubElo/Opta, GNNs, TacticAI, LLMs. |
| [Key Papers & Research](./Key_Papers_and_Research.md) | Maher 1982, **Dixon-Coles 1997**, Rue-Salvesen, Karlis-Ntzoufras, Baio-Blangiardo, Constantinou (pi-football/dolores), ML surveys — with verified DOIs. |

## 🌏 Regional & market deep-dives
| Page | What's inside |
|---|---|
| [Asia & India — Data, Leagues & Markets](./Asia_and_India_Football_Data_and_Markets.md) | J-League, K-League, CSL, **ISL/India**, Saudi Pro League, AFC + the origin of the **Asian handicap**; data sources per country. |
| [Europe — Data, Leagues & Markets](./Europe_Football_Data_and_Markets.md) | Top-5 + lower divisions + all UEFA nations; **football-data.co.uk**, ClubElo, Understat/FBref, Opta — the deepest, most-modelled region. |
| [Bet Selection, Staking & High-Odds ("Apostas Bomba")](./Bet_Selection_Staking_and_High_Odds_Analysis.md) | Value/edge, CLV, devigging, Kelly & fractional Kelly; an **honest, evidence-based debunk** of accumulators/longshots (margin compounds → usually −EV). |
| [Betting Exchanges, Trading & Microstructure](./Betting_Exchanges_Trading_and_Microstructure.md) | Betfair Exchange (back/lay), betfairlightweight/flumine, Smarkets; in-play trading, liquidity, matched betting (educational). |
| [Corners, Cards & Secondary-Markets Modeling](./Corners_Cards_and_Secondary_Markets_Modeling.md) | Over/under corners, cards, fouls; BTTS & total-goals; player props — modeling & data for "secondary" markets. |
| [Fantasy Football (FPL) Analytics](./Fantasy_Football_and_FPL_Analytics.md) | Official FPL API, vaastav historical dataset, xP models, LP optimizers; Dream11/Sorare context. |
| [Injury, Lineup & Player-Availability Data](./Injury_Lineup_and_Availability_Data.md) | Predicted/confirmed lineups, injury feeds, suspensions; impact of key-player absence (pre-kickoff timing). |
| [Women's & Lower-Tier Football](./Womens_and_Lower_Tier_Football_Data.md) | WSL/NWSL/UWCL (StatsBomb free women's data), lower divisions — less-efficient markets & the honest tradeoffs. |

> 🔜 A few more triple-checked pages are still finalizing (Americas regional deep-dive; community & open projects; data providers/tracking/CV; referees; sentiment/alt-data; odds archives; simulation engines; benchmarks; evaluation/calibration; academic preprints & aggregators) and will be added.

## Related in AIForge
- Parent vertical: [`../`](../) (Sports Analytics AI)
- Fundamentals: [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/) · [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Bayesian_and_Probabilistic_ML/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Bayesian_and_Probabilistic_ML/) · [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/)

**Keywords:** football prediction, soccer prediction, betting model, Dixon-Coles, expected goals xG, Poisson model, Elo pi-ratings, value betting, closing line value, RPS, football datasets, previsão de futebol, modelo de apostas, aposta de valor.
