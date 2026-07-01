# Innovative Models & Deep Learning for Football

> Authoritative, densely-sourced index of **cutting-edge models (2020–2026) for predicting football (soccer / futebol) matches** — gradient boosting, Bayesian hierarchical Poisson, deep learning, graph neural networks, transformers/foundation models, LLMs, and simulation/RL — with real paper/repo/API links, free-vs-paid marks, and honest benchmarking against the classical **Dixon–Coles + market** baseline. **Research & education only**, Brazil-aware, current 2024–2026.

> ⚠️ **RESPONSIBLE GAMBLING — READ FIRST. This page is for research & education, NOT betting advice.** Football betting markets are **highly efficient**: the bookmaker **closing line** is close to the best available forecast, and the built-in margin (*overround / vig*, typically ~5–7% on 1X2 markets) means the *average* punter loses — bet all three outcomes and you simply hand the bookmaker the overround. Academic work finds inefficiencies exist only sporadically and are **not persistent or systematic** across leagues ([Winkelmann et al., 2024, *J. Sports Economics*](https://journals.sagepub.com/doi/10.1177/15270025231204997)); even a carefully calibrated model is generally out-calibrated by the bookmaker's own closing odds ([Wilkens, 2026, *J. Sports Analytics*](https://journals.sagepub.com/doi/10.1177/22150218261416681)). **Most bettors lose money. Sustained edges are rare, small, and hard to keep.** Nothing here is a tip. If gambling stops being fun, get help — see the [Responsible-gambling resources](#responsible-gambling-resources-obrigatório) at the bottom. 🇧🇷 CVV **188**.

---

## 0. The baseline you must beat (why "innovative" ≠ "better")

Before any deep net, know the classical benchmark. Almost every 2020–2026 paper that claims an edge is measured against these — and the honest finding is that **marginal gains over a well-tuned Dixon–Coles / bivariate-Poisson model blended with market odds are small.**

| Baseline | Idea | Reference / repo |
|---|---|---|
| **Poisson (independent)** | Goals ~ Poisson(attack×defense×home) | classic Maher (1982) |
| **Dixon–Coles (1997)** | Two Poisson goal counts with a low-score dependence correction (ρ) + time-decayed MLE; corrects Poisson's under-count of 0–0/1–1 | [Dixon & Coles, *JRSS-C*](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9876.00065) · tutorial [dashee87](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/) |
| **Bayesian hierarchical Poisson** | Attack/defense as shared random effects (partial pooling) | [Baio & Blangiardo, 2010](http://discovery.ucl.ac.uk/16040/) · [PyMC tutorial](https://pena.lt/y/2021/08/25/predicting-football-results-using-bayesian-statistics-with-python-and-pymc3/) |
| **Market (closing odds)** | De-vig the bookmaker's closing 1X2 → probabilities | [football-data.co.uk](https://www.football-data.co.uk/data.php) (free CSV odds) |

**Rule of thumb:** if a fancy model can't beat de-vigged **closing** odds on **Ranked Probability Score (RPS)** and log-loss out-of-sample, it has no betting value — only analytical value.

---

## 1. Gradient boosting (the practical SOTA for 1X2 & goals)

XGBoost / LightGBM / CatBoost on engineered features (ratings, form, rest, xG aggregates) are the strongest *tabular* approach and repeatedly top public benchmarks. **CatBoost tends to give the best-calibrated 1X2 probabilities out of the box.**

| Model / study | Idea | Example paper / repo | Link |
|---|---|---|---|
| **CatBoost + pi-ratings** | Boosted trees on rating features; best-reported **RPS 0.1925**, beating all 2017 Soccer Prediction Challenge entries | Bunker, Yeung & Fujii survey | [arXiv:2403.07669](https://arxiv.org/abs/2403.07669) |
| **GBT + DL feature study** | Feature optimisation for gradient-boosted trees vs deep nets on match outcome | Evaluating Soccer Match Prediction Models | [arXiv:2309.14807](https://arxiv.org/abs/2309.14807) |
| **XGBoost for xG / goals** | Boosting on shot features; SHAP shows angle & GK distance dominate | Expected Goals w/ XGBoost (ESP JETA 2023) | [open PDF](https://www.espjeta.org/Volume3-Issue1/JETA-V3I1P104.pdf) |
| **Poisson vs ML head-to-head** | ML (RF/XGBoost) vs Poisson on identical features — gains are modest | Match predictions in soccer | [arXiv:2408.08331](https://arxiv.org/abs/2408.08331) |
| **penaltyblog** | Production Poisson/Dixon–Coles + rating + boosting pipeline (Cython) | martineastwood/penaltyblog | [GitHub](https://github.com/martineastwood/penaltyblog) |

**Ordinal & multi-output tricks:** frame 1X2 as **ordinal** (loss < draw < win) or predict the full **score matrix** (independent/bivariate Poisson heads, or a softmax over 0–0…N–N) and marginalise to 1X2 / over-under / BTTS. Score-grid models give you *all* markets from one fit.

---

## 2. Bayesian models (uncertainty + hierarchy + priors)

Bayesian methods shine where data is thin (new season, lower leagues) and when you need **honest predictive intervals** rather than point estimates.

| Model | Idea | Example paper / repo | Link |
|---|---|---|---|
| **Hierarchical Poisson** | Partial pooling of attack/defense; home effect | Baio & Blangiardo 2010 | [UCL](http://discovery.ucl.ac.uk/16040/) · [PyMC](https://pena.lt/y/2021/08/25/predicting-football-results-using-bayesian-statistics-with-python-and-pymc3/) |
| **pi-football (Bayesian network)** | Expert-informed BN forecasting EPL outcomes | Constantinou, Fenton & Neil, *Knowledge-Based Systems* 2012 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705112001967) |
| **Dolores** | Dynamic ratings + hybrid Bayesian networks; predicts matches in leagues the teams never played in; 2nd in 2017 ML Challenge | Constantinou, *Machine Learning* 108:49–75 (2019) | [Springer](https://link.springer.com/article/10.1007/s10994-018-5703-7) |
| **Ratings + BN for Asian handicap** | Tests Asian-handicap market efficiency with ratings & BNs | Constantinou, *J. Sports Analytics* 2022 | [arXiv:2003.09384](https://arxiv.org/abs/2003.09384) |
| **Dynamic / state-space skill** | Online skill-rating (Elo/TrueSkill/Kalman) as Bayesian filtering | State-space skill rating | [arXiv:2308.02414](https://arxiv.org/abs/2308.02414) |
| **In-game win probability** | Bayesian live win-prob from score/time state | A Bayesian Approach to In-Game Win Probability | [arXiv:1906.05029](https://arxiv.org/abs/1906.05029) |

**Tooling:** [PyMC](https://www.pymc.io/) (see the [rugby hierarchical example](https://www.pymc.io/projects/examples/en/latest/case_studies/rugby_analytics.html), directly adaptable to football) and [Stan](https://mc-stan.org/) are the standard back-ends; approximate Bayesian inference (ADVI/variational) speeds up large-league fits.

---

## 3. Deep learning: neural nets, LSTM/temporal, CNN, autoencoders

Plain feed-forward nets rarely beat boosted trees on tabular match data. The value of DL appears on **raw sequences (event streams)** and **spatial (tracking) data**, not on hand-made aggregates.

| Model | Idea | Example paper / repo | Link |
|---|---|---|---|
| **MLP / dense scoreline net** | Softmax over scorelines or Poisson-rate heads | Deep-learning + GBT study | [arXiv:2309.14807](https://arxiv.org/abs/2309.14807) |
| **LSTM match-winner** | Sequence of past matches → next result; "form" as memory | krishnakartik1/LSTM-footballMatchWinner | [GitHub](https://github.com/krishnakartik1/LSTM-footballMatchWinner) |
| **RNN on team form** | Time-series framing of results; reported accuracy ≈ market baseline | uditsharma29/soccer-result-prediction-using-rnn | [GitHub](https://github.com/uditsharma29/soccer-result-prediction-using-rnn) |
| **GRU "SoccerNet"** | Gated recurrent net for outcome prediction | *PLOS ONE* 2023 | [PLOS](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0288933&type=printable) |
| **Graph-conv net on tracking data** | Raw 22-player tracking → GCN predicts receiver/threat & models defensive effect in real time | Making Offensive Play Predictable (Stöckl et al., Sloan 2021) | [MIT Sloan](https://www.sloansportsconference.com/research-papers/making-offensive-play-predictable-using-a-graph-convolutional-network-to-understand-defensive-performance-in-soccer) |
| **Autoencoders** | Compress event/tracking data into embeddings for downstream models | see EPV survey below | [arXiv:2502.02565](https://arxiv.org/abs/2502.02565) |

Honest note: reported single-match accuracies of **~50–58%** are typical and *close to what de-vigged odds already give* — "beating the bookie" claims should be checked against RPS and a real staking back-test, not accuracy alone.

---

## 4. Graph neural networks (players & passes as graphs)

Football is 22 interacting agents → a **graph** is the natural representation (nodes = players, edges = passes/proximity). This is the most active 2024–2026 research frontier.

| Model | Idea | Example paper / repo | Link |
|---|---|---|---|
| **TacticAI** ⭐ (DeepMind × Liverpool FC) | Geometric DL / GNN with pitch-reflection (D₂) equivariance; predicts corner-kick receiver & shot, and *suggests* better setups. Trained on 7,176 Premier-League corners; experts preferred its tips **90%** of the time | Wang et al., *Nature Communications* 15 (2024), DOI 10.1038/s41467-024-45965-x | [Nature](https://www.nature.com/articles/s41467-024-45965-x) · [DeepMind blog](https://deepmind.google/blog/tacticai-ai-assistant-for-football-tactics/) |
| **GNN for sports outcomes** | Sport-agnostic graph game-state representation; improves outcome prediction over non-graph baselines (demonstrated on American football & esports) | Graph Neural Networks to Predict Sports Outcomes | [arXiv:2207.14124](https://arxiv.org/abs/2207.14124) |
| **HIGFormer** | Player-Team Heterogeneous Interaction Graph *Transformer* for outcome prediction | 2025 | [arXiv:2507.10626](https://arxiv.org/abs/2507.10626) |
| **GoalNet** | GNN player-evaluation surfacing hidden pivotal players | 2025 | [arXiv:2503.09737](https://arxiv.org/abs/2503.09737) |
| **Temporal Graph Nets** | Pass receiver & outcome prediction on dynamic graphs | Rahimian et al. | [ResearchGate PDF](https://www.researchgate.net/profile/Pegah-Rahimian/publication/374145087_Pass_Receiver_and_Outcome_Prediction_in_Soccer_Using_Temporal_Graph_Networks) |

Why it matters for prediction: GNN embeddings of *how* a team plays (not just results) are strong features for downstream match/goal models, and TacticAI shows **symmetry-aware** architectures learn from very little labelled tactical data.

---

## 5. Transformers, sequence & foundation models on event data ("football as a language")

Treat a match as a **sequence of events** (pass, shot, tackle…) and model it like text. These are the closest thing football has to GPT-style foundation models, mainly used for **simulation, valuation, and next-event prediction** rather than direct 1X2 betting.

| Model | Idea | Example paper / repo | Link |
|---|---|---|---|
| **Seq2Event** | Transformer/RNN predicts next event (time, zone, action); derives xG-correlated `poss-util` (r=0.91) | Simpson et al., *KDD* 2022 | [ACM](https://dl.acm.org/doi/10.1145/3534678.3539138) · [code](https://github.com/statsonthecloud/Soccer-SEQ2Event) |
| **Forecasting Events Through Language** | Language-model framing of match event streams | 2024 | [arXiv:2402.06820](https://arxiv.org/abs/2402.06820) |
| **LEM (Large Events Model)** ⭐ | Autoregressive "language of soccer"; simulates full matches from any game state; foundation-model framing | Mendes-Neves et al., *Machine Learning* 2024 (DOI 10.1007/s10994-024-06606-y) | [Springer](https://link.springer.com/article/10.1007/s10994-024-06606-y) · [code](https://github.com/nvsclub/LargeEventsModel) |
| **Fine-tuned LEM** | Player-performance estimation in different contexts via fine-tuning | 2024 | [arXiv:2402.06815](https://arxiv.org/abs/2402.06815) |
| **A Foundation Model for Soccer** | Transformer trained on 3 seasons; forecasts next actions; beats Markov & MLP baselines | Baron, Hocevar & Salehe, 2024 | [arXiv:2407.14558](https://arxiv.org/abs/2407.14558) |
| **OpenSTARLab** ⭐ | Open framework: unified event format (UIED) + deep event modelling + RLearn | Yeung et al., *Complex & Intelligent Systems* 2025 | [arXiv:2502.02785](https://arxiv.org/abs/2502.02785) · [Springer](https://link.springer.com/article/10.1007/s40747-025-01965-y) |

---

## 6. LLMs for football prediction — and their hard limits

Off-the-shelf LLMs are tempting but **weak forecasters**. Use them for parsing news/injuries into features, not as the predictor.

| Finding | Detail | Link |
|---|---|---|
| **LLMs ≈ RF/XGBoost, no training** | GPT-4 reaches comparable 1X2 accuracy with zero feature engineering — *but* relies on numeric features & lacks deep domain reasoning | [OSF preprint](https://sciety.org/articles/activity/10.31235/osf.io/e5wpy_v2) |
| **GPT-4 is a poor forecaster** | On real-world tournament questions GPT-4 did **not** beat a naive 50% baseline; knowledge-cutoff blindness | [arXiv:2310.13014](https://arxiv.org/abs/2310.13014) |
| **Silicon-crowd ensembles help** | An ensemble of 12 LLMs rivals human-crowd accuracy — single models don't | [PMC11800985](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11800985/) |

**Bottom line:** LLMs add no proven betting edge over calibrated statistical models; their value is data plumbing (NER on team news, summarising form), not probability estimation.

---

## 7. Simulation & reinforcement learning

Two distinct uses: (a) **agent RL** to learn to *play*; (b) **Monte-Carlo** to turn per-match models into **tournament** probabilities.

| Tool / method | Idea | Example paper / repo | Link |
|---|---|---|---|
| **Google Research Football (gfootball)** ⭐ | Physics-based 11v11 RL environment; benchmarks (PPO, IMPALA, Ape-X DQN) + Academy scenarios | Kurach et al., 2019 | [arXiv:1907.11180](https://arxiv.org/abs/1907.11180) · [GitHub](https://github.com/google-research/football) |
| **GRF MARL benchmark** | Multi-agent RL toolkit on gfootball | jidiai/GRF_MARL | [GitHub](https://github.com/jidiai/GRF_MARL) |
| **Monte-Carlo tournament sim** | Draw scorelines from a Poisson/Dixon–Coles model, replay bracket 50k× → title/advance odds | Elo + Dixon–Coles + MC (World Cup 2026) | [GitHub](https://github.com/Hicruben/world-cup-2026-prediction-model) |
| **Bayesian MC forecaster** | Full Bayesian tournament simulation | 0xNadr/wc2026 | [GitHub](https://github.com/0xNadr/wc2026) |

Back-tested MC World-Cup models report **Brier ≈ 0.57–0.58, ~55–58% accuracy** — in line with bookmaker-grade systems, i.e. no free lunch.

---

## 8. xG & possession-value models (the feature factories)

xG (*gols esperados*) and possession value aren't 1X2 predictors themselves, but they are the **most important engineered features** for every model above.

| Model | Idea | Example paper / repo | Link |
|---|---|---|---|
| **xG (logistic / GBM / NN)** | P(goal) from shot distance, angle, body part, GK distance; GBT reaches r≈0.90 vs benchmark | expected-goals-thesis (Rowlinson) | [GitHub](https://github.com/andrewRowlinson/expected-goals-thesis) |
| **VAEP / xT (socceraction)** ⭐ | Value every on-ball action via Δ scoring/conceding probability (VAEP) or possession grid (xT); SPADL format | ML-KULeuven/socceraction | [GitHub](https://github.com/ML-KULeuven/socceraction) · [docs](https://socceraction.readthedocs.io/) |
| **Expected Possession Value (U-Net)** | Fine-grained pitch-value surfaces; benchmark + risk/reward for passes | Revisiting EPV | [arXiv:2502.02565](https://arxiv.org/abs/2502.02565) |
| **xG bias caveat** | xG confounds chance quality with finishing skill — don't over-trust it | Biases in Expected Goals Models | [arXiv:2401.09940](https://arxiv.org/abs/2401.09940) |

---

## 9. Ensembles, model+market blending & calibration

The most reliable real-world gains come **not** from a single exotic model but from **blending model probabilities with market odds** and then **calibrating**.

| Technique | Idea | Link |
|---|---|---|
| **Stacking / meta-learner** | Base models (Poisson, Elo, XGBoost, LSTM) → meta-model produces final 1X2 | [scikit-learn: stacked generalization](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization) |
| **Model + market blend** | Weighted average of your model and de-vigged closing odds; markets supply calibration, model supplies latent signal | [Wilkens, Bundesliga 2026](https://journals.sagepub.com/doi/10.1177/22150218261416681) |
| **Platt scaling** | Fit logistic on model scores → probabilities; good for sigmoidal distortion | [scikit-learn: calibration](https://scikit-learn.org/stable/modules/calibration.html) |
| **Isotonic regression** | Non-parametric, more flexible, needs more data; corrects non-linear miscalibration | same |
| **Odds as a tale of two markets** | 1X2 shows favourite–longshot bias; Asian-handicap odds map to (near-)efficient probabilities | [Hegarty & Whelan, *Int. J. Forecasting* 2025](https://www.sciencedirect.com/science/article/pii/S0169207024000670) |
| **Exploiting inefficiency (honest)** | Beating the *average* (not the best) odds can yield modest returns — fragile & book-limited | [arXiv:2303.16648](https://arxiv.org/abs/2303.16648) |

**Wilkens (2026)** reports ~10% ROI at *average* market odds (up to ~15% at *best* prices) from a simple xG-Skellam + isotonic model over 11 Bundesliga seasons — a rare positive result, and it depends critically on **line shopping** and stakes bookmakers may restrict. Treat such numbers as upper bounds, not promises.

---

## Data, datasets & tooling (free vs paid)

| Resource | Type | Free/Paid | Link |
|---|---|---|---|
| **StatsBomb Open Data** | Event data (WC, some leagues) — teaching corpus | **Free** (attribution req.) | [GitHub](https://github.com/statsbomb/open-data) |
| **football-data.co.uk** | Historical results **+ bookmaker odds** (CSV) | **Free** | [site](https://www.football-data.co.uk/data.php) |
| **football-data.org** | Fixtures/results REST API, 12 comps free tier | **Free** tier / paid | [site](https://www.football-data.org/) |
| **Understat** | xG per shot/match, top-5 leagues | **Free** (scrape) | [understat.com](https://understat.com/) |
| **Open Int'l Soccer Database** | 216,743 matches, 52 leagues (2017 ML Challenge) | **Free** | [Springer](https://link.springer.com/article/10.1007/s10994-018-5726-0) |
| **penaltyblog** | Poisson/Dixon–Coles/Bayesian + betting utils | **Free** (MIT) | [GitHub](https://github.com/martineastwood/penaltyblog) |
| **socceraction** | VAEP/xT + SPADL loaders (StatsBomb/Opta/Wyscout) | **Free** (MIT) | [GitHub](https://github.com/ML-KULeuven/socceraction) |
| **soccerdata** | Scraper wrappers (FBref, Understat, ELO…) | **Free** (Apache-2.0) | [PySport](https://opensource.pysport.org/?sports=Soccer) |
| **mplsoccer** | Pitch/shot-map/pass-network viz | **Free** | [docs](https://mplsoccer.readthedocs.io/) |
| **Opta / Sportradar / StatsBomb / Wyscout** | Pro event & tracking feeds | **Paid** | commercial |

Sibling pages in this folder: [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets](./Kaggle_Football_Datasets_and_Competitions.md).

---

## 🇧🇷 Brazil & regional notes

- **Data (dados):** Brasileirão Série A results/odds are on [football-data.co.uk (extra leagues)](https://www.football-data.co.uk/all_new_data.php) and Kaggle — e.g. [Brazilian Soccer Odds 2012–2024](https://www.kaggle.com/datasets/felipebandeiraramos/brazilian-soccer-odds-data), [Matches 2003–2025](https://www.kaggle.com/datasets/rczekster/matches-brazilian-football-from-2003-to-2019), and fantasy features from [Cartola FC Scouts](https://www.kaggle.com/datasets/lgmoneda/cartola-fc-brasil-scouts).
- **TacticAI in Brazil:** DeepMind's set-piece system is reportedly being piloted with **Palmeiras** ([The Next Web](https://thenextweb.com/news/google-deepmind-tacticai-football-palmeiras-predict-plays)) — a concrete case of GNN tactics reaching South-American football.
- **Regulação (regulation):** Brazil's fixed-odds betting market is regulated under **Lei 14.790/2023** (in force since 1 Jan 2025) via the **Secretaria de Prêmios e Apostas (SPA/MF)**, with mandatory self-limits, CPF/face verification, no credit-card funding, and no sign-up bonuses ([Secretaria de Prêmios e Apostas — Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas)).

---

## How to evaluate honestly (leia isto / read this)

1. **Metrics, not accuracy:** report **RPS** (rank-aware, the football standard), **log-loss / Brier**, and a **reliability/calibration curve (ECE)** — not just win-rate.
2. **Out-of-sample & time-ordered:** train on past, test on future; never shuffle across seasons (leakage).
3. **Beat the closing line:** the only real edge test is positive **Closing Line Value (CLV)** and profit *after* the overround, at odds you can actually get.
4. **Calibrate before betting:** apply Platt/isotonic; a miscalibrated model that "predicts well" still loses money.
5. **Expect small gains:** peer-reviewed work consistently shows innovative models beat Dixon–Coles + market by **little or nothing** in profit terms. Analytical insight (tactics, valuation, scouting) is where DL clearly wins — betting is not.

---

## Responsible-gambling resources (OBRIGATÓRIO)

**Gambling is not an investment strategy. The house edge is mathematical and permanent. If you or someone you know is struggling, reach out now — help is free and confidential.**

| Resource | Region | Contact |
|---|---|---|
| **GamCare** — National Gambling Helpline | 🇬🇧 UK (global info) | [gamcare.org.uk](https://www.gamcare.org.uk/) · **0808 8020 133** (24/7) |
| **BeGambleAware / National Gambling Support Network** | 🇬🇧 UK | [gambleaware.org](https://www.gambleaware.org/) *(GambleAware ceased operations 31 Mar 2026; gambling-harm treatment is now funded via the statutory levy and delivered through the **National Gambling Support Network**, a Great Britain-wide network of specialist gambling-treatment providers)* |
| **Gambling Therapy** | 🌍 Worldwide | [gamblingtherapy.org](https://www.gamblingtherapy.org/) |
| **🇧🇷 Jogo Responsável — SPA/MF** | 🇧🇷 Brazil | [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) |
| **🇧🇷 IBJR — Instituto Brasileiro de Jogo Responsável** | 🇧🇷 Brazil | [ibjr.org.br](https://ibjr.org.br/en/responsible-gaming/) |
| **🇧🇷 CVV — emotional support** | 🇧🇷 Brazil | **188** · [cvv.org.br](https://www.cvv.org.br/) |

---

**Keywords:** football / soccer prediction, futebol previsão de resultados, apostas esportivas (research only), gradient boosting XGBoost LightGBM CatBoost, Dixon-Coles bivariate Poisson, Bayesian hierarchical model rede bayesiana, PyMC Stan, deep learning aprendizado profundo, LSTM RNN transformer, graph neural network rede neural em grafo GNN, TacticAI DeepMind, Seq2Event Large Events Model foundation model modelo fundacional, LLM GPT football forecasting, Google Research Football gfootball reinforcement learning aprendizado por reforço, Monte Carlo tournament simulation simulação, expected goals xG gols esperados, VAEP xT socceraction, calibration Platt isotonic calibração, ensemble stacking model+market blend, closing line value eficiência de mercado, Ranked Probability Score Brier, Brasileirão Cartola FC, jogo responsável responsible gambling.
