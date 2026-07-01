# Key Papers & Research — Football Prediction

> A curated, source-verified map of the academic literature on predicting football (soccer / *futebol*) match outcomes and scorelines — from the 1982 Poisson foundations to 2024–2025 graph/transformer deep learning — with real DOIs, arXiv IDs and author pages. **Research & education only (pesquisa e educação); not betting advice.**

---

## ⚠️ Responsible-Gambling Notice (Aposta Responsável) — read first

- This page indexes **peer-reviewed research and open datasets** for data science / ML study. It is **NOT** tipping, a betting system, or financial advice (*não é aconselhamento de apostas*).
- **Football betting markets are highly efficient.** Bookmaker margins (*overround* / *vig*) plus sharp price discovery mean the market **closing line** is a very strong probability estimate. The academic consensus (Štrumbelj 2014; Wunderlich & Memmert 2018) is that **odds-derived probabilities usually beat published statistical models.**
- **Most bettors lose money over time.** Even papers that report positive back-tested returns (Dixon & Coles 1997; Constantinou 2013; Kaunitz 2017) stress that edges are small, fragile, regime-dependent, shrink after publication, and are eroded by margin, limits and account restrictions. **Beating the closing line consistently is extremely hard.**
- If gambling stops being fun, seek help. Resources: **[BeGambleAware](https://www.begambleaware.org/)**, **[GamCare](https://www.gamcare.org.uk/)** (National Gambling Helpline), **[Gambling Therapy](https://www.gamblingtherapy.org/)**. 🇧🇷 Brazil: **[Jogo Responsável / SPA-MF](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas)** and **[CVV — 188](https://cvv.org.br/)** (emotional support, 24h).

---

## Scope & how to read this page

Football-prediction research splits into five overlapping strands. Each table below lists **paper | year | contribution | link** and marks access as 🟢 open / 🔵 paywalled (preprint usually free). Coverage is **worldwide** — English/European leagues dominate the literature, but the modern datasets and challenges span **52+ leagues in 35+ countries** (incl. 🇧🇷 Brasileirão, MLS, Asia, Africa).

| Strand | Core idea | Canonical works |
|---|---|---|
| **Statistical (goal-based)** | Model goals as (biv.) Poisson; teams = attack/defence params | Maher; Dixon–Coles; Karlis–Ntzoufras; Baio–Blangiardo; Koopman–Lit |
| **Ratings systems** | Compress form into a single strength number (Elo/pi/odds) | Hvattum–Arntzen; Constantinou–Fenton pi-ratings; Wunderlich–Memmert |
| **ML / Deep learning** | Learn from engineered features / graphs / event data | Hubáček; Berrar; Constantinou *Dolores*; HIGFormer |
| **Betting-market efficiency** | Are odds beatable? margins, biases, Kelly staking | Constantinou–Fenton; Štrumbelj; Kaunitz; Dmochowski |
| **Surveys / datasets** | Reviews + open benchmark data & challenges | Bunker–Thabtah; Hubáček review; Open Intl. Soccer DB |

---

## 1) Statistical foundations — Poisson & Bayesian goal models

The workhorse assumption: goals scored by each team are (approximately) Poisson, driven by latent **attack** (*ataque*) and **defence** (*defesa*) strengths plus **home advantage** (*fator casa*). Everything else is a refinement (dependence between scores, time-decay, dynamics, Bayesian shrinkage).

| Paper | Year | Contribution | Link |
|---|---|---|---|
| Maher — *Modelling association football scores* (Statistica Neerlandica 36:109–118) | 1982 | 🟢 The origin: independent Poisson with team attack/defence parameters + home effect | [DOI 10.1111/j.1467-9574.1982.tb00782.x](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9574.1982.tb00782.x) |
| Dixon & Coles — *Modelling Association Football Scores and Inefficiencies in the Football Betting Market* (JRSS-C / Applied Statistics 46(2):265–280) | 1997 | 🔵 The most-cited model: low-score dependence correction + exponential **time-decay** weighting; first credible **profitable** betting back-test | [DOI 10.1111/1467-9876.00065](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9876.00065) |
| Rue & Salvesen — *Prediction and Retrospective Analysis of Soccer Matches in a League* (JRSS-D, The Statistician 49(3):399–418) | 2000 | 🔵 Bayesian **dynamic** generalized linear model; MCMC; time-varying team skills | [DOI 10.1111/1467-9884.00243](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9884.00243) |
| Karlis & Ntzoufras — *Analysis of sports data by using bivariate Poisson models* (JRSS-D 52:381–393) | 2003 | 🟢 **Bivariate Poisson**: models correlation between the two scores; better fit for **draws** ([PDF](http://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf)) | [DOI 10.1111/1467-9884.00366](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9884.00366) |
| Baio & Blangiardo — *Bayesian hierarchical model for the prediction of football results* (J. Applied Statistics 37(2):253–264) | 2010 | 🟢 Bayesian **hierarchical** model with mixture to fix over-shrinkage; author code/page ([Baio](https://gianluca.statistica.it/research/football/)) | [DOI 10.1080/02664760802684177](https://www.tandfonline.com/doi/full/10.1080/02664760802684177) |
| Koopman & Lit — *A dynamic bivariate Poisson model … English Premier League* (JRSS-A 178(1):167–186) | 2015 | 🔵 **State-space** bivariate Poisson; intensities evolve stochastically; out-of-sample betting evaluation | [DOI 10.1111/rssa.12042](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssa.12042) |
| Angelini & De Angelis — *PARX model for football match predictions* (J. Forecasting 36(7):795–807) | 2017 | 🔵 **PARX**: Poisson autoregression w/ exogenous covariates; goal-count dynamics ([PDF](https://cris.unibo.it/bitstream/11585/584368/10/PARX_model_for_football.pdf)) | [DOI 10.1002/for.2471](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.2471) |

**Reading path:** Maher → Dixon–Coles (learn time-decay + low-score dependence correction) → Karlis–Ntzoufras (correlation) → Baio–Blangiardo / Koopman–Lit (Bayesian & dynamic). The Dixon–Coles model is still the community baseline; open implementations exist (e.g. R [`regista`](https://torvaney.github.io/regista/reference/dixoncoles.html), Python `penaltyblog`).

---

## 2) Ratings systems — Elo, pi-ratings, odds-derived

Ratings compress a team's history into one number and are cheap, transferable, and surprisingly strong features for downstream ML.

| Paper / system | Year | Contribution | Link |
|---|---|---|---|
| Hvattum & Arntzen — *Using ELO ratings for match result prediction in association football* (Int. J. Forecasting 26(3):460–470) | 2010 | 🔵 First rigorous **Elo-for-forecasting** study; Elo as covariate in ordered-logit; benchmark vs market | [DOI 10.1016/j.ijforecast.2009.10.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001708) |
| Constantinou & Fenton — *Determining the level of ability of football teams by dynamic ratings* (**pi-ratings**, J. Quantitative Analysis in Sports 9(1):37–50) | 2013 | 🟢 **pi-ratings** from score discrepancies; reported to beat Elo; profitable back-test ([PDF](http://www.constantinou.info/downloads/papers/pi-ratings.pdf)) | [DOI 10.1515/jqas-2012-0036](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2012-0036/html) |
| Wunderlich & Memmert — *The Betting Odds Rating System: Using soccer forecasts to forecast soccer* (PLOS ONE 13(6):e0198668) | 2018 | 🟢 Builds a rating **from betting odds themselves**; shows odds carry most of the predictive signal | [DOI 10.1371/journal.pone.0198668](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668) |
| Groll, Ley, Schauberger & Van Eetvelde — *Prediction of the FIFA World Cup 2018 — a random forest approach with estimated team-ability parameters* | 2018 | 🟢 **Hybrid**: random forest + Poisson-ranking ability params; used to simulate 🌍 World Cup | [arXiv:1806.03208](https://arxiv.org/abs/1806.03208) |

**Note (international / World Cup):** ability-parameter + ranking hybrids (Groll et al.) and Elo variants dominate **tournament** forecasting where per-team histories are short; see also newer international-ranking comparisons — *Alternative ranking measures to predict international football results* (Macrì Demartino, Egidi & Torelli, [arXiv:2405.10247](https://arxiv.org/abs/2405.10247)).

---

## 3) Machine learning & deep learning

The 2017 & 2023 **Soccer Prediction Challenges** turned this into a benchmarked field: on **goals-only** data, gradient-boosted trees over rating features (pi-ratings) are still state-of-the-art; richer **event/tracking** data unlocks graph and transformer models.

| Paper | Year | Contribution | Link |
|---|---|---|---|
| Constantinou — *Dolores: a model that predicts football match outcomes from all over the world* (Machine Learning 108:49–75) | 2019 | 🔵 Dynamic ratings + **Hybrid Bayesian Networks**; predicts a league from *other* leagues; 2nd in 2017 Challenge (52 leagues) | [DOI 10.1007/s10994-018-5703-7](https://link.springer.com/article/10.1007/s10994-018-5703-7) |
| Hubáček, Šourek & Železný — *Learning to predict soccer results from relational data with gradient boosted trees* (Machine Learning 108:29–47) | 2019 | 🔵 **Winner, 2017 Challenge**: pi-ratings + PageRank features + **gradient-boosted trees** | [DOI 10.1007/s10994-018-5704-6](https://link.springer.com/article/10.1007/s10994-018-5704-6) |
| Berrar, Lopes & Dubitzky — *Incorporating domain knowledge in machine learning for soccer outcome prediction* (Machine Learning 108:97–126) | 2019 | 🔵 **Recency** feature extraction + **rating feature learning**; top-ranked kNN model | [DOI 10.1007/s10994-018-5747-8](https://link.springer.com/article/10.1007/s10994-018-5747-8) |
| Constantinou, Fenton & Neil — *pi-football: A Bayesian network model for forecasting Association Football match outcomes* (Knowledge-Based Systems 36:322–339) | 2012 | 🔵 **Bayesian network** blending objective + subjective info; hierarchical inference; EPL back-test | [DOI 10.1016/j.knosys.2012.07.008](https://www.sciencedirect.com/science/article/abs/pii/S0950705112001967) |
| Berrar, Lopes & Dubitzky — *A data- and knowledge-driven framework … to predict soccer match outcomes* (Machine Learning 113:8165–8204) | 2024 | 🔵 **2023 Challenge** framework (736 matches; Task 1 = exact score, Task 2 = 1X2); kNN / ANN / naive-Bayes / ordinal-forest models ([OA PDF](https://oro.open.ac.uk/100167/9/s10994-024-06625-9.pdf)) | [DOI 10.1007/s10994-024-06625-9](https://link.springer.com/article/10.1007/s10994-024-06625-9) |
| Wang, Xu, Horton, Gudmundsson & Wang — *Player-Team Heterogeneous Interaction Graph Transformer (**HIGFormer**)* (KDD 2025) | 2025 | 🟢 Graph-augmented **transformer** over player+team interaction graphs; WyScout Open Access event data | [arXiv:2507.10626](https://arxiv.org/abs/2507.10626) |

**Player- & action-value models** (features that feed match models; also core to *análise de desempenho*):

| Model | Year | What it measures | Link |
|---|---|---|---|
| Decroos, Bransen, Van Haaren & Davis — **VAEP** (*Actions Speak Louder than Goals*, KDD '19) | 2019 | 🟢 Value of **every on-ball action** via Δ scoring/conceding probability | [DOI 10.1145/3292500.3330758](https://dl.acm.org/doi/10.1145/3292500.3330758) · [arXiv:1802.07127](https://arxiv.org/abs/1802.07127) |
| Spearman — **Beyond Expected Goals** (MIT Sloan Sports Analytics Conf.) | 2018 | 🟢 Pitch-control → spatial **off-ball scoring probability** (extends xG) | [PDF](https://www.researchgate.net/publication/327139841_Beyond_Expected_Goals) |

---

## 4) Betting-market efficiency — can models beat the odds?

The honest core of this literature: markets are **close to efficient**, edges are small and unstable, and the **closing line** is a brutal benchmark. Positive-return studies exist but come with heavy caveats (limited stakes, in-play latency, bookmaker restrictions).

| Paper | Year | Finding (with caveats) | Link |
|---|---|---|---|
| Constantinou, Fenton & Neil — *Profiting from an inefficient association football gambling market* (Knowledge-Based Systems 50:60–86) | 2013 | 🔵 BN forecasts published pre-match; reports profit vs market — but edge is thin & margin-sensitive | [DOI 10.1016/j.knosys.2013.05.008](https://www.sciencedirect.com/science/article/pii/S095070511300169X) |
| Constantinou & Fenton — *Profiting from arbitrage and odds biases of the European football gambling market* (J. Gambling Business & Economics 7(2):41–70) | 2013 | 🟢 Documents cross-book **odds biases / arbitrage** over 14 leagues, 7 seasons ([PDF](http://constantinou.info/downloads/papers/evidenceofinefficiency.pdf)) | [Journal](https://www.ubplj.org/index.php/jgbe/article/view/630) |
| Štrumbelj — *On determining probability forecasts from betting odds* (Int. J. Forecasting 30(4):934–943) | 2014 | 🔵 How to de-margin odds into probabilities; **Shin's model** beats basic normalization; odds ≈ hard-to-beat forecasts | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207014000533) |
| Kaunitz, Zhong & Kreiner — *Beating the bookies with their own numbers — and how the online sports betting market is rigged* | 2017 | 🟢 Strategy on **mispriced odds** was profitable in sim + real money — until bookmakers **limited/closed accounts** ([code](https://github.com/Lisandro79/BeatTheBookie)) | [arXiv:1710.02824](https://arxiv.org/abs/1710.02824) |
| Dmochowski — *A statistical theory of optimal decision-making in sports betting* (PLOS ONE 18(6):e0287601) | 2023 | 🟢 Theory of **optimal wager selection & staking** (Kelly-flavoured); quantiles needed, not just the mean ([code](https://github.com/dmochow/optimal_betting_theory)) | [DOI 10.1371/journal.pone.0287601](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0287601) |

**Takeaway:** treat any claimed "profitable strategy" with skepticism — verify it clears the **margin**, survives on **closing odds**, holds **out-of-sample**, and is not silently killed by staking limits. Beating the closing line is the real bar, and it is *extremely* hard.

---

## 5) Surveys, reviews & benchmark datasets

| Work | Year | Type | Link |
|---|---|---|---|
| Bunker & Thabtah — *A machine learning framework for sport result prediction* (Applied Computing & Informatics 15(1):27–33) | 2019 | 🟢 Review + CRISP-DM-style **ML framework** for sport results | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2210832717301485) |
| Hubáček, Šourek & Železný — *Forty years of score-based soccer match outcome prediction: an experimental review* (IMA J. Management Mathematics 33(1):1–18) | 2022 | 🔵 Head-to-head **experimental benchmark** of Poisson/Weibull + Elo/pi/Berrar ratings on the largest DB | [DOI 10.1093/imaman/dpab029](https://academic.oup.com/imaman/article/33/1/1/6342916) |
| Bunker, Yeung & Fujii — *Machine Learning for Soccer Match Result Prediction* (book chapter, pp. 7–49, *Artificial Intelligence, Optimization, and Data Sciences in Sports*) | 2024 | 🟢 Broad, current survey (datasets, features, models, evaluation) — **best single entry point** | [arXiv:2403.07669](https://arxiv.org/abs/2403.07669) · [DOI 10.1007/978-3-031-76047-1_2](https://link.springer.com/chapter/10.1007/978-3-031-76047-1_2) |
| Galekwa, Tshimula, Tajeuna & Kyandoghere — *A Systematic Review of Machine Learning in Sports Betting: Techniques, Challenges, and Future Directions* | 2024 | 🟢 Systematic review across sports (SVM/RF/NN), techniques, challenges & future directions | [arXiv:2410.21484](https://arxiv.org/abs/2410.21484) |
| Dubitzky, Lopes, Davis & Berrar — *The Open International Soccer Database for machine learning* (Machine Learning 108:9–28) | 2019 | 🟢 **Open dataset**: 216,743 matches, 52 leagues, 35 countries — basis of the 2017 Challenge | [DOI 10.1007/s10994-018-5726-0](https://link.springer.com/article/10.1007/s10994-018-5726-0) |
| Berrar, Lopes, Davis & Dubitzky — *Guest editorial: special issue on machine learning for soccer* (Machine Learning 108:1–7) | 2019 | 🟢 Frames the whole special issue + Challenge; predictability limits | [DOI 10.1007/s10994-018-5763-8](https://link.springer.com/article/10.1007/s10994-018-5763-8) |

Related recent empirical papers worth scanning: *Match predictions in soccer: Machine learning vs. Poisson approaches* (Fischer & Heuer, [arXiv:2408.08331](https://arxiv.org/abs/2408.08331)); *Evaluating Soccer Match Prediction Models: A Deep Learning Approach & Feature Optimization for Gradient-Boosted Trees* (Yeung, Bunker, Umemoto & Fujii, [arXiv:2309.14807](https://arxiv.org/abs/2309.14807)).

---

## 6) Regional coverage 🌍 (worldwide / all-countries)

| Region | Notes for researchers | Representative source |
|---|---|---|
| 🇬🇧🇪🇺 Europe (top-5 leagues) | Best-studied; most open odds & event data; baseline for method papers | Dixon–Coles, Koopman–Lit, PARX (EPL) |
| 🌍 International / World Cup | Short team histories → ability-param + Elo hybrids; simulation of brackets | Groll et al. [arXiv:1806.03208](https://arxiv.org/abs/1806.03208) |
| 🇧🇷 South America — Brasileirão | See §7; NN/SVM/ML studies + Portuguese-language work | SciELO / *J. Phys. Educ. (Maringá)* |
| 🇺🇸🇨🇦 North America — MLS | Included in multi-league challenge datasets (52 leagues) | Open Intl. Soccer DB |
| 🌏 Asia / 🌍 Africa | Present in the 35-country Open DB; thinner odds coverage; transfer-learning helps (*Dolores* shows cross-league prediction works) | Constantinou 2019 (Dolores) |

---

## 7) 🇧🇷 Brazil / Brasileirão & Lusophone research

| Work | Year | Contribution | Link |
|---|---|---|---|
| *Artificial intelligence techniques applied to predict teams position of the Brazilian football championship* (J. Physical Education / Maringá, v32:e3254) | 2021 | 🟢 **MLP + SVM** to predict final group/position in Brasileirão Séries A & B | [SciELO](https://www.scielo.br/j/jpe/a/Z3PVmqLxFcCn68Ns7SG87Bx/) |
| Diniz, Izbicki, Lopes & Salasar — *Comparing probabilistic predictive models applied to football* (evaluated on 1,710 Brazilian top-division matches) | 2017 | 🟢 Bayesian multinomial-Dirichlet vs. established models; proper scoring rules & calibration | [arXiv:1705.04356](https://arxiv.org/abs/1705.04356) |
| *Modelos preditivos baseados em Machine Learning para otimização de estratégias em apostas esportivas no futebol* (Revista FT, v28 n139) | 2024 | 🟢 Portuguese-language applied ML-for-betting study (RF/LogReg/SVM/NN) — **read with the efficiency caveats above** | [Revista FT](https://revistaft.com.br/modelos-preditivos-baseados-em-machine-learning-para-otimizacao-de-estrategias-em-apostas-esportivas-no-futebol/) |

**🇧🇷 Regulation context (important):** fixed-odds betting (*apostas de quota fixa*) is regulated by **[Lei nº 14.790/2023](http://www.planalto.gov.br/ccivil_03/_ato2023-2026/2023/lei/l14790.htm)**, overseen by the **Secretaria de Prêmios e Apostas (SPA/MF)**; only SPA-authorized operators may legally operate since **Jan 1, 2025**, and operators must implement **jogo responsável** measures. Research use of odds/results data is fine; this index does **not** promote betting.

---

## Responsible-gambling resources (recap)

| Resource | Region | Link |
|---|---|---|
| BeGambleAware | 🇬🇧 / intl | https://www.begambleaware.org/ |
| GamCare (National Gambling Helpline) | 🇬🇧 | https://www.gamcare.org.uk/ |
| Gambling Therapy (Gordon Moody) | 🌍 global (multi-language) | https://www.gamblingtherapy.org/ |
| Jogo Responsável — SPA / Ministério da Fazenda | 🇧🇷 | https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas |
| CVV — apoio emocional (188) | 🇧🇷 | https://cvv.org.br/ |

---

## How to use this literature (practitioner notes)

1. **Baseline first.** Reproduce Dixon–Coles (goals + time-decay) before anything fancy; it beats most naive ML.
2. **Ratings are cheap signal.** pi-ratings / Elo as features consistently rank among top approaches (Hubáček 2019/2022).
3. **Benchmark honestly.** Use **RPS / log-loss**, walk-forward splits, and always compare to **de-margined market odds** (Štrumbelj 2014), not just accuracy.
4. **Deep learning needs event/tracking data.** On goals-only data, GBTs win; graphs/transformers (HIGFormer) need WyScout/StatsBomb-style inputs.
5. **Never confuse a back-test with profit.** Efficiency, margins, limits and closing-line value dominate. Research ≠ a licence to bet.

---

**Keywords:** football/soccer prediction, match outcome forecasting, Poisson model, Dixon-Coles, bivariate Poisson, Bayesian hierarchical, Elo ratings, pi-ratings, xG expected goals, VAEP, machine learning, gradient boosted trees, graph neural network, betting market efficiency, closing line value, Kelly criterion, Brasileirão, responsible gambling · *previsão de futebol, resultado de partidas, modelo de Poisson, ratings, aprendizado de máquina, eficiência de mercado de apostas, aposta responsável, jogo responsável, Lei das Bets, Brasileirão*
