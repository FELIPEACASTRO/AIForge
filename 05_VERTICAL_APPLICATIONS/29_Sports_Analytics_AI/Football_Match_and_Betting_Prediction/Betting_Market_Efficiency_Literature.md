# Betting-Market Efficiency & Sports-Economics Literature

> A verified, citation-checked map of the academic literature on **betting-market (in)efficiency and sports economics** (*(in)eficiência do mercado de apostas*) — the theory behind whether football (*futebol*) betting can be beaten. Every paper below was confirmed against a live publisher, RePEc, NBER, arXiv, or SciELO record in 2024–2026. **Research & education only — not betting advice.**

## ⚠️ Responsible gambling / Jogo Responsável (read first)

The scholarly consensus in this exact literature is sobering: bookmaker and exchange odds are **efficient forecasts** of match outcomes, published "edges" are small, fragile, and usually vanish after the bookmaker margin (*overround / vig / margem*), and **most bettors lose money over time**. Even papers reporting inefficiency (Kuypers 2000; Constantinou & Fenton 2013) find edges that are thin, regime-dependent, or arbitraged away — and Winkelmann et al. (2024) show many "anomalies" are just sampling noise. Statistical accuracy (beating a naive baseline) is **not** long-run profit after margin. Treat every model as a hypothesis; back-test out-of-sample; never stake money you cannot lose.

- **BeGambleAware** — <https://www.begambleaware.org> · **GamCare** (UK, free 24/7 helpline) — <https://www.gamcare.org.uk>
- 🇧🇷 **Jogo Responsável** — apostar é entretenimento adulto (18+), nunca fonte de renda. Ajuda: **Jogadores Anônimos Brasil** <https://jogadoresanonimos.com.br> · SUS **CAPS** para *Transtorno do Jogo (Ludopatia)*. Mercado regulado no Brasil pela **Lei 14.790/2023** (SPA/Ministério da Fazenda).

---

## Why this field exists

Wagering markets are economists' favourite "natural laboratory": prices (odds) map directly to probabilities, and every bet has a definite, dated resolution, so the **Efficient Market Hypothesis** can be tested cleanly. The central questions: (1) do odds embed all public information (*weak/semi-strong efficiency*)? (2) are there systematic biases — the **favourite–longshot bias (viés favorito–zebra)**, home/sentiment bias — a bettor could exploit? (3) is the market's consensus price (the **closing line**) the best available forecast? The honest bottom line across 35 years: markets are **hard to beat net of margin**, and where inefficiencies appear they are small, short-lived, or explained by bookmaker risk-management, not free money.

---

## Where to find this literature (discovery venues)

| Platform | Concrete football/betting query | Free? | API | Link |
|---|---|:--:|:--:|---|
| **SSRN** (working papers) | eLibrary search `betting market efficiency football` | ✅ | ❌ | [papers.ssrn.com](https://papers.ssrn.com/) |
| **RePEc / IDEAS** | search form → `football betting market efficiency` | ✅ | ✅ (metadata) | [ideas.repec.org/search.html](https://ideas.repec.org/search.html) |
| **EconPapers** (RePEc mirror) | `econpapers.repec.org` keyword `favourite-longshot bias` | ✅ | ✅ | [econpapers.repec.org](https://econpapers.repec.org) |
| **NBER** (US working papers) | `nber.org/search?q=sports+betting+market` | ✅* | ❌ | [nber.org](https://www.nber.org/search?q=sports+betting+market) |
| **arXiv** (q-fin / stat) | `arxiv.org/search/?searchtype=all&query=football+betting+market+efficiency` | ✅ | ✅ | [arxiv.org](https://arxiv.org/search/?searchtype=all&query=football+betting+market+efficiency) |
| **Google Scholar** | `scholar.google.com/scholar?q=%22betting+market%22+efficiency+football&as_ylo=2024` | ✅ | ❌ | [scholar.google.com](https://scholar.google.com/scholar?q=%22betting+market%22+efficiency+football&as_ylo=2024) |
| **SciELO Preprints** 🇧🇷 | `preprints.scielo.org` search `apostas esportivas` | ✅ | ✅ (OAI) | [preprints.scielo.org](https://preprints.scielo.org/index.php/scielo/search?query=apostas%20esportivas) |
| **Crossref** (DOI/bibliographic) | `api.crossref.org/works?query=football+betting+market+efficiency` | ✅ | ✅ (no key) | [search.crossref.org](https://search.crossref.org) |

\* NBER papers are gratis for most academic/developing-country users; some downloads are gated. **Key journals** to browse directly: *Journal of Sports Economics*, *International Journal of Forecasting (IJF)*, *Journal of the Royal Statistical Society (JRSS)*, *Economic Journal*, *Journal of Economic Literature/Perspectives*, *Journal of Gambling Business and Economics*, *Economic Inquiry*, *Economica*, *Applied Economics*.

---

## 1) Surveys & foundational reviews

Start here — these frame everything below.

| Paper | Year | Finding (verified) | Link |
|---|:--:|---|---|
| Sauer — *The Economics of Wagering Markets*, **J. Economic Literature** 36(4):2021–2064 | 1998 | Landmark survey: wagering prices are broadly efficient forecasts, but documents real departures explained by diverse information, heterogeneous agents, transaction costs. | [RePEc](https://ideas.repec.org/a/aea/jeclit/v36y1998i4p2021-2064.html) |
| Vaughan Williams (ed.) — *Information Efficiency in Financial and Betting Markets*, **Cambridge Univ. Press** (ISBN 978-0521816038) | 2005 | Edited volume surveying theory + international evidence on how information efficiency operates in betting vs financial markets. | [Cambridge](https://www.cambridge.org/core/books/information-efficiency-in-financial-and-betting-markets/BAF6D626DEBEE0586AB53BCF5E1A996D) |
| Ottaviani & Sørensen — *The Favorite–Longshot Bias: An Overview of the Main Explanations*, in *Handbook of Sports & Lottery Markets* (Hausch & Ziemba, eds.), pp. 83–101 | 2008 | Taxonomy of every leading FLB explanation (risk-love, misperception, informed insiders, market-power). | [PDF (U. Copenhagen)](https://web.econ.ku.dk/sorensen/papers/FLBsurvey.pdf) |
| Wolfers & Zitzewitz — *Prediction Markets*, **J. Economic Perspectives** 18(2):107–126 (NBER WP 10504) | 2004 | Market-aggregated forecasts are typically accurate and beat moderately sophisticated benchmarks — the wisdom-of-crowds baseline. | [AEA](https://www.aeaweb.org/articles?id=10.1257/0895330041371321) · [NBER](https://www.nber.org/papers/w10504) |

## 2) Market efficiency — foundational & modern evidence

| Paper | Year | Finding (verified) | Link |
|---|:--:|---|---|
| Pope & Peel — *Information, Prices and Efficiency in a Fixed-odds Betting Market*, **Economica** 56(223):323–341 (DOI 10.2307/2554281) | 1989 | Seminal test on early-1980s English odds from four UK bookmakers: broadly efficient, some inefficiency esp. for the **draw** (*empate*), not reliably profitable. | [EconPapers](https://econpapers.repec.org/article/blaeconom/v_3a56_3ay_3a1989_3ai_3a223_3ap_3a323-41.htm) |
| Dixon & Coles — *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, **JRSS Ser. C** 46(2):265–280 | 1997 | The canonical dynamic-Poisson model fitted to 1992–1995 English data; found *some* exploitable inefficiency in mid-1990s odds — but the model, not the market, is the lasting legacy. | [Wiley (JRSS-C)](https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9876.00065) |
| Kuypers — *Information and efficiency: an empirical study of a fixed odds betting market*, **Applied Economics** 32(11):1353–1363 | 2000 | Models a profit-maximising bookmaker who *rationally* sets inefficient odds; finds evidence of profitable betting opportunities. | [RePEc](https://ideas.repec.org/a/taf/applec/v32y2000i11p1353-1363.html) |
| Levitt — *Why are Gambling Markets Organised so Differently from Financial Markets?*, **Economic Journal** 114(495):223–246 | 2004 | Bookmakers don't just match buyers/sellers — they **take positions** and set prices exploiting bettor bias, earning more than a balanced book would. Key for understanding closing lines. | [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0297.2004.00207.x) · [RePEc](https://ideas.repec.org/a/ecj/econjl/v114y2004i495p223-246.html) |
| Constantinou, Fenton & Neil — *Profiting from an inefficient association football gambling market: Prediction, risk and uncertainty using Bayesian networks*, **Knowledge-Based Systems** 50:60–86 | 2013 | The *pi-football* Bayesian-network model; reports profit vs published EPL market odds when subjective info is added. | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S095070511300169X) |
| Constantinou & Fenton — *Profiting from arbitrage and odds biases of the European football gambling market*, **J. Gambling Business & Economics** 7(2):41–70 | 2013 | 14 leagues, 2005/06–2011/12: bookmaker accuracy did **not** improve; documents consistent odds biases + arbitrage. | [RePEc](https://ideas.repec.org/a/buc/jgbeco/v7y2013i2p41-70.html) |
| Angelini & De Angelis — *Efficiency of online football betting markets*, **IJF** 35(2):712–721 | 2019 | Finds out-of-sample abnormal returns from econometric strategies → statistical inefficiency in online markets. | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207018301134) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3070329) |
| Kaunitz, Zhong & Kreiner — *Beating the bookies with their own numbers — and how the online sports betting market is rigged*, **arXiv:1710.02824** | 2017 | Bet when one bookmaker's odds beat the **market average** (consensus); profitable in 10-yr sim + real money — **until bookmakers limited/closed the winning accounts**. | [arXiv](https://arxiv.org/abs/1710.02824) |
| Winkelmann, Ötting, Deutscher & Makarewicz — *Are Betting Markets Inefficient? Evidence From Simulations and Real Data*, **J. Sports Economics** 25(1):54–97 | 2024 | Simulation + 14 seasons: reported inefficiencies are largely **sampling variation**; little evidence of persistent, exploitable edges. The sceptical modern benchmark. | [SAGE](https://journals.sagepub.com/doi/10.1177/15270025231204997) |

## 3) Favourite–longshot bias (viés favorito–zebra)

| Paper | Year | Finding (verified) | Link |
|---|:--:|---|---|
| Cain, Law & Peel — *The Favourite-Longshot Bias and Market Efficiency in UK Football Betting*, **Scottish J. Political Economy** 47(1):25–36 | 2000 | UK fixed-odds football shows the **same FLB as horse racing**: longshots systematically over-bet / underpay. | [RePEc](https://ideas.repec.org/a/bla/scotjp/v47y2000i1p25-36.html) |
| Ottaviani & Sørensen (survey, §1) | 2008 | Competing theoretical explanations for why FLB persists across markets. | [PDF](https://web.econ.ku.dk/sorensen/papers/FLBsurvey.pdf) |
| Whelan — *Risk aversion and favourite–longshot bias in a competitive fixed-odds betting market*, **Economica** 91(361):188–209 | 2024 | Modern model: FLB can arise from bettor disagreement + **risk-averse bookmakers** in a competitive fixed-odds market, not only bettor error. | [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/ecca.12500) |

## 4) Behavioural biases — sentiment, home & popularity effects

| Paper | Year | Finding (verified) | Link |
|---|:--:|---|---|
| Braun & Kvasnicka — *National Sentiment and Economic Behavior: Evidence From Online Betting on European Football*, **J. Sports Economics** 14(1):45–64 | 2013 | Domestic bookmakers profitably shade odds on **own national teams** to exploit loyalty/perception bias. | [SAGE](https://journals.sagepub.com/doi/10.1177/1527002511414718) · [RePEc](https://ideas.repec.org/a/sae/jospec/v14y2013i1p45-64.html) |
| Feddersen, Humphreys & Soebbing — *Sentiment Bias and Asset Prices: Evidence from Sports Betting Markets and Social Media*, **Economic Inquiry** 55(2):1119–1129 | 2017 | Across 7 leagues, bookmakers raise prices on teams with more Facebook "Likes" → **sentiment-driven, price-insensitive** bettors. | [RePEc](https://ideas.repec.org/a/bla/ecinqu/v55y2017i2p1119-1129.html) · [WVU WP](https://researchrepository.wvu.edu/econ_working-papers/87/) |
| Flepp, Nüesch & Franck — *Does Bettor Sentiment Affect Bookmaker Pricing?*, **J. Sports Economics** 17(1):3–11 | 2016 | Over/under 2.5-goals market: >80% of volume backs the "over," yet high **price transparency** stops bookmakers from systematically distorting odds — no bias in bettor returns. Sentiment ≠ automatic mispricing. | [SAGE](https://journals.sagepub.com/doi/abs/10.1177/1527002514521427) · [RePEc](https://ideas.repec.org/a/sae/jospec/v17y2016i1p3-11.html) |

## 5) Forecasting vs the market (odds as consensus / closing line)

Practitioner shorthand: the **closing line** (final consensus odds) is the sharpest public forecast; "closing-line value" is the closest thing to a leading indicator of long-run edge. The verified literature backs the underlying claim — market/exchange prices are excellent forecasters, and beating them consistently is rare.

| Paper | Year | Finding (verified) | Link |
|---|:--:|---|---|
| Forrest, Goddard & Simmons — *Odds-setters as forecasters: The case of English football*, **IJF** 21(3):551–564 | 2005 | Bookmaker odds are **good probabilistic forecasters** and grew *more* accurate over a ~5-year window under intensifying competitive pressure. | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207005000300) · [RePEc](https://ideas.repec.org/a/eee/intfor/v21y2005i3p551-564.html) |
| Goddard — *Regression models for forecasting goals and match results in association football*, **IJF** 21(2):331–340 | 2005 | Compares goals-based vs results-based forecasting models on identical data; both yield similar match-result accuracy. | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207004000676) |
| Štrumbelj — *On determining probability forecasts from betting odds*, **IJF** 30(4):934–943 | 2014 | **Shin's model** de-biases odds into sharper probabilities than naïve normalisation/regression; bookmakers differ as forecast sources, and exchange odds are **not always** the sharpest (esp. smaller markets). | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207014000533) |
| Franck, Verbeek & Nüesch — *Prediction accuracy of different market structures — bookmakers versus a betting exchange*, **IJF** 26(3):448–459 (Big-5, 5,478 games) | 2010 | **Betting exchange (Betfair) forecasts beat bookmakers**; FLB is weaker on the exchange — market microstructure matters. | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207010000105) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1503350) |
| Spann & Skiera — *Sports forecasting: a comparison of the forecast accuracy of prediction markets, betting odds and tipsters*, **J. Forecasting** 28(1):55–72 | 2009 | Prediction markets ≈ betting odds in accuracy; both **beat tipsters** — but no systematic profit after Germany's 25% margin. | [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1091) · [RePEc](https://ideas.repec.org/a/jof/jforec/v28y2009i1p55-72.html) |

## 6) LatAm / Brazil (🇧🇷 SciELO & Portuguese scholarship)

Betting research aimed at a Brazilian audience is emerging fast since **Lei 14.790/2023** legalised *bets*. Search **SciELO Preprints** (`apostas esportivas`, `eficiência de mercado`) and **RePEc** Brazilian series (IPEA, EESP-FGV, ANPEC). Verified examples:

| Item | Year | What it is | Link |
|---|:--:|---|---|
| El Khatib — *Diversão ou Armadilha? Um estudo exploratório das Apostas Esportivas (Bets) entre Universitários Brasileiros sob a Lente da Teoria do Comportamento Planejado (TCP)* — **SciELO Preprints** #10133 | 2024 | Theory-of-Planned-Behaviour study of sports betting (*bets*) among Brazilian university students; examines behavioural intention and the moderating role of problem-gambling severity. Directly relevant to **responsible-gambling** framing. | [SciELO Preprints](https://preprints.scielo.org/index.php/scielo/preprint/view/10133) |
| Lobão & Rolla — *Um outro olhar sobre a eficiência dos mercados: o caso das bolsas de apostas de tênis* — **RAE (FGV) / SciELO Brazil** 55(4) | 2015 | Portuguese-language market-efficiency study of a betting **exchange** (Betfair tennis): prices forecast well, but under-reaction/representativeness biases leave some profitable strategies — a LatAm-published efficiency test methodologically adjacent to football work. | [SciELO](https://www.scielo.br/j/rae/a/yY7nkGzpXsYSxrBmywNKX3G/?lang=pt) · [RePEc](https://ideas.repec.org/a/fgv/eaerae/v55y2015i4a52958.html) |

---

## Core concepts & test methods (glossary)

Recurring vocabulary you will meet across every paper above (*termos-chave*):

| Concept (EN / PT) | What it means in this literature |
|---|---|
| **Weak / semi-strong / strong efficiency** (*eficiência fraca/semiforte/forte*) | Whether odds already price in past results, all public info, or even private info. Football tests are almost all weak/semi-strong (Pope & Peel 1989; Kuypers 2000). |
| **Overround / vig / margin** (*margem / overround*) | Sum of implied probabilities > 100%; the bookmaker's built-in edge that any "profitable" strategy must first overcome (Spann & Skiera 2009). |
| **Favourite–longshot bias, FLB** (*viés favorito–zebra*) | Longshots are over-bet and underpay relative to true odds; favourites are (mildly) underpriced (Cain–Law–Peel 2000; Ottaviani & Sørensen 2008). |
| **Closing line / consensus price** (*linha de fechamento*) | The final pre-kickoff market price; treated as the sharpest public forecast (Forrest–Goddard–Simmons 2005). |
| **Shin probabilities** (*modelo de Shin*) | Method to strip the margin + insider effect out of quoted odds to recover cleaner forecasts (Štrumbelj 2014). |
| **Common test designs** | (a) Regression of outcomes on implied probabilities; (b) betting-simulation returns (out-of-sample, net of margin); (c) forecast-scoring via **Brier score / log-loss**; (d) arbitrage / value-bet screens. Winkelmann et al. (2024) stress persistence testing over single-season results. |

---

## How to read this literature honestly

1. **"Inefficiency" ≠ profit.** Many "beat-the-market" papers report gross returns or ignore that bookmakers **limit or close winning accounts** (documented explicitly by Kaunitz et al. 2017). A statistical edge you cannot *stake at scale* is not a real edge.
2. **Watch the margin (overround).** Fair-value tests must subtract the vig; Spann & Skiera (2009) show a 25% margin erases forecasting superiority entirely.
3. **Beware small samples.** Winkelmann et al. (2024) show single-season "anomalies" are consistent with pure chance — always test persistence out-of-sample and across leagues.
4. **The market is the benchmark to beat.** Forrest–Goddard–Simmons (2005), Franck et al. (2010) and Štrumbelj (2014) agree: the **consensus/closing/exchange price** is a strong forecaster. A model that merely tracks it adds nothing after costs.
5. **Bias ≠ opportunity.** FLB (Cain–Law–Peel 2000; Whelan 2024) and sentiment effects (Braun & Kvasnicka 2013; Feddersen et al. 2017) are real, but often reflect **bookmaker risk-management** and price transparency can neutralise them (Flepp et al. 2016) — frequently too small to exploit net of margin.

**Consensus:** football betting markets are broadly efficient and **hard to beat net of margin**; documented edges are small, fragile, regime-dependent, and actively defended against by bookmakers. Study this literature to understand markets — **not** as a staking strategy.

---

**Sources:** [Sauer 1998 (RePEc)](https://ideas.repec.org/a/aea/jeclit/v36y1998i4p2021-2064.html) · [Vaughan Williams 2005 (Cambridge)](https://www.cambridge.org/core/books/information-efficiency-in-financial-and-betting-markets/BAF6D626DEBEE0586AB53BCF5E1A996D) · [Ottaviani & Sørensen 2008 (PDF)](https://web.econ.ku.dk/sorensen/papers/FLBsurvey.pdf) · [Wolfers & Zitzewitz 2004 (AEA)](https://www.aeaweb.org/articles?id=10.1257/0895330041371321) / [NBER 10504](https://www.nber.org/papers/w10504) · [Pope & Peel 1989 (EconPapers)](https://econpapers.repec.org/article/blaeconom/v_3a56_3ay_3a1989_3ai_3a223_3ap_3a323-41.htm) · [Dixon & Coles 1997 (Wiley)](https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9876.00065) · [Kuypers 2000 (RePEc)](https://ideas.repec.org/a/taf/applec/v32y2000i11p1353-1363.html) · [Levitt 2004 (Wiley)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0297.2004.00207.x) · [Constantinou, Fenton & Neil 2013 (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S095070511300169X) · [Constantinou & Fenton 2013 (RePEc)](https://ideas.repec.org/a/buc/jgbeco/v7y2013i2p41-70.html) · [Angelini & De Angelis 2019 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0169207018301134) · [Kaunitz et al. 2017 (arXiv)](https://arxiv.org/abs/1710.02824) · [Winkelmann et al. 2024 (SAGE)](https://journals.sagepub.com/doi/10.1177/15270025231204997) · [Cain, Law & Peel 2000 (RePEc)](https://ideas.repec.org/a/bla/scotjp/v47y2000i1p25-36.html) · [Whelan 2024 (Wiley)](https://onlinelibrary.wiley.com/doi/10.1111/ecca.12500) · [Braun & Kvasnicka 2013 (SAGE)](https://journals.sagepub.com/doi/10.1177/1527002511414718) · [Feddersen, Humphreys & Soebbing 2017 (RePEc)](https://ideas.repec.org/a/bla/ecinqu/v55y2017i2p1119-1129.html) · [Flepp, Nüesch & Franck 2016 (RePEc)](https://ideas.repec.org/a/sae/jospec/v17y2016i1p3-11.html) · [Forrest, Goddard & Simmons 2005 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0169207005000300) · [Goddard 2005 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0169207004000676) · [Štrumbelj 2014 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0169207014000533) · [Franck, Verbeek & Nüesch 2010 (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1503350) · [Spann & Skiera 2009 (Wiley)](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1091) · [El Khatib 2024 (SciELO Preprints)](https://preprints.scielo.org/index.php/scielo/preprint/view/10133) · [Lobão & Rolla 2015 (SciELO)](https://www.scielo.br/j/rae/a/yY7nkGzpXsYSxrBmywNKX3G/?lang=pt)

**Keywords:** betting market efficiency, market (in)efficiency, favourite-longshot bias, efficient market hypothesis, closing line value, wisdom of crowds, prediction markets, sentiment bias, home bias, bookmaker margin / overround, sports economics, football / soccer forecasting, sharp book, arbitrage; *eficiência do mercado de apostas, viés favorito-zebra, hipótese de mercado eficiente, mercados de previsão, viés de sentimento, margem da casa (overround), economia do esporte, previsão de futebol, jogo responsável, apostas esportivas (bets), Lei 14.790/2023*
