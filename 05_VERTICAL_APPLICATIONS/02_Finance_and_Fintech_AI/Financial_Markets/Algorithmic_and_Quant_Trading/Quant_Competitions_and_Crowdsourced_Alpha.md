# Quant Competitions & Crowdsourced Alpha

> Platforms that crowdsource trading signals and run quant tournaments — where data scientists build models, stake or submit, and (sometimes) get paid in cash, tokens, or capital allocation. This page catalogs the live ecosystem (2024–2026) beyond plain Kaggle, with honest notes on realistic earnings, IP/data terms, and Brazil 🇧🇷 equivalents.

These platforms turn the [quant research workflow](./README.md) into a public game: you get obfuscated data (or bring your own), train a model, submit predictions, and are scored out-of-sample. The economic models differ sharply — **stake-to-earn** (Numerai), **piece-rate alpha pay** (WorldQuant BRAIN), **prize pools** (CrunchDAO, Kaggle), **capital allocation + profit share** (Quantiacs), **strategy sharing / research tooling** (QuantConnect, after its alpha marketplace closed), and **pure recruiting/glory** (Citadel datathons, IMC Prosperity, Rotman). Read the IP and data-licensing fine print before investing months — see the *Honest caveats* section at the end.

---

## At-a-glance comparison

| Platform | Model you build | How you're paid | Data | Free/Paid to enter | Link |
|---|---|---|---|---|---|
| **Numerai Classic** | Tabular GBDT/NN on obfuscated stock features | Stake NMR → earn/burn weekly | Provided, obfuscated | Free (staking optional) | [numer.ai](https://numer.ai/) |
| **Numerai Signals** | Bring-your-own-data stock signals | Stake NMR → earn/burn | You source it | Free | [signals.numer.ai](https://signals.numer.ai/) |
| **Numerai Crypto** | Rank top ~500 tokens | Stake NMR → earn/burn | Provided + BYO | Free | [crypto.numer.ai](https://crypto.numer.ai/) |
| **WorldQuant BRAIN** | Formulaic "alphas" via operators | Daily piece-rate (consultants) | Provided (WebSim) | Free | [worldquant.com/brain](https://www.worldquant.com/brain/) |
| **CrunchDAO** | Cross-sectional return ranking, regimes | USDC prize pools (weekly/monthly) | Provided, obfuscated | Free | [crunchdao.com](https://crunchdao.com/) |
| **ADIA Lab Market Prediction** | Cross-section ranking of assets | Cash prize pool (~$100k) | Provided, obfuscated | Free | [adialab.ae](https://www.adialab.ae/) |
| **Quantiacs** | Python futures/equity strategies | Capital allocation + ~10% profit share | Provided (free quant data) | Free | [quantiacs.com](https://quantiacs.com/) |
| **QuantConnect** | LEAN algos | "Strategies" sharing (Alpha Streams marketplace closed) | Provided (broad) | Free tier | [quantconnect.com](https://www.quantconnect.com/) |
| **Kaggle (finance firms)** | Supervised tabular/time-series | One-off prize pools | Provided | Free | [kaggle.com](https://www.kaggle.com/competitions) |
| **Citadel / Correlation One** | Datathon analysis + presentation | Cash + recruiting | Provided | Free (students) | [citadel.com](https://www.citadel.com/careers/programs-and-events/datathons/) |
| **IMC Prosperity** | Algo + manual trading rounds | Cash + recruiting | Sim | Free (students) | [prosperity.imc.com](https://prosperity.imc.com/) |
| **Rotman RIT (RITC)** | Live market-sim cases | Glory + recruiting | Sim (RIT) | University-team entry | [rotman.utoronto.ca](https://www.rotman.utoronto.ca/) |
| **Quantopian** | (defunct) | — | — | Archive only | [github.com/quantopian](https://github.com/quantopian/research_public) |

---

## 1. Numerai — the "hardest data science tournament in the world"

Numerai runs an AI hedge fund powered by an ensemble ("Meta Model") of thousands of community models. You **never see the real tickers** — data is obfuscated, so models can't be lifted out for your own trading. You retain your model and only submit predictions; Numerai cannot reverse-engineer your model, and you can quit anytime. Cumulative payouts to data scientists have exceeded **US$43M**, with over **US$1M paid in January 2025 alone**.

| Tournament | What you predict | Data | Submit | Scoring | Link |
|---|---|---|---|---|---|
| **Classic** | 20-day forward stock returns (ranked) | Provided obfuscated features + targets | Tue–Sat daily | CORR (correlation to target) + **MMC** (Meta Model Contribution) | [docs.numer.ai](https://docs.numer.ai/) |
| **Signals** | Stock performance in Numerai's universe | **You bring your own** data/signals (ticker → value) | Daily | Signal neutralized vs Numerai's existing data — rewards *originality* | [signals.numer.ai](https://signals.numer.ai/) |
| **Crypto** | Rank of ~top 500 tokens | Provided + BYO | Daily | Correlation to forward token returns; public [Meta Model](https://crypto.numer.ai/meta-model) | [crypto.numer.ai](https://crypto.numer.ai/) |

**Staking & payouts.** You optionally lock **NMR** (the Numeraire ERC-20 token) on a model. After the ~20-day scoring window, positive scores mint extra NMR to you; negative scores **burn** part of your stake (up to ~25%/week either direction). Stake size linearly scales reward *and* penalty. Payouts are "at Numerai's discretion based on a blackbox target" — i.e., not a guaranteed contract. Released stakes take ~1 month to unlock.

- **Atomic Blockchain Staking** (announced for **July 4, 2026**) moves stakes back on-chain with wallet choice, flexible stake management, and a **USDC staking option** — a meaningful UX change. ([blog.numer.ai](https://blog.numer.ai/numerai-monthly-numercon-agents-staking-risk-1m-nmr-buyback/))
- **Erasure / NMR token economics.** NMR underpins Numerai's "Erasure" data-staking concept (stake-to-signal-confidence) and periodic NMR buybacks. ([Erasure overview](https://www.gemini.com/cryptopedia/nmr-token-numerai-prediction-markets))
- **Reality check.** Most newcomers *lose* NMR initially; beating the benchmark + MMC consistently is hard. Treat early rounds as paying tuition. Data **cannot be used outside the tournament**.

Sources: [docs.numer.ai](https://docs.numer.ai/) · [Staking docs](https://docs.numer.ai/numerai-tournament/staking) · [Signals overview](https://docs.numer.ai/numerai-signals/signals-overview)

---

## 2. WorldQuant BRAIN — formulaic alphas at scale

WorldQuant's **BRAIN** is a browser platform (**WebSim**) where you build "alphas" — short expressions combining predefined **operators** and **datafields** (e.g. `rank(-returns)`) that map to simulated equity positions. It is the on-ramp to WorldQuant's paid **Research Consultant** program and student championships.

| Track | Who | Reward | Notes |
|---|---|---|---|
| **Research Consultant Program** | Anyone (apply) | Merit-based quarterly payment for accepted alphas | Quality-weighted payout; published tier guidance: **Master ≈ $2,000+/quarter, Grandmaster ≈ $8,000+/quarter**. You reach the program by accumulating points (≈10,000 / gold status earns an invite). Most submitted alphas earn little; sums vary widely. ([consultant program](https://worldquantbrain.com/consultant)) |
| **International Quant Championship (IQC)** | University students (18+), teams 1–4 same-uni | Prizes + recruiting | 3-stage, team-based, ~Mar–Sep; free to enter; 2026 edition live. ([IQC](https://www.worldquant.com/brain/iqc/) · [guidelines](https://www.worldquant.com/brain/iqc-guidelines/)) |
| **Global Alphathon** | Open competition (original 2022 launch) | Prizes | Inaugural event that launched the BRAIN platform. ([announcement](https://www.worldquant.com/ideas/worldquant-announces-completion-of-inaugural-global-alphathon-competition/)) |

**How to participate.** Free BRAIN account → learn operators/datafields → simulate alphas → submit. IP: alphas you submit feed WorldQuant's research; you're compensated as a contractor for accepted ideas, not an owner of resulting strategies. Sources: [worldquant.com/brain](https://www.worldquant.com/brain/) · [worldquantbrain.com/consultant](https://worldquantbrain.com/consultant)

---

## 3. CrunchDAO — decentralized data-science crowdsourcing (ADIA Lab partner)

CrunchDAO (founded 2021; **10,000+ ML engineers, 1,200+ PhDs, 100+ countries**) blends top community models into ensemble **"meta-models."** It is the operator behind the **ADIA Lab** challenges (research arm of the Abu Dhabi Investment Authority) and pays in **USDC**, ranking among the top-paying ML competition venues. In Oct 2025 the team ("Crunch Lab") raised **$5M** to build an "intelligence layer for decentralized AI" and is moving toward a Solana-based **Crunch Protocol** with a native **CRNCH** token.

| Competition | Task | Prize / payout | Cadence | Link |
|---|---|---|---|---|
| **ADIA Lab Market Prediction** | Cross-section *ranking* of investment vehicles; Spearman rank-corr scoring; submit Python that trains then runs OOS on live data | ~**$100k** pool, ~$40k top prize, top-10 paid | Two 12-week phases (Submission → Out-of-Sample) | [docs.adialab.crunchdao.com](https://docs.adialab.crunchdao.com/) |
| **DataCrunch** | Forecast obfuscated financial targets | ~**$60k/yr** USDC + ~$10k cumulative-alpha bonus; weekly ~$1k to top models | Weekly/monthly | [hub.crunchdao.com/competitions/datacrunch](https://hub.crunchdao.com/competitions/datacrunch) |
| **Numinous** | Forecast binary real-world events (e.g. Polymarket questions); Brier-scored | ~**$5k** USDC pool; top-10 by weighted score paid weekly | Weekly (scored Mondays GMT) | [docs.crunchdao.com](https://docs.crunchdao.com/real-time-competitions/competitions/numinous) |
| **Falcon (Synth)** | Crypto price forecasting | **$30k** USDC, claimable on Solana mainnet | Weekly | [crunchdao.com](https://crunchdao.com/) |
| Causal discovery / time-series anomaly / biomedical | Beyond finance (regime detection, gene-disease) | USDC pools | Varies | [docs.crunchdao.com](https://docs.crunchdao.com/competitions/competitions) |

**How it works.** You submit code; the platform retrains/runs it OOS and blends top performers into the meta-model — you're paid on *contribution to the ensemble*, not raw rank alone, which reduces over-fitting incentives. CrunchDAO reports top crowd models detected market-regime/structural-break changes with **double-digit-percentage higher accuracy** than standard baselines. Sources: [crunchdao.com](https://crunchdao.com/) · [ADIA Lab case study](https://crunchdao.com/case-studies/adia-lab-crunchdao) · [ADIA Lab news](https://www.adialab.ae/adialab-news/adia-lab-crunchdao-competition-launch)

---

## 4. Quantiacs — futures/equity contests with real capital

Quantiacs is a crowdsourced quant platform: write Python strategies (free historical + macro/futures data, including IMF commodity data), pass a multi-month **simulated evaluation**, then compete in funding contests.

- **Payout model:** top strategies are allocated **investor capital** (past contests allocated **$2M / $1M** to winners) with the quant taking **~10% profit share**, **no personal capital required, no downside risk**, and **retaining IP**.
- **How to participate:** free account → clone a [strategy template](https://github.com/quantiacs) → submit → strategy is evaluated on risk-adjusted return, drawdown, robustness.
- Sources: [quantiacs.com](https://quantiacs.com/) · [GitHub](https://github.com/quantiacs)

---

## 5. QuantConnect — from Alpha Streams to "Strategies"

QuantConnect (110,000+ members) pairs the open-source **LEAN** engine with a community marketplace. Its monetization story has shifted significantly — read this before counting on payouts here.

| Mechanism | What it is | Economics | Status |
|---|---|---|---|
| **Alpha Streams Market** | Licensed your algo's signals to funds (creator-set monthly subscription + exclusivity premium; funds get signals via API, not source) | Creator-set fees | **Closed.** Alpha Streams **v1.0 ceased Feb 2022** (alphas underperformed S&P 500, overfitting); a refactored **v2.0** has been discussed but is not a live earning channel. ([overview](https://www.quantconnect.com/docs/alpha-streams/overview)) |
| **Quant League** | Quarterly community trading competition | Glory / ranking | **Sunsetting** — Q4-2025 is the *final* league; evolving into **"Strategies"** (share/discover strategies). ([league](https://www.quantconnect.com/league/)) |

**Reality check.** QuantConnect today is best treated as a top-tier free research/backtesting environment (LEAN, broad data) and a place to *share* strategies — not a reliable income venue, since the alpha-licensing marketplace is offline.

Sources: [quantconnect.com](https://www.quantconnect.com/) · [Pioneering a Free Market for Alpha](https://www.quantconnect.com/announcements/15944/pioneering-a-free-market-for-alpha/)

---

## 6. Quantopian — defunct, but the lectures live on

Quantopian (crowdsourced hedge fund) **shut down community services Nov 2020** and its team joined Robinhood. The platform is gone, **but** its educational corpus and open-source libs survive — invaluable for learning the [quant workflow](./README.md).

| Asset | What | Link |
|---|---|---|
| **research_public** | Lecture notebooks (pairs trading, factor analysis, risk) | [github.com/quantopian/research_public](https://github.com/quantopian/research_public) |
| **Lectures saved (gist)** | Curated index of notebooks + videos | [gist](https://gist.github.com/ih2502mk/50d8f7feb614c8676383431b056f4291) |
| **Quant Finance Lectures (QuantRocket)** | Maintained re-host | [quantrocket.com](https://www.quantrocket.com/codeload/quant-finance-lectures/quant_finance_lectures/Introduction.ipynb.html) |
| Open-source libs | Zipline, Alphalens, Pyfolio, Empyrical (now community-forked) | [github.com/quantopian](https://github.com/quantopian) |

Sources: [Closing announcement](https://quantopian-archive.netlify.app/forum/threads/quantopians-community-services-are-closing.html)

---

## 7. Trading-firm Kaggle competitions (link to the [Kaggle page](../../Market_Prediction/))

Top prop/quant firms run flagship Kaggle competitions — excellent real-world tabular/time-series problems, modest one-off prizes, big reputational value. (Generic Kaggle is already covered in the repo; these specific finance comps are the high-value entries.)

| Host | Competition | Task | Prize | Link |
|---|---|---|---|---|
| **Jane Street** | Real-Time Market Data Forecasting (2024) | Predict "responders" from anonymized features, time-series eval API | $100k | [kaggle](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) |
| **Jane Street** | Market Prediction (2020) | Trade/skip on anonymized signals | $100k | [kaggle](https://www.kaggle.com/competitions/jane-street-market-prediction) |
| **Optiver** | Trading at the Close (2023) | Predict Nasdaq closing-auction price moves | $100k | [kaggle](https://www.kaggle.com/competitions/optiver-trading-at-the-close) |
| **Optiver** | Realized Volatility Prediction (2021) | Short-term vol from order-book/trade data | $100k | [kaggle](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) |
| **Two Sigma** | Financial Modeling (2017) / Using News (2018) | Stock returns; news-augmented returns | Prizes | [modeling](https://www.kaggle.com/competitions/two-sigma-financial-modeling) · [news](https://www.kaggle.com/competitions/two-sigma-financial-news) |

**Why they matter:** time-series-safe evaluation APIs (no look-ahead), real anonymized market microstructure data, and public top solutions = a free curriculum. Sources: [Jane Street blog](https://blog.janestreet.com/announcing-our-market-prediction-kaggle-competition-index/)

---

## 8. Student & recruiting competitions (cash + interviews, not income)

| Event | Host | Format | Prize | Link |
|---|---|---|---|---|
| **Datathons / Data Open** | Citadel + Citadel Securities (via **Correlation One**) | Week-long datathon; team analysis + judging | **$100k** grand (Data Open Championship); $15k Women's Datathon | [citadel.com](https://www.citadel.com/careers/programs-and-events/datathons/) |
| **Terminal** | Citadel + Correlation One | Games-based algo coding, round-robin | ~$25k events | [terminal](https://www.citadel.com/careers/programs-and-events/terminal/) |
| **IMC Prosperity** | IMC Trading | 5 rounds × (algo + manual) over ~2 weeks; **22,600 players / 12,600 teams** (P4, 2025) | **$50k** pool ($25k top, $5k best manual trader) | [prosperity.imc.com](https://prosperity.imc.com/) |
| **Ready Trader Go** | Optiver | Build an algo for Optiver's sim exchange; teams 1–3, 3 weeks, top-16 finals | **€30k** grand prize | [readytradergo.optiver.com](https://readytradergo.optiver.com/) |
| **Rotman International Trading Competition (RITC)** | U. of Toronto / **Rotman Interactive Trader (RIT)** | 2-day in-person live market-sim cases, 40+ universities | Recognition + recruiting | [rotman.utoronto.ca](https://www.rotman.utoronto.ca/) |

These are recruiting funnels first, payouts second — superb for breaking into the industry, not a revenue stream. Sources: [Correlation One datathons](https://www.correlation-one.com/blog/tag/datathons) · [IMC Prosperity 4](https://www.imc.com/us/corporate-news/prosperity-4)

---

## 9. Retail "trading-contest" platforms (paper trading + prizes)

Distinct from *model* crowdsourcing — these are discretionary/algo **trading leaderboards**, often with cash or funded accounts. Lower research value but accessible.

| Platform | What | Prize | Link |
|---|---|---|---|
| **TradingView — The Leap** | Paper-trading competition, real-money prizes | Cash | [tradingview.com/the-leap](https://www.tradingview.com/the-leap/) |
| **NinjaTrader Arena** | Sim futures contests on live CME data | Cash | [ninjatrader.com](https://ninjatrader.com/ninjatrader-arena/) |
| **CME Group University Trading Challenge** | University futures trading sim (CQG platform, live data) | Recognition | [cmegroup.com](https://www.cmegroup.com/events/university-trading-challenge.html) |
| **BullRush / ForTraders / The5ers** | Retail contests; some award funded accounts | Cash / funding | [bullrush.com](https://bullrush.com/) |

---

## 10. Brazil 🇧🇷 equivalents

No Brazilian platform crowdsources *alpha for a hedge fund* (Numerai/CrunchDAO style) — these are **simulated day-trading championships** (real B3 quotes, fictional capital), popular for education and recruiting.

| Competition | Promoter | Format | Prize | Link |
|---|---|---|---|---|
| **Torneio Nacional de Trading (TNT)** | TradersClub (TC) + **B3** | Day-trade sim on WIN/WDO/crypto-futures (BIT/ETR/SOL) + GLD; R$10k fictional capital | **R$200k** total (R$150k champion) | [lp.tradersclub.com.br/tnt](https://lp.tradersclub.com.br/tnt) |
| **Copa BTG Trader** | BTG Pactual (+ B3, Nelogica) | 3-week gamified sim championship | **R$1M** champion (R$1.25M total, top 100) | [lp.btgpactual.com/copa-btg-trader](https://lp.btgpactual.com/copa-btg-trader) |
| **Copa Brasil de Trade** | Elliot / TradeArena | Online elims → in-person final at B3 | **R$450k+** | [copabrasildetrade.com.br](https://copabrasildetrade.com.br/) |
| **GainCast Trader League** | GainCast (sim on RocketTrader) | Trader championship | **R$100k** | [infomoney](https://www.infomoney.com.br/mercados/gaincast-lanca-seu-primeiro-campeonato-de-traders-com-r-100-mil-em-premios/) |

For *quant model* competitions, Brazilians compete globally on Numerai, CrunchDAO, Kaggle, and WorldQuant BRAIN — all open worldwide and free to enter. 🇧🇷 quants should note Numerai/CrunchDAO payouts are in **NMR/USDC crypto** (tax + custody implications under Receita Federal rules).

---

## Honest caveats (read before you commit months)

- **Realistic earnings.** WorldQuant BRAIN piece-rates and Numerai net returns are *small for most people*; survivor stories dominate marketing. Stake-to-earn means you can **lose principal** (NMR burns). Treat the first months as paid learning.
- **IP & data terms.** Numerai/WorldQuant: you feed a hedge fund; obfuscated data **can't be reused** elsewhere and predictions/alphas are licensed to the host. Quantiacs: you **retain IP**, license capital, and keep a profit share — better terms if you want ownership. QuantConnect: you keep your code (the alpha-licensing marketplace is now closed, so it's mainly research + strategy sharing today).
- **Crypto exposure.** Numerai (NMR) and CrunchDAO (USDC/CRNCH) pay in tokens — price/volatility, on-chain custody, and local tax (🇧🇷 Receita Federal) all apply.
- **Look-ahead / leakage.** Well-run comps (Jane Street, Optiver, ADIA Lab) use time-series-safe eval APIs and OOS phases precisely to kill look-ahead bias — study their evaluation design, it's a free lesson in [avoiding the pitfalls](./README.md) (survivorship, point-in-time, multiple-testing / deflated Sharpe).
- **Recruiting vs income.** Citadel/IMC/Rotman are career on-ramps, not earnings. Optimize for the *learning + network*, not the prize.
- **Where to find live comps:** the [Kaggle competitions list](https://www.kaggle.com/competitions) (live) plus the OpenQuant [trading-competitions roundup](https://openquant.co/blog/trading-competitions) (useful overview, but a periodic snapshot — verify dates on each event's own site).

---

## Related in AIForge
- [Algorithmic & Quant Trading](./README.md) (workflow, strategy families, pitfalls) · [Market Prediction](../../Market_Prediction/) · [Backtesting & Frameworks](../Backtesting_and_Frameworks/) · [Alternative Data & Sentiment](../Alternative_Data_and_Sentiment_Analysis/)
- Generic competition/data pulls (Kaggle, HuggingFace) are covered elsewhere in the repo; this page focuses on *finance-specific* tournaments and crowdsourced-alpha venues.

**Sources:** [docs.numer.ai](https://docs.numer.ai/) · [signals.numer.ai](https://signals.numer.ai/) · [crypto.numer.ai](https://crypto.numer.ai/) · [blog.numer.ai](https://blog.numer.ai/numerai-monthly-numercon-agents-staking-risk-1m-nmr-buyback/) · [worldquant.com/brain](https://www.worldquant.com/brain/) · [IQC](https://www.worldquant.com/brain/iqc/) · [WorldQuant consultant](https://worldquantbrain.com/consultant) · [crunchdao.com](https://crunchdao.com/) · [docs.adialab.crunchdao.com](https://docs.adialab.crunchdao.com/) · [adialab.ae](https://www.adialab.ae/adialab-news/adia-lab-crunchdao-competition-launch) · [quantiacs.com](https://quantiacs.com/) · [quantconnect.com](https://www.quantconnect.com/) · [QuantConnect Alpha Streams](https://www.quantconnect.com/docs/alpha-streams/overview) · [github.com/quantopian/research_public](https://github.com/quantopian/research_public) · [Jane Street Kaggle](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) · [Optiver Kaggle](https://www.kaggle.com/competitions/optiver-trading-at-the-close) · [Citadel datathons](https://www.citadel.com/careers/programs-and-events/datathons/) · [IMC Prosperity](https://prosperity.imc.com/) · [Rotman RITC](https://www.rotman.utoronto.ca/) · [TradersClub/B3 (TNT)](https://lp.tradersclub.com.br/tnt) · [Copa BTG Trader](https://lp.btgpactual.com/copa-btg-trader) · [openquant.co](https://openquant.co/blog/trading-competitions)

**Keywords:** quant competitions, crowdsourced alpha, alfa crowdsourcing, Numerai, NMR staking, Numerai Signals, Numerai Crypto, Erasure, WorldQuant BRAIN, Alphathon, formulaic alphas, International Quant Championship, CrunchDAO, ADIA Lab, market prediction competition, Quantiacs, capital allocation, QuantConnect Alpha Streams, Quant League, Quantopian lectures, Jane Street Kaggle, Optiver, Two Sigma, Citadel datathon, Correlation One, IMC Prosperity, Ready Trader Go, Rotman RIT RITC, CME University Trading Challenge, trading competition, torneio de trading, campeonato de trading, B3, TradersClub, Copa BTG Trader, data science tournament, meta-model, USDC payouts.
