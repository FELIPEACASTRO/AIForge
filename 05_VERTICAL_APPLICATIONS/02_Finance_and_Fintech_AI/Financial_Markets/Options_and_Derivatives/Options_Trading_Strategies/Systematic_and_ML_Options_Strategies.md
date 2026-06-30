# Systematic & ML-Driven Options Strategies

> Turning discretionary options strategies into **rules-based, quant, and ML-driven** systems — CBOE strategy-benchmark indices, variance-risk-premium (VRP) harvesting, vol-targeted premium selling, signal-driven strategy selection, deep hedging / RL, and how to **backtest options honestly**. Education/research only — **not investment advice**.

---

The leap from "I sold a put because IV felt high" to a **systematic** options program means: (1) fixing rules for *entry, sizing, and exit*; (2) attaching a measurable *edge* (almost always the volatility/variance risk premium); (3) **backtesting** on clean option data while respecting the many ways options backtests lie; and (4) sizing so a single tail event does not end the program. This page maps the rules-based benchmarks, the signals, the ML frontier, and the tooling. It is the quant counterpart to the discretionary [Strategy Selection & Playbooks](./Strategy_Selection_and_Playbooks.md), [Volatility, Income & Neutral Strategies](./Volatility_Income_and_Neutral_Strategies.md), and [Directional & Spread Strategies](./Directional_and_Spread_Strategies.md) pages.

---

## 1. Rules-Based Systematic Options: the CBOE Strategy Benchmark Indices

CBOE publishes fully **rules-based, transparent** option-overlay indices with 30+ years of daily history — the canonical reference for "does selling premium systematically work, and at what risk?" They are *hypothetical* (no slippage/commissions/capacity) but their methodologies are public and replicable, making them the gold-standard baseline any systematic strategy must beat. Master list: [cboe.com/us/indices/benchmark_indices](https://www.cboe.com/us/indices/benchmark_indices/). All on **S&P 500 (SPX)** unless noted; verified June 2026.

| Ticker | Name | Structure (verified) | Market view | Risk profile |
|---|---|---|---|---|
| **BXM** | S&P 500 BuyWrite | Long SPX + short 1-month **ATM** call, rolled monthly | Mild bull / range | Capped upside, full downside; short vol |
| **BXMD** | S&P 500 30-Delta BuyWrite | Long SPX + short **30-delta** (further OTM) call | Bull / range | Higher upside capture than BXM |
| **BXY** | S&P 500 2% OTM BuyWrite | Long SPX + short **2% OTM** call | Bull / range | More upside than BXM, less premium |
| **PUT** | S&P 500 PutWrite | Cash-secured short **ATM** SPX put, T-bill collateral | Mild bull / range | Equity-like risk, short vol; historically the least volatile equity index |
| **WPUT** | S&P 500 One-Week PutWrite | Weekly ATM put-write | Range | Higher turnover, more frequent premium |
| **PUTR** | Russell 2000 PutWrite | Cash-secured ATM put on RUT | Range (small-cap) | Small-cap vol exposure |
| **CLL** | S&P 500 95-110 Collar | Long SPX + long **5% OTM put** (3-mo) − short **10% OTM call** (1-mo) | Hedged long | Defined downside, capped upside |
| **CLLZ** | S&P 500 Zero-Cost Put-Spread Collar | Self-financing put-spread collar | Hedged long | Cheaper but caps tail protection |
| **CNDR** | S&P 500 Iron Condor | Short ~**0.20-delta** OTM put + OTM call, buy ~0.05-delta wings, T-bill collateral, monthly | Range-bound, short vol | Defined risk; loses on big moves either way |
| **BFLY** | S&P 500 Iron Butterfly | Short **ATM** put + ATM call, buy **5% OTM** put + call wings, T-bill collateral, monthly | Pin near spot | Defined risk; large credit, narrow profit zone |
| **RXM** | S&P 500 Risk Reversal | Short OTM put + long OTM call | Bullish, short skew | Levered-bull, short put tail |

BFLY/CNDR methodology details: [CBOE Insights — Volatility Management with BFLY & CNDR](https://www.cboe.com/insights/posts/benchmark-indices-series-volatility-management-with-cboes-bfly-and-cndr-indices/). Fact sheets / daily values: [BXM dashboard](https://www.cboe.com/us/indices/dashboard/bxm/), [benchmarks fact-sheet PDF](https://cdn.cboe.com/resources/indices/documents/benchmarks-fact-sheet.pdf). **Honest read:** these indices show steady premium income with *equity-like drawdowns* (BXM/PUT fell hard in 2008 and March 2020) — they harvest a real premium but do **not** remove crash risk; PUT and BXM are economically near-identical via put-call parity (see Israelov below).

---

## 2. The Edge: Variance / Volatility Risk Premium (VRP)

Almost every profitable systematic *selling* program harvests the **VRP** — option **implied** volatility has historically exceeded subsequently **realized** volatility, so the seller is paid for bearing the risk that realized > implied (i.e., crashes). VRP = `implied variance − realized variance`, usually positive but **violently negative in tail events**. This is the structural reason BXM/PUT/CNDR earn premium; it is *compensation for risk*, not arbitrage.

| Concept (PT) | Definition | Systematic use |
|---|---|---|
| VRP (*prêmio de risco de variância*) | IV² − RV² over the horizon | Sell premium when large & positive; size down when compressing |
| Volatility risk premium | IV − RV (vol units) | Same signal, vol space |
| IV Rank / IV Percentile (*IV rank*) | Where current IV sits vs its own trailing year | Gate: prefer selling when IVR high (mean-reversion tailwind) |
| Term structure (*estrutura a termo*) | Front vs back IV (contango/backwardation) | Backwardation → stress, often pause/short selling |
| Skew (*assimetria/smirk*) | Put IV vs call IV | Rich put skew → risk-reversals, put-spread financing |
| RV-vs-IV spread | Realized minus implied | Direct VRP estimate for entry timing |

Foundational papers (all real, working URLs):
- Bollerslev, Tauchen, Zhou (2009), *Expected Stock Returns and Variance Risk Premia*, RFS 22(11):4463–4492 — VRP predicts returns: [oup abstract](https://academic.oup.com/rfs/article-abstract/22/11/4463/1565787) · [author PDF (Duke)](https://public.econ.duke.edu/~boller/Published_Papers/rfs_09.pdf) · [Fed FEDS landing page](https://www.federalreserve.gov/econres/feds/expected-stock-returns-and-variance-risk-premia.htm).
- Israelov & Nielsen, *Covered Calls Uncovered* (SSRN) — decomposes BXM into equity + short-vol; short-vol Sharpe ≈ 1 but <10% of risk: [SSRN 2444999](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2444999).
- Israelov & Nielsen, *Covered Call Strategies: One Fact and Eight Myths*: [SSRN 2444993](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2444993).
- Israelov, *PutWrite versus BuyWrite: Yes, Put-Call Parity Holds Here Too* — PUT beat BXM ~1.1%/yr 1986–2015, explained by parity/timing: [SSRN 2894610](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2894610).
- Ang, Israelov, Sullivan, Tummala, *Understanding the Volatility Risk Premium* (AQR): [aqr.com](https://www.aqr.com/Insights/Research/White-Papers/Understanding-the-Volatility-Risk-Premium).
- Quantpedia overview & replication: [Volatility Risk Premium Effect](https://quantpedia.com/strategies/volatility-risk-premium-effect).

> **Be honest:** VRP is real and persistent but **not free money**. It is a *short-volatility carry* — many small wins punctuated by rare large losses (negative skew, fat left tail). Backtests over benign decades flatter it; the payoff resembles selling insurance. EV, not win-rate, is what matters.

---

## 3. Risk-Managed Premium Selling & Vol-Targeting

Naive constant-notional premium selling blows up because losses scale with volatility *exactly when* you are short the most risk. Systematic programs fix this with:

- **Volatility targeting (*meta de volatilidade*)** — scale position size inversely to current/forecast vol (or VIX) so portfolio risk ≈ constant; cut exposure as IV spikes. This converts the lumpy short-vol P&L into a steadier stream and reduces tail bleed.
- **Delta management** — keep the book near delta-neutral (or a chosen target) by rolling/hedging; isolates the vol bet from the direction bet.
- **Mechanical entry/exit rules** — the widely-cited **tastylive/tastytrade** house framework: enter ~**45 DTE**, **manage winners at ~50% of max profit**, and **exit/roll near 21 DTE** to dodge accelerating gamma risk into expiry ([tastytrade — Expiration risk / managing before expiry](https://support.tastytrade.com/support/s/solutions/articles/43000484765); IV Rank gating, sell when IVR elevated). These are *defaults*, not laws — backtest them on your own data and costs.
- **Defined-risk by construction** — prefer spreads/condors/butterflies (CNDR/BFLY style) over naked options so a single gap cannot exceed the wing width.

Honest counterpoint on the limits of "managing" tail risk and the cost of protection: AQR, *Tail Risk Hedging: Contrasting Put and Trend Strategies* ([PDF](https://images.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-Tail-Risk-Hedging-Contrasting-Put-and-Trend-Strategies.pdf)); Israelov & Tummala, *Being Right is Not Enough: Buying Options to Bet on Higher Realized Volatility* ([SSRN 3248500](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3248500)).

---

## 4. Signals → Strategy Selection (the systematic decision layer)

A systematic options program maps **observable signals** to a strategy and to position size. The matrix below is the rules-engine skeleton; thresholds must be fit/validated per underlying, not copied.

| Signal | Reading | Implied action | Caveat |
|---|---|---|---|
| IV Rank / Percentile high | Premium rich | Sell defined-risk premium (condor/credit spread/put-write) | High IVR can persist or spike higher in a crisis |
| IV Rank low | Premium cheap | Buy premium / debit spreads / calendars; avoid selling | Low IV can stay low for years |
| Term structure backwardation | Front > back IV (stress) | Reduce/halt short vol; consider long gamma | Backwardation = the regime where sellers lose most |
| Steep put skew | Downside fear priced | Put-spread financing, risk-reversal, ratio | Skew is rich *because* tails happen |
| RV > IV recently | VRP negative/compressed | Stand aside or buy vol | RV is backward-looking |
| Forecast RV << IV (model) | Positive expected VRP | Size up short vol within vol target | Model risk / regime change |

This is exactly where ML enters: a **vol-forecast model** (HAR-RV, GARCH-family, LSTM/TFT) produces expected realized vol; the *gap* vs the current IV surface drives the sell/buy decision and the sizing. See the repo's prediction pages (plain text, not external): `Options_Market_Prediction/Models_and_Innovative_Techniques_for_Options.md` and `Options_Market_Prediction/Features_for_Options_Prediction.md` for vol forecasting, IV-surface modeling, and feature engineering (IV rank, skew, GEX, term structure).

---

## 5. ML / RL for Options

ML rarely "predicts price." Where it adds genuine value in options is **hedging, market making, surface modeling, and vol forecasting → strategy mapping**. Real, verified literature:

| Topic | Reference | Working URL |
|---|---|---|
| **Deep Hedging** — NN learns hedge under transaction costs/frictions, beats delta-hedge in incomplete markets | Buehler, Gonon, Teichmann, Wood (2019), *Quantitative Finance* 19:1271–1291 | [RePEc/IDEAS](https://ideas.repec.org/a/taf/quantf/v19y2019i8p1271-1291.html) (DOI 10.1080/14697688.2019.1571683) |
| Deep hedging + generative market simulation | Wiese, Bai, Wood, Buehler, *Deep Hedging: Learning to Simulate Equity Option Markets* | [SSRN 3470756](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3470756) · [arXiv 1911.01700](https://arxiv.org/abs/1911.01700) |
| Deep Bellman / RL hedging across risk aversions | Buehler et al., *Deep Bellman Hedging* | [arXiv 2207.00932](https://arxiv.org/abs/2207.00932) |
| Deep gamma hedging | *Deep Gamma Hedging* (2024) | [arXiv 2409.13567](https://arxiv.org/abs/2409.13567) |
| **RL market making (equities)** | Spooner, Fearnley, Savani, Koukorinis, *Market Making via Reinforcement Learning* (AAMAS) | [arXiv 1804.04216](https://arxiv.org/abs/1804.04216) · code: [github.com/tspooner/rl_markets](https://github.com/tspooner/rl_markets) |
| Robust/adversarial RL market making | Spooner & Savani (2020) | [arXiv 2003.01820](https://arxiv.org/abs/2003.01820) |
| **RL option market making** | Fang & Xu (2023, rev. 2025), *Option Market Making via Reinforcement Learning* | [arXiv 2307.01814](https://arxiv.org/abs/2307.01814) |
| RL for quantitative trading (survey) | Sun et al. (2021) | [arXiv 2109.13851](https://arxiv.org/abs/2109.13851) |

**Honest framing:** deep hedging is the most production-credible of these (it solves a well-posed optimization with no look-ahead). "RL picks my strategy / prints money" claims are usually overfit; the action space is huge, rewards are sparse and tail-dominated, and live ≠ backtest. Treat ML as a *vol-forecast/hedging* enhancer to a VRP core, not as alpha by itself.

---

## 6. Backtesting Options Systematically

Options backtests are **far** harder than equity backtests: you need per-strike, per-expiry quotes (bid/ask, IV, Greeks), and most of the failure modes are data-quality, not strategy.

| Tool | What it is | Notes / URL |
|---|---|---|
| **optopsy** | Nimble options research/backtest lib; 38 built-in structures (condors, butterflies, calendars, diagonals, ratios) | Active fork [goldspanlabs/optopsy](https://github.com/goldspanlabs/optopsy); historical [michaelchu/optopsy](https://github.com/michaelchu/optopsy), [PyPI](https://pypi.org/project/optopsy/) |
| **OptionSuite** | Modular options/stock backtester → live; SPX EOD focus | [github.com/sirnfs/OptionSuite](https://github.com/sirnfs/OptionSuite) (requires your own data, e.g. iVolatility) |
| **QuantConnect / LEAN** | Open-source event-driven engine; same code backtest→live; equities/options/futures/crypto; IBKR/Tradier brokers | [github.com/QuantConnect/Lean](https://github.com/QuantConnect/Lean) · [docs](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting) |
| **backtrader** | General event-driven Python framework; options via custom data/contracts | [github.com/mementum/backtrader](https://github.com/mementum/backtrader) |
| **vectorbt** | Numba-vectorized fast signal triage ("does this even have alpha?") — port winners to event-driven for realistic fills | [github.com/polakowo/vectorbt](https://github.com/polakowo/vectorbt) |
| **ib_async / ib_insync** | Python sync/async IBKR API; option chains, live execution | [ib-api-reloaded/ib_async](https://github.com/ib-api-reloaded/ib_async) (maintained successor) · legacy [erdewit/ib_insync](https://github.com/erdewit/ib_insync) |

### Pitfalls (the part most backtests get wrong)
- **Data quality** — stale/wide quotes, mid-price fills that are unachievable, missing illiquid strikes; options bid/ask spreads are *wide* and dominate net P&L. Always model the spread, not the mid.
- **Early exercise / assignment (*exercício antecipado*)** — American options (and dividends) cause early assignment on short ITM calls/puts; index options (SPX) are European/cash-settled — do not mix the assumptions.
- **Liquidity & capacity** — a backtest that trades the whole chain ignores that you cannot fill size at the screen; 0DTE/weeklies look great until slippage is modeled.
- **Survivorship & look-ahead** — using settlement IV/Greeks that were not known at the trade time, or only surviving tickers, inflates results.
- **Transaction costs & IV crush** — commissions, exchange fees, and the **IV crush** after earnings/events; a "winning" long straddle into earnings often loses on the vol collapse.
- **Validation** — apply the repo's [Backtesting & Frameworks](../../Backtesting_and_Frameworks/) discipline: purged/embargoed CV, deflated Sharpe, out-of-sample, and a benchmark vs the relevant CBOE index (don't claim alpha you can't beat BXM/PUT with).

---

## 7. Datasets (see existing pages for full detail)

| Dataset | Coverage | Access |
|---|---|---|
| **OptionMetrics IvyDB** | US/global equity & index options, IV surface, Greeks, 1996+ | Paid/academic |
| **ORATS** | EOD + intraday options, IV, signals, backtester data | Paid; [orats.com](https://orats.com) |
| **CBOE DataShop** | Official US options & index data | Paid; free VIX/SKEW history |
| **Deribit** | Crypto (BTC/ETH) options — best **free** proxy for research | Free API; [deribit.com](https://www.deribit.com) |
| **B3 / OpLab / opcoes.net.br** | Brazilian listed options data & analytics | See B3 section below |

Full comparison, APIs, and free/paid split: `Options_Market_Prediction/Datasets_and_Data_Sources_for_Options.md` and [Datasets & Data Sources](../Options_Market_Prediction/Datasets_and_Data_Sources_for_Options.md).

---

## 8. Brazil / B3 specifics (*especificidades B3*)

- **Underlyings:** liquid single-name equity options (*opções sobre ações* — PETR4, VALE3, BBAS3, BOVA11) and **index options on Ibovespa** (*opções sobre o Ibovespa*, IBOV). Liquidity concentrates near expiry and in a few names — systematic selling must model thin chains and wide spreads carefully.
- **Style/settlement:** Brazilian equity options are typically **American-style** (early-exercise risk on dividends — *exercício antecipado* around *proventos*); Ibovespa index options follow B3 contract specs (verify settlement per series). This breaks naive European/cash-settled backtest assumptions.
- **Analytics platforms:** **OpLab** ([oplab.com.br](https://oplab.com.br/), market portal [opcoes.oplab.com.br](https://opcoes.oplab.com.br/mercado)) — payoff, vol surfaces/heatmaps, IV history, scenario analysis, portfolio Greeks, and an API for systematic workflows; **opcoes.net.br** ([opcoes.net.br](https://opcoes.net.br/)) — chains, OI, and strategy screening.
- **Data for backtests:** B3 market data via MT5/Cedro feeds, OpLab API, or opcoes.net.br history; quality and depth are below US vendors — validate harder. See [B3 Options & Derivatives (Brazil)](../B3_Options_and_Derivatives_Brazil.md).

---

## 9. A minimal systematic VRP loop (skeleton, not a recommendation)

1. **Universe:** liquid index/ETF options (SPX/SPY, or IBOV/BOVA11 on B3).
2. **Signal:** estimate expected RV (HAR-RV/GARCH/ML) vs current ATM IV → expected VRP; require positive VRP **and** elevated IV Rank.
3. **Structure:** defined-risk short premium (iron condor / put-write with protective wing) at ~45 DTE.
4. **Sizing:** volatility-target the book; cap risk per trade to wing width; total short-vega budget capped.
5. **Management:** take profit ~50% max, roll/exit ~21 DTE, hard stop on term-structure backwardation.
6. **Validation:** backtest with modeled spreads/costs/assignment; purged CV; benchmark vs **PUT/CNDR**; report deflated Sharpe and worst drawdown, **not** win-rate.

---

## Related in AIForge
- Siblings: [Strategy Selection & Playbooks](./Strategy_Selection_and_Playbooks.md) · [Volatility, Income & Neutral Strategies](./Volatility_Income_and_Neutral_Strategies.md) · [Directional & Spread Strategies](./Directional_and_Spread_Strategies.md)
- Parent: [Options & Derivatives](../) · [Options Market Prediction](../Options_Market_Prediction/) · [Strategies & Analytics Tools](../Options_Strategies_and_Analytics_Tools.md) · [B3 Options (Brazil)](../B3_Options_and_Derivatives_Brazil.md) · [US Options & Derivatives](../US_Options_and_Derivatives.md)
- Discipline: [Backtesting & Frameworks](../../Backtesting_and_Frameworks/) · [Risk Management & Derivatives Pricing](../../Risk_Management_and_Derivatives_Pricing/)

> **Not investment advice.** This is research/education. Options carry undefined-risk tails, assignment and early-exercise risk, liquidity and IV-crush risk; systematic short-volatility programs can lose many years of premium in a single crash. No strategy is free money. Backtest with realistic costs and size for survival.

**Keywords:** systematic options, variance risk premium, VRP harvesting, volatility risk premium, CBOE BXM PUT CNDR BFLY CLL, put-write buy-write, iron condor butterfly index, vol targeting, IV rank skew term structure, deep hedging, reinforcement learning options market making, optopsy OptionSuite LEAN QuantConnect vectorbt ib_async, options backtesting pitfalls, opções sistemáticas, prêmio de risco de variância, volatilidade implícita, venda coberta lançamento coberto, B3 OpLab opcoes.net.br Ibovespa, exercício antecipado.
