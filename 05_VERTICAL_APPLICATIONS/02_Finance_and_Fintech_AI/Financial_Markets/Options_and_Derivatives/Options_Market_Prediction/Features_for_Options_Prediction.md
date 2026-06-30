# Features for Options-Market Prediction

> A dense, sourced catalogue of the signals (Greeks, IV-surface shape, vol indices, options flow, dealer-gamma, microstructure, events) used to forecast implied/realized volatility, option prices/returns, direction, and to drive hedging — with honest notes on low signal-to-noise, no-arbitrage constraints, look-ahead leakage, and data cost. Global / ML-quant focus, 2024–2026, with B3 (Brazil) notes.

This page is the **feature/signal companion** to the modeling pages in this folder. It assumes you already have a clean point-in-time option chain (strikes × expiries × calls/puts, bid/ask/last, volume, open interest) plus the underlying spot, a risk-free/forward curve, and dividends. The "label" you are predicting (next-day ATM IV, realized vol over `[t, t+h]`, option return, underlying direction, optimal hedge) determines which features are predictive vs. circular.

For the raw data feeds referenced throughout, see the datasets pages: [Global_Datasets_for_Markets.md](../../Models_Features_and_Datasets/Global_Datasets_for_Markets.md), [Brazil_B3_Market_Data_APIs.md](../../Datasets_APIs_and_Data_Vendors/Brazil_B3_Market_Data_APIs.md), and the vendor list below.

---

## 0. Mental model: what each feature actually is

| Concept | One-line meaning | Forward/backward looking | Primary use as a feature |
|---|---|---|---|
| **Implied volatility (IV)** | Vol that makes Black–Scholes/forward price = market price | **Forward-looking** (risk-neutral expectation) | Predict RV, surface dynamics, returns |
| **Realized volatility (RV)** | Vol actually realized from underlying returns | Backward-looking (sample) | Target, and the "physical" leg of VRP |
| **Greeks** | Local sensitivities of price to spot/vol/time/rate | Snapshot (derived from IV) | Risk normalization, hedging, exposure aggregation |
| **VRP** | IV² − E[RV²]: compensation for variance risk | Mostly forward (IV) minus realized | Return/vol predictor; "why option sellers earn" |
| **Dealer gamma (GEX)** | $ hedging flow per 1% move implied by dealer book | Snapshot (positioning) | Regime/vol-of-vol context, **not** a clean alpha |

Two non-negotiable disciplines, stressed throughout: **(a) point-in-time / no leakage** — IV, OI, and flow must be timestamped *before* the prediction window; OI in particular is an end-of-day number and is revised; **(b) no-arbitrage** — surface features must come from a chain that is (or has been cleaned to be) free of calendar and butterfly arbitrage, or models learn artifacts.

---

## 1. Greeks & sensitivities as features

Greeks are derivatives of the option value w.r.t. inputs. First-order: **delta** (∂V/∂S), **vega** (∂V/∂σ), **theta** (∂V/∂t), **rho** (∂V/∂r). Second/third-order ("higher-order Greeks"): **gamma** (∂²V/∂S²), **vanna** (∂²V/∂S∂σ, also ∂delta/∂σ), **volga/vomma** (∂²V/∂σ²), **charm** (∂delta/∂t, "delta decay"), **speed**, **color**, **zomma**. Definitions and formulae: [Wikipedia — Greeks (finance)](https://en.wikipedia.org/wiki/Greeks_(finance)); vendor-computed Greeks ship with OptionMetrics IvyDB (delta, gamma, vega, theta) — [optionmetrics.com](https://optionmetrics.com/).

| Greek | Symbol | Sensitivity | Feature usage |
|---|---|---|---|
| Delta | Δ | ∂V/∂S | Moneyness proxy; bucketing strikes (25Δ, ATM, 75Δ); directional exposure |
| Gamma | Γ | ∂²V/∂S² | Convexity; drives dealer-hedging flow (→ GEX); peaks ATM near expiry |
| Vega | ν | ∂V/∂σ | Vol exposure; normalize IV moves; vega-weighted surface fitting |
| Theta | Θ | ∂V/∂t | Carry/decay; short-vol P&L driver |
| Rho | ρ | ∂V/∂r | Rate sensitivity; matters for long-dated / high-rate (e.g. B3 Selic) |
| Vanna | — | ∂Δ/∂σ | Skew-driven hedging; "vanna rally" narratives near OPEX |
| Volga/Vomma | — | ∂ν/∂σ | Convexity of vega; pricing of vol-of-vol, smile wings |
| Charm | — | ∂Δ/∂t | Delta decay into expiry; intraday/overnight hedging flow |

**As features, use Greeks for normalization, not as raw alpha.** Examples: vega-normalize IV changes; delta-bucket the surface so a "25Δ put IV" is comparable across names and time; aggregate `gamma × OI × contract-multiplier × spot²` across the chain to build positioning features (Section 5). Greeks are model-dependent (Black–Scholes vs. local/stochastic vol give different vanna/volga), so fix one convention and keep it point-in-time.

---

## 2. Implied-volatility-surface features

The IV surface σ(K, T) — or in coordinates (log-moneyness `k = ln(K/F)`, maturity τ) — is the single richest feature source. Cont & da Fonseca's classic result is that surface dynamics are well described by a **level–slope–curvature (skew–convexity)** factor structure ([Cont & da Fonseca, *Dynamics of Implied Volatility Surfaces*, 2002, PDF](http://rama.cont.perso.math.cnrs.fr/pdf/ImpliedVolDynamics.pdf)). Arbitrage-free fitting/parametrization (SVI, SSVI) is the standard front-end; Gatheral & Jacquier, *Arbitrage-free SVI volatility surfaces* ([arXiv:1204.0646](https://arxiv.org/abs/1204.0646)).

| Feature | Definition | Notes / pitfalls |
|---|---|---|
| **ATM IV** | IV at k≈0 for fixed τ (often 30d, "IV30") | The "level"; interpolate, don't snap to nearest listed strike |
| **Skew / smile slope** | ∂σ/∂k near ATM, or `25Δ put IV − 25Δ call IV` (risk reversal) | Sign/convention varies; equity index skew is steeply negative |
| **IVSkew / volatility smirk** | `OTM put IV − ATM call IV` | Steep smirk negatively predicts stock returns — Xing, Zhang & Zhao, *What Does the Individual Option Volatility Smirk Tell Us About Future Equity Returns?*, [JFQA 45 (2010) 641–662](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/what-does-the-individual-option-volatility-smirk-tell-us-about-future-equity-returns/ECFD16BA9ACBDC8D577D1BD866FBEA72). See also the structural cross-sectional skew model of Wu & Tian, [*Cross-Sectional Variation of Option-Implied Volatility Skew*, Management Science 70 (2024)](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.4872). Noisy 2-option measure — weaker once proper risk-neutral skew is controlled |
| **Term-structure slope** | `IV(long τ) − IV(short τ)`, e.g. 3m−1m | Contango vs. backwardation = vol regime signal |
| **IV Rank / IV Percentile** | Rank/percentile of current IV vs. trailing 1y | Cheap, robust mean-reversion feature; define window point-in-time |
| **Surface PCA factors** | PC1/PC2/PC3 ≈ level / slope / curvature | First 3 modes explain most variance (Cont–da Fonseca); fit PCA on past data only |
| **Butterfly / convexity** | `(25Δ put IV + 25Δ call IV)/2 − ATM IV` | Smile curvature; tied to risk-neutral kurtosis |
| **VRP** | Section 3 | The most-studied option-derived return/vol predictor |

**ML on the whole surface (2024–2026):** treat the surface as an image/field and forecast it one step ahead, under arbitrage constraints. Examples with code/repos: *Forecasting implied volatility surface with generative diffusion models* (DDPM, arbitrage-free, one-day-ahead) — [arXiv:2511.07571](https://arxiv.org/abs/2511.07571) (Jin & Agarwal, 2025), [code](https://github.com/Austinjinc/rep_volgan/tree/refactor/new-ddpm); *A Two-Step Framework for Arbitrage-Free Prediction of the IVS* — [arXiv:2106.07177](https://arxiv.org/abs/2106.07177); *Controllable Generation of IV Surfaces with VAEs* — [arXiv:2509.01743](https://arxiv.org/abs/2509.01743); *The implied volatility surface (also) is path-dependent* — [arXiv:2312.15950](https://arxiv.org/abs/2312.15950); *SABR-Informed Multitask Gaussian Process* — [arXiv:2506.22888](https://arxiv.org/abs/2506.22888).

---

## 3. Volatility-risk-premium (VRP) and the IV-vs-RV spread

**VRP = (option-implied variance) − (expected realized variance).** It is the most documented option-derived predictor: IV systematically exceeds subsequent RV (you are paid to *sell* variance), and the one-month VRP predicts equity-index returns at monthly/quarterly horizons (Bollerslev, Tauchen & Zhou). Practitioner explainers: [PredictingAlpha — Variance Risk Premium](https://www.predictingalpha.com/variance-risk-premium/), [SpotGamma — VRP](https://support.spotgamma.com/hc/en-us/articles/15249783047315-VRP-Variance-Risk-Premium). The premium is generally **positive** because vol spikes coincide with crashes (negative correlation = expensive insurance), and it has a pronounced **downside** component (Federal Reserve FEDS 2015-020, [PDF](https://www.federalreserve.gov/econresdata/feds/2015/files/2015020pap.pdf)).

To build VRP you need a clean RV estimate. Close-to-close vol throws away intraday range; **range/OHLC estimators** are far more efficient.

| RV estimator | Inputs | Property | Reference |
|---|---|---|---|
| Close-to-close | C | Simplest, noisy | baseline |
| Parkinson | H, L | ~5× efficiency vs C2C; no overnight gap, no drift | classic |
| Garman–Klass | O,H,L,C | Adds open/close; assumes no jumps/overnight | [PyQuant — 6 ways](https://www.pyquantnews.com/the-pyquant-newsletter/how-to-compute-volatility-6-ways) |
| Rogers–Satchell | O,H,L,C | **Drift-independent** | classic |
| **Yang–Zhang** | O,H,L,C | Handles overnight jumps + drift; min-variance combo of overnight + Rogers–Satchell; `k = 0.34/(1.34 + (n+1)/(n−1))` | [Yang & Zhang RV in Python, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2665963824000010) |
| GKYZ | O,H,L,C | Garman–Klass + overnight term | [TradingView GKYZ](https://www.tradingview.com/script/kTcoVSiN-Garman-Klass-Yang-Zhang-Volatility-Estimator/) |

For high-frequency data, use **realized variance / bipower variation / realized kernels** (jump-robust) instead. The IV leg can be a single ATM IV, a VIX-style model-free integral, or per-strike. **VRP feature design tips:** compute IV and RV on the *same horizon*; lag RV so it is strictly past data at prediction time; report VRP in variance and vol units (they behave differently); beware that VRP collapses/inverts in crises (IV < RV), which is itself a regime signal. Recent intraday vs overnight VRP decomposition: [Papagelis, *VRP over trading and nontrading periods*, JFM 2025](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22589).

---

## 4. Volatility indices & cross-asset risk

Model-free, exchange-published vol indices are ready-made features (and sometimes targets). All CBOE methodologies are public.

| Index | What it measures | Source |
|---|---|---|
| **VIX** | 30-day model-free implied variance of S&P 500 | [CBOE VIX Methodology PDF](https://cdn.cboe.com/resources/vix/VIX_Methodology.pdf) |
| **VVIX** | Vol-of-vol: 30-day expected vol of VIX, from VIX options | [CBOE VVIX term-structure PDF](https://cdn.cboe.com/resources/indices/documents/vvix-termstructure.pdf) |
| **VIX term structure** | VIX9D / VIX / VIX3M / VIX6M, futures curve | contango/backwardation regime |
| **SKEW** | Risk-neutral tail/skewness from OTM SPX puts (≈100–150; 100 = normal) | [CarryTrader — CBOE indexes](https://carrytrader.com/carry-trade/cboe-volatility-indexes) |
| **MOVE** | Implied vol of US Treasuries (the "bond VIX", ICE/BofA) | rates-vol cross-asset feature |
| **Put/Call & VRP indices** | sentiment / variance premium | Sections 3, 5 |

Cross-asset usage: VVIX and the VIX term-structure slope are strong **vol-of-vol / regime** features; SKEW proxies crash fear; MOVE links rates vol to equity vol. B3 analogue: the **S&P/B3 Ibovespa VIX** (30-day implied vol of Ibovespa from Ibovespa option mid prices, launched March 2024 on Cboe's VIX methodology) — [B3 Volatility Indices](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indices-in-partnership-with-s-p-dowjones/volatility-indices.htm), [methodology PDF](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-b3-ibovespa-vix.pdf). Caveat: index levels are non-stationary and autocorrelated — model changes/innovations, standardize point-in-time, and never regress raw level on raw level without care.

---

## 5. Options flow, positioning & dealer-gamma features

These come from volume, open interest, and trade prints. They split into **flow/sentiment** (relatively well-evidenced, modest signal) and **dealer-positioning/gamma** (mechanically real, predictively contested).

### 5a. Flow & positioning

| Feature | Definition | Evidence (honest) |
|---|---|---|
| **Put/Call ratio — volume** | put vol / call vol | Short-horizon (≈days) predictor; contrarian at extremes. [MDPI Economies 7(1):24](https://www.mdpi.com/2227-7099/7/1/24) |
| **Put/Call ratio — OI** | put OI / call OI | Predictive at longer (≈weeks) horizons than volume PCR (same study) |
| **OI changes (call vs put)** | Δ open interest | Rising call OI → higher subsequent returns; effect attenuates with controls. [Why does option OI predict stock returns?, AUT](https://acfr.aut.ac.nz/__data/assets/pdf_file/0004/686830/1b-Yi-Zhou.pdf) |
| **O/S ratio** | option volume / stock volume | Weekly/monthly predictability; complements PCR |
| **Unusual options activity / sweeps / dark flow** | large, aggressive, multi-exchange prints | Vendor-defined, noisy, survivorship-/selection-prone; treat skeptically |

Why any of this works: a strand of literature argues **information appears in options first** (leverage, short-sale relief), so option order flow can lead the stock. But effect sizes are small, decay fast, and are easy to overfit.

### 5b. Dealer-gamma / positioning features (and the debate)

**GEX (Gamma Exposure)** = aggregate $ that delta-neutral dealers must trade per 1% underlying move, summed over the chain (signing puts/calls by assumed dealer position). **Positive GEX** → dealers sell rallies / buy dips → *dampened* vol; **negative GEX** → dealers amplify moves → *higher* vol. Related: **vanna/charm exposure** (skew- and time-driven hedging, the "OPEX/vanna" narrative), **zero-gamma / gamma-flip level**, and **max pain** (strike minimizing total ITM option value at expiry).

| Feature | Construction | Honest verdict |
|---|---|---|
| **GEX / dealer gamma** | Σ `Γ · OI · 100 · S² · 0.01 · sign` | Mechanically real (hedging exists); a **regime/vol context**, not a clean alpha. 8-yr SPY backtest: GEX mostly tracks VIX, thin standalone edge — [FlashAlpha](https://flashalpha.com/articles/gex-dex-vex-chex-8-year-backtest-spy-vix-control). Definition: [SpotGamma GEX](https://support.spotgamma.com/hc/en-us/articles/15214161607827-GEX-Gamma-Exposure-Explained-What-It-Is-and-How-SpotGamma-Uses-It) |
| **Zero-gamma / flip level** | spot where aggregate dealer GEX = 0 | Dynamic threshold, not fixed support/resistance |
| **Vanna / charm exposure** | Σ vanna·OI, Σ charm·OI | Drives systematic hedging into expiry; hard to isolate empirically |
| **Max pain** | strike minimizing aggregate option payout | Descriptive, weak/contested as a predictor; any predictability looks more like a post-decline reversal than pinning toward max pain. [SSRN 4140487 (Filippou, Garcia-Ares & Zapatero)](https://papers.ssrn.com/sol3/Delivery.cfm/4140487.pdf?abstractid=4140487&mirid=1) |

**The catch:** dealer-side sign of OI is *unobservable* — every GEX/vanna number relies on an assumption (e.g. "dealers are short customer puts, long customer calls") that vendors implement differently, so the same chain yields different GEX across providers. Use these as **conditioning/regime features** (is vol likely to be pinned or amplified?) rather than as directional predictors, and validate that any edge survives controlling for VIX.

---

## 6. Microstructure features

From the order book and quotes — most relevant for intraday option-return/RV prediction and execution.

| Feature | Definition | Note |
|---|---|---|
| **Bid–ask spread (and %)** | `ask − bid`, `/mid` | Liquidity & cost; wide in OTM/long-dated; gates tradability |
| **Quote / OI imbalance** | (bid size − ask size)/(sum); call vs put pressure | Short-horizon pressure signal |
| **Option order-flow imbalance (OFI)** | signed aggressive volume | Microstructure analogue of equity OFI |
| **Effective/realized spread** | trade vs mid; mid drift after | Execution-quality & informed-flow proxy |
| **Mid vs last, stale-quote flags** | quote freshness | **Critical for IV:** compute IV off *synchronized* mids; stale quotes create phantom arbitrage and corrupt Greeks/surface features |

Honest note: option microstructure data is voluminous (every strike × expiry × quote update), expensive, and noisy. For most prediction tasks, end-of-day cleaned surfaces beat raw quote noise unless you genuinely operate intraday.

---

## 7. Cross-asset & event features

Volatility is event-driven; an "event term structure" beats calendar time.

| Feature | Definition | Use |
|---|---|---|
| **Earnings flag / countdown** | days to next earnings | IV ramps pre-earnings, **crushes** after ("IV crush") — model RV/option-return around it |
| **Implied-move from straddle** | ATM straddle price / spot | Market's expected event jump; compare to realized |
| **Macro-event calendar** | FOMC, CPI, NFP; in BR: **COPOM (Selic)**, IPCA | Scheduled vol; term structure of events |
| **Event term structure** | front-expiry IV vs back, isolating event | Decompose diffusive vs jump vol |
| **Cross-asset vol** | VVIX, MOVE, FX vol, commodity vol | Spillovers; for B3, USD/BRL and commodity vol matter (Petrobras/Vale weight) |

---

## 8. Starter feature set (a concrete, leakage-aware baseline)

A compact set that is cheap to compute, point-in-time, and covers the families above. Horizon `h` is your label horizon; all features use data with timestamp ≤ `t`.

| # | Feature | Family | Construction (point-in-time) |
|---|---|---|---|
| 1 | ATM IV (30d, interpolated) | Surface level | interpolate σ at k=0, τ=30d |
| 2 | IV Rank (1y) | Surface level | percentile of #1 vs trailing 252d |
| 3 | 25Δ risk reversal | Skew | `25Δ put IV − 25Δ call IV` |
| 4 | Smile butterfly | Curvature | `(25Δp+25Δc)/2 − ATM` |
| 5 | Term slope 3m−1m | Term structure | IV(90d) − IV(30d) |
| 6 | Surface PC1–PC3 | Surface factors | PCA fit on past surfaces only |
| 7 | VRP (1m) | VRP | IV30² − YangZhang RV²(past 21d) |
| 8 | Yang–Zhang RV (10d, 21d) | Realized vol | OHLC estimator, lagged |
| 9 | VIX / VVIX / SKEW (or B3 VIX) | Vol indices | level + 1d change, standardized |
| 10 | VIX term slope | Vol indices | VIX3M − VIX (or futures curve) |
| 11 | Put/Call ratio (vol & OI) | Flow | both ratios, smoothed |
| 12 | ΔOI call vs put | Positioning | 1–5d open-interest change |
| 13 | Net GEX + zero-gamma dist | Dealer gamma | regime feature; flag vendor assumption |
| 14 | ATM bid–ask % | Microstructure | spread/mid of front ATM |
| 15 | Days-to-earnings / implied move | Events | countdown + ATM straddle/spot |
| 16 | Macro-event flag (FOMC/COPOM) | Events | next scheduled event within h |

Targets to pair with it: next-day ATM IV, RV over `[t,t+h]`, sign/return of the underlying, delta-hedged option P&L, or the full next-day surface.

---

## 9. Honest difficulty notes (read before backtesting)

- **Low signal-to-noise.** Most option-derived return predictors have small, regime-dependent effects that decay fast; many fail to survive transaction costs (spreads on OTM/long-dated options are wide).
- **IV is forward-looking — circularity risk.** Predicting an option's own price from its IV is near-tautological. Predict the *underlying*, *future RV*, or *surface change* instead, and keep the predictor and target on the same clock.
- **No-arbitrage constraints.** Surfaces must be calendar- and butterfly-arbitrage-free; otherwise PCA/ML learn artifacts and "predict" impossible states. Constrain generators (SVI/SSVI, soft penalties, or arbitrage-free architectures).
- **Point-in-time & leakage.** OI is end-of-day and revised; vendor IV/Greeks may be back-filled; settlement prices differ from tradable mids; corporate actions and contract adjustments break OI/strike series. Snapshot everything as it was known.
- **Dealer-positioning is an assumption, not data.** GEX/vanna/max-pain depend on an unobservable dealer sign; treat as regime context, validate against VIX, and expect them to fail on news-driven moves.
- **Data cost & survivorship.** Quality history (OptionMetrics IvyDB, ORATS, CBOE DataShop) is expensive; free sources are sparse and gap-ridden. B3 option chains are liquid mainly in PETR4, VALE4/VALE3 and Ibovespa options — beware thin strikes and stale quotes elsewhere.
- **Overfitting.** Hundreds of surface/flow features × short samples × regime shifts → high overfitting risk. Prefer few robust features, walk-forward/purged-CV, and out-of-regime tests.

---

## 10. Data sources (brief)

| Source | Coverage | Link |
|---|---|---|
| OptionMetrics **IvyDB** | US/global options, IV + Greeks from 1996 | [optionmetrics.com](https://optionmetrics.com/) |
| **ORATS** | vol surfaces, earnings forecasts, historical Greeks, API | [orats.com](https://orats.com/) · [curated list: awesome-options-analytics](https://github.com/FlashAlpha-lab/awesome-options-analytics) |
| **CBOE DataShop** | exchange options, VIX/settlement, IV indices | [datashop.cboe.com](https://datashop.cboe.com/data-products) |
| CBOE historical | end-of-day options data | [cboe.com historical](https://www.cboe.com/us/options/market_statistics/historical_data/) |
| **B3** | Brazil options, Ibovespa VIX, statistics | [b3.com.br](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indices-in-partnership-with-s-p-dowjones/volatility-indices.htm) |
| Repo datasets pages | curated global + B3 feeds | [Global](../../Models_Features_and_Datasets/Global_Datasets_for_Markets.md) · [B3](../../Datasets_APIs_and_Data_Vendors/Brazil_B3_Market_Data_APIs.md) |

See also, in this folder: model methods in [Models_and_Innovative_Techniques_for_Options.md](./Models_and_Innovative_Techniques_for_Options.md) and [Datasets_and_Data_Sources_for_Options.md](./Datasets_and_Data_Sources_for_Options.md), the parent [README](../README.md), and [B3_Options_and_Derivatives_Brazil.md](../B3_Options_and_Derivatives_Brazil.md) / [US_Options_and_Derivatives.md](../US_Options_and_Derivatives.md). For hedging-as-prediction, the deep-hedging line: Buehler, Gonon, Teichmann & Wood, *Deep Hedging* ([arXiv:1802.03042](https://arxiv.org/abs/1802.03042)); Cao, Chen, Hull & Poulos, *Deep Hedging of Derivatives Using RL* ([arXiv:2103.16409](https://arxiv.org/abs/2103.16409)); *Deep Hedging with RL: A Practical Framework for Option Risk Management* ([arXiv:2512.12420](https://arxiv.org/abs/2512.12420)).

---

**Sources:** [Cont–da Fonseca 2002](http://rama.cont.perso.math.cnrs.fr/pdf/ImpliedVolDynamics.pdf) · [Gatheral–Jacquier SVI 1204.0646](https://arxiv.org/abs/1204.0646) · [IVS diffusion 2511.07571](https://arxiv.org/abs/2511.07571) · [Two-step arb-free IVS 2106.07177](https://arxiv.org/abs/2106.07177) · [IVS-VAE 2509.01743](https://arxiv.org/abs/2509.01743) · [Path-dependent IVS 2312.15950](https://arxiv.org/abs/2312.15950) · [SABR-MTGP 2506.22888](https://arxiv.org/abs/2506.22888) · [Xing–Zhang–Zhao smirk, JFQA 2010](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/what-does-the-individual-option-volatility-smirk-tell-us-about-future-equity-returns/ECFD16BA9ACBDC8D577D1BD866FBEA72) · [Wu–Tian skew, MS 2024](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.4872) · [VRP explainer](https://www.predictingalpha.com/variance-risk-premium/) · [Downside VRP FEDS 2015-020](https://www.federalreserve.gov/econresdata/feds/2015/files/2015020pap.pdf) · [VRP trading/nontrading JFM 2025](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22589) · [Yang–Zhang Python](https://www.sciencedirect.com/science/article/pii/S2665963824000010) · [GKYZ](https://www.tradingview.com/script/kTcoVSiN-Garman-Klass-Yang-Zhang-Volatility-Estimator/) · [CBOE VIX method](https://cdn.cboe.com/resources/vix/VIX_Methodology.pdf) · [CBOE VVIX](https://cdn.cboe.com/resources/indices/documents/vvix-termstructure.pdf) · [PCR MDPI](https://www.mdpi.com/2227-7099/7/1/24) · [OI predicts returns AUT](https://acfr.aut.ac.nz/__data/assets/pdf_file/0004/686830/1b-Yi-Zhou.pdf) · [SpotGamma GEX](https://support.spotgamma.com/hc/en-us/articles/15214161607827-GEX-Gamma-Exposure-Explained-What-It-Is-and-How-SpotGamma-Uses-It) · [GEX backtest FlashAlpha](https://flashalpha.com/articles/gex-dex-vex-chex-8-year-backtest-spy-vix-control) · [Max pain SSRN 4140487](https://papers.ssrn.com/sol3/Delivery.cfm/4140487.pdf?abstractid=4140487&mirid=1) · [Greeks Wikipedia](https://en.wikipedia.org/wiki/Greeks_(finance)) · [OptionMetrics](https://optionmetrics.com/) · [CBOE DataShop](https://datashop.cboe.com/data-products) · [B3 Vol Indices](https://www.b3.com.br/en_us/market-data-and-indices/indexes/indices-in-partnership-with-s-p-dowjones/volatility-indices.htm) · [S&P/B3 Ibovespa VIX method](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-b3-ibovespa-vix.pdf) · [Deep Hedging 1802.03042](https://arxiv.org/abs/1802.03042) · [Deep hedging RL 2103.16409](https://arxiv.org/abs/2103.16409) · [Deep Hedging practical 2512.12420](https://arxiv.org/abs/2512.12420)

**Keywords:** implied volatility surface, IV skew (assimetria), term structure (estrutura a termo), variance risk premium / prêmio de risco de variância, realized volatility (volatilidade realizada), Yang–Zhang, Garman–Klass, VIX, VVIX, SKEW, MOVE, IV rank, Greeks (gregas), delta gamma vega theta vanna charm volga, gamma exposure / GEX, dealer gamma (gama do dealer), max pain, put/call ratio (razão put/call), open interest (contratos em aberto), options flow (fluxo de opções), order-flow imbalance, no-arbitrage (não-arbitragem), point-in-time, leakage (vazamento de dados), B3, Ibovespa, COPOM/Selic, deep hedging, PETR4, VALE3
