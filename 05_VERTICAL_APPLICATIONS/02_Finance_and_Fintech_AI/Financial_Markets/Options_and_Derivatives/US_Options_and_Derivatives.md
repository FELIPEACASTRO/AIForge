# US Options & Derivatives (Opções e Derivativos Americanos)

> The deepest, most liquid listed-derivatives market on earth: equity/ETF options, cash-settled index options (SPX), the VIX volatility complex, CME futures (ES, NQ, crude, gold, Treasuries), and the OPRA consolidated feed — with the data and ML toolchain quants actually use.

US listed derivatives split into two regulatory and clearing universes: **securities options** (equity, ETF, and index options), cleared by the **Options Clearing Corporation (OCC)** and overseen by the SEC; and **futures and options on futures**, cleared by exchange CCPs such as **CME Clearing** and overseen by the CFTC. This page covers the contract mechanics, the Greeks and implied volatility, the 0DTE phenomenon, the major futures, and the data/ML stack for modeling them.

---

## 1. Equity & ETF Options (Opções sobre Ações e ETFs)

The standardized US equity option is the workhorse retail and institutional instrument. Per OCC and the *Characteristics and Risks of Standardized Options* disclosure document (ODD), the standard contract represents **100 shares** of the underlying, is **American-style** (exercisable any business day through expiration), and **physically settles** into shares on assignment. Since the May 2024 move to a **T+1** settlement cycle, exercise/assignment now delivers the underlying shares on the first business day (T+1) following exercise. Premiums are quoted in points where **1 point = $100** per contract (a $2.00 premium costs $200). Corporate actions (splits, mergers, special dividends) can produce *adjusted* contracts that no longer deliver exactly 100 shares.

| Attribute | Standard Equity/ETF Option |
|---|---|
| Multiplier (multiplicador) | 100 shares |
| Exercise style | American (exercício a qualquer dia útil) |
| Settlement | Physical delivery of shares (T+1) |
| Premium quotation | Points; 1 pt = $100 |
| Clearing | OCC |
| Regulator | SEC |
| Common expirations | Monthlys (3rd Friday), Weeklys, Quarterlys, LEAPS, FLEX |

Source: OCC equity options product specs and the June 2024 disclosure document ([theocc.com — equity options product specifications](https://www.theocc.com/clearance-and-settlement/clearing/equity-options-product-specifications); [Options Disclosure Document landing page](https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document); [riskstoc.pdf — June 2024 ODD](https://www.theocc.com/getcontentasset/a151a9ae-d784-4a15-bdeb-23a029f50b70/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/riskstoc.pdf)). Assignment mechanics: [FINRA — Understanding Assignment](https://www.finra.org/investors/insights/trading-options-understanding-assignment).

---

## 2. Index Options: SPX & XSP (Opções de Índice)

Index options are **cash-settled** and (for the SPX complex) **European-style** — no early exercise, no share delivery, no pin/assignment risk on the underlying. This makes them the preferred vehicle for systematic vol traders.

| Spec | SPX (S&P 500) | XSP (Mini-SPX) |
|---|---|---|
| Underlying | S&P 500 Index | 1/10 of S&P 500 Index |
| Multiplier | $100 / index point | $100 / index point |
| Notional (~6,000 index) | ~$600,000 | ~$60,000 |
| Exercise style | European | European |
| Settlement | Cash | Cash |
| Settlement timing | Standard 3rd-Friday = **AM-settled** (opening prices, ticker **SET**); SPXW Weeklys/EOM = **PM-settled** (closing prices) | **PM-settled** (closing prices) |
| Tax treatment (US) | §1256 (60/40) | §1256 (60/40) |

SPX trades **9:30–16:15 ET** (regular trading hours) plus a **curb session 16:15–17:00 ET** and **Global Trading Hours (GTH) 20:15–09:25 ET** (8:15 p.m.–9:25 a.m. ET), giving near-24-hour access for hedging around overseas news. Standard SPX (3rd Friday) ceases trading the preceding business day and settles to the **opening** print (settlement ticker **SET**); weekly **SPXW** settles to the **closing** print.

Sources: [Cboe SPX specifications](https://www.cboe.com/tradable_products/sp_500/spx_options/specifications/), [Cboe SPX product page](https://www.cboe.com/tradable-products/sp-500/spx-options/), [SPX fact sheet (PDF)](https://cdn.cboe.com/resources/spx/spx-fact-sheet.pdf), [Cboe XSP options](https://www.cboe.com/tradable-products/sp-500/xsp-options/), [XSP fact sheet (PDF)](https://cdn.cboe.com/resources/xsp/XSP_Options_Fact_Sheet.pdf), [Cboe Global Trading Hours](https://www.cboe.com/insights/posts/cboe-global-trading-hours/). Index-option clearing: [OCC — Index Options](https://www.theocc.com/clearance-and-settlement/clearing/index-options).

---

## 3. Exchanges, Clearing & the OPRA Feed

Listed equity options trade across **18 SEC-registered options exchanges**, with **Cboe** dominant in index products. All of them clear through the single guarantor, **OCC**, which novates every trade and acts as central counterparty. Last-sale and quote data from every venue is consolidated by the **Options Price Reporting Authority (OPRA)** — the options SIP — which disseminates the consolidated NBBO, last sale, volume, and open interest.

OPRA participants include **Cboe Options, Cboe BZX, Cboe C2, Cboe EDGX, BOX, MIAX / MIAX Emerald / MIAX Pearl / MIAX Sapphire, Nasdaq ISE / GEMX / MRX / PHLX / BX / Nasdaq, MEMX, NYSE American, and NYSE Arca**. OPRA is one of the highest-message-rate market-data feeds in the world (peaks in the tens-of-millions of messages/second — e.g., Q3 2024 peaks near 44.8M msg/s, with microburst spikes higher), which is why most quants consume a vendor-normalized version rather than the raw feed.

Sources: [OPRA plan](https://www.opraplan.com/), [Databento — What is OPRA](https://databento.com/microstructure/opra), [Wikipedia — OPRA](https://en.wikipedia.org/wiki/Options_Price_Reporting_Authority).

---

## 4. The Greeks (As Gregas)

Risk sensitivities of an option's price, the core of any hedging or market-making book:

| Greek | Measures sensitivity to | Note |
|---|---|---|
| Delta (Δ) | Underlying price | ~probability ITM; hedge ratio |
| Gamma (Γ) | Change in delta | Largest near ATM / near expiry — drives 0DTE dynamics |
| Theta (Θ) | Time decay | Negative for long options |
| Vega (ν) | Implied volatility | Largest for ATM, longer-dated |
| Rho (ρ) | Interest rates | Matters most for LEAPS |

Black–Scholes–Merton gives closed-form European Greeks; American equity options require binomial/PDE methods. Reference: Hull, *Options, Futures, and Other Derivatives*; OCC disclosure document above.

---

## 5. Implied Volatility & the VIX (Volatilidade Implícita)

**Implied volatility (IV)** is the volatility that, plugged into a pricing model, reproduces the market premium. Across strikes and maturities it forms the **volatility surface** (smile/skew). The **Cboe Volatility Index (VIX)** distills the 30-day expected volatility of the S&P 500 from a strip of out-of-the-money **SPX** option prices using a model-free, variance-swap-style replication.

History: Cboe introduced the **original VIX in 1993**, measuring 30-day implied vol from **at-the-money S&P 100 (OEX)** option prices. The **current model-free methodology** — based on a wide strip of OTM **SPX** options (developed with Goldman Sachs) — was adopted in **2003**; the original 1993 series now lives on as the VXO. The VIX is **mean-reverting** and inversely correlated with the S&P 500, hence the "fear gauge" nickname.

The volatility complex is itself tradable:

| Product | Description |
|---|---|
| VIX Index | 30-day SPX implied vol (spot, not directly tradable) |
| VIX Futures | Launched 2004 on Cboe Futures Exchange (CFE) |
| Mini VIX Futures (VXM) | Launched 2020; 1/10 size of standard VIX future ($100 multiplier) |
| VIX Options | Listed options on the VIX |
| DVOL | Deribit's VIX-style 30-day crypto (BTC/ETH) vol index |

Sources: [Cboe VIX product page](https://www.cboe.com/tradable-products/vix/), [VIX Index methodology (PDF)](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Cboe_Volatility_Index.pdf), [Cboe VIX futures](https://www.cboe.com/tradable-products/vix/vix-futures/).

---

## 6. Weeklys & the 0DTE Phenomenon (Opções de Zero Dias)

After Cboe expanded SPX weeklys to expire **every trading day** (Tuesday added 2022-04-18 and Thursday added 2022-05-11, completing the M–F cycle alongside the existing Mon/Wed/Fri), **zero-days-to-expiration (0DTE)** options exploded. For full-year 2025, 0DTE SPX volume hit a record **~2.3 million contracts/day ≈ 59% of total SPX volume**, and 0DTE reached **~24.1% of all US listed options volume** (up from 21.5% in 2024). 0DTE concentrates enormous **gamma** near the money intraday, creating dealer-hedging flows that can amplify or dampen index moves.

Sources: [Cboe — Evaluating the market impact of SPX 0DTE](https://www.cboe.com/insights/posts/volatility-insights-evaluating-the-market-impact-of-spx-0-dte-options/), [Traders Magazine — VOL Report 2025](https://www.tradersmagazine.com/vol-report/vol-report-0dte-flex-options-are-2025-heroes/), [Cboe — SPX options 74% market share](https://www.cboe.com/insights/posts/spx-options-jump-to-record-74-market-share/), [Cboe — Add Tuesday/Thursday SPX Weeklys (2022)](https://www.prnewswire.com/news-releases/cboe-to-add-tuesday-and-thursday-expirations-for-spx-weeklys-options-301524687.html).

---

## 7. Core Option Strategies (Estratégias)

| Strategy | Construction | View / payoff |
|---|---|---|
| Covered call (lançamento coberto) | Long 100 shares + short call | Income; caps upside |
| Cash-secured put | Short put + cash collateral | Acquire stock / earn premium |
| Vertical spread (trava) | Long+short same-type, different strikes | Defined-risk directional |
| Straddle | Long call + long put, same strike | Long volatility (event play) |
| Strangle | OTM call + OTM put | Cheaper long-vol |
| Iron condor | Short OTM put spread + short OTM call spread | Defined-risk short-vol, range-bound |
| Calendar / diagonal | Different expirations | Term-structure / theta play |

These are the standard building blocks taught in OCC/Cboe education and most options curricula; payoffs follow directly from the multiplier mechanics in §1–2.

---

## 8. Futures: CME E-mini, Micros, Commodities & Treasuries

CME Group equity-index, energy, metals, and rates futures are the backbone of macro and systematic trading. Equity-index contracts are **cash-settled**, listed quarterly (Mar/Jun/Sep/Dec), with a tick of **0.25 index points** and near-24/5 trading (one-hour daily maintenance break, ~16:00–17:00 CT).

| Contract | Code | Multiplier | Tick | Tick value | Settlement |
|---|---|---|---|---|---|
| E-mini S&P 500 | ES | $50 × index | 0.25 | $12.50 | Cash |
| Micro E-mini S&P 500 | MES | $5 × index | 0.25 | $1.25 | Cash |
| E-mini Nasdaq-100 | NQ | $20 × index | 0.25 | $5.00 | Cash |
| Micro E-mini Nasdaq-100 | MNQ | $2 × index | 0.25 | $0.50 | Cash |
| WTI Crude Oil | CL | 1,000 bbl | $0.01/bbl | $10.00 | Physical (Cushing, OK) |
| Gold | GC | 100 troy oz | $0.10/oz | $10.00 | Physical |
| 10-Year T-Note | ZN | $100,000 face | ½ of 1/32 | ~$15.625 | Physical delivery |
| 30-Year T-Bond | ZB | $100,000 face | 1/32 | $31.25 | Physical delivery |

Micros are exactly **1/10** their E-mini parents, lowering the capital and risk granularity for retail and for fine-tuning hedges. Treasury futures deliver against a basket of eligible securities via **conversion factors** (the cheapest-to-deliver / CTD mechanism); the invoice = settlement price × conversion factor + accrued interest. (The 30-Year T-Bond delivery basket is bonds with 15–25 years to maturity; conversion factors are computed to a 6% notional yield.)

Sources: [CME Micro E-mini S&P 500 specs](https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.contractSpecs.html), [CME E-mini S&P 500 overview](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html), [CME E-mini Nasdaq-100 specs](https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html), [CME Gold specs](https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html), [CME WTI Crude specs](https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html), [CME 10-Year T-Note specs](https://www.cmegroup.com/markets/interest-rates/us-treasury/10-year-us-treasury-note.contractSpecs.html), [CME 30-Year Bond specs](https://www.cmegroup.com/markets/interest-rates/us-treasury/30-year-us-treasury-bond.contractSpecs.html), [CME — Understand Treasuries contract specs](https://www.cmegroup.com/education/courses/introduction-to-treasuries/understand-treasuries-contract-specifications).

---

## 9. Machine Learning & Quant Angle

Options are a natural ML frontier because the payoff is nonlinear, the state space (surface + flows) is high-dimensional, and frictions break textbook replication.

- **Deep hedging** — Buehler, Gonon, Teichmann & Wood (2019) reframe hedging as a deep-RL problem: a neural policy learns to replicate a derivative under transaction costs, liquidity limits, and risk preferences, *outperforming Black–Scholes delta when realistic spreads are included*. Extensions feed the **implied-vol surface** as forward-looking state (RNN-FNN / LSTM architectures) and beat practitioner/smile-implied delta hedges in backtests. ([Deep Hedging, arXiv:1802.03042](https://arxiv.org/abs/1802.03042); [IV-surface deep hedging, arXiv:2504.06208](https://arxiv.org/html/2504.06208v2); [arXiv:2407.21138](https://arxiv.org/html/2407.21138v1)).
- **Vol-surface modeling & generation** — neural SDEs, autoencoders, and GANs learn arbitrage-free surface dynamics for scenario simulation and risk; Wiese/Bai/Wood/Buehler simulate equity-option markets ([SSRN 3470756](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3470756); [risk-neutral IV dynamics, arXiv:2103.11948](https://arxiv.org/pdf/2103.11948)).
- **Pricing/calibration** — supervised nets approximate expensive pricers (American/exotic) for real-time calibration; static-hedge nets replicate contingent claims ([arXiv:1911.11362](https://arxiv.org/pdf/1911.11362)).
- **Microstructure & 0DTE** — order-flow / dealer-gamma features feed intraday forecasting given the gamma concentration documented in §6.

---

## 10. Data & APIs (Dados e APIs)

| Source | Coverage | Access |
|---|---|---|
| OPRA SIP (via vendors) | Consolidated US options NBBO/last sale | Databento, dxFeed, ICE, LSEG, FactSet |
| Cboe DataShop / LiveVol | SPX/XSP + equity options, EOD & history | Commercial subscription |
| ORATS | All US equity/ETF/index options + Greeks, history to 2007 | REST Data API ([orats.com/data-api](https://orats.com/data-api)) |
| Polygon.io | Real-time options snapshot: quotes, IV, OI, Greeks | REST/WebSocket ([polygon.io](https://polygon.io/)) |
| Databento | OPRA + CME normalized, historical & live | API ([databento.com](https://databento.com/)) |
| yfinance | Current Yahoo option chains (unofficial; no expired history) | `pip install yfinance` |
| Deribit | BTC/ETH/SOL crypto options & futures (~85% BTC/ETH options share), DVOL — a Coinbase company since Aug 2025 | JSON-RPC / WebSocket / FIX ([docs.deribit.com](https://docs.deribit.com/)) |
| CME (DataMine / via vendors) | Futures & options-on-futures | Subscription |

```python
# yfinance: quick US equity option chain (unofficial)
import yfinance as yf
spy = yf.Ticker("SPY")
exp = spy.options[0]                 # nearest expiration
chain = spy.option_chain(exp)
print(chain.calls[["strike","lastPrice","impliedVolatility","openInterest"]].head())
```

Crypto note: Deribit's **DVOL** is constructed VIX-style (30-day model-free expected vol) for BTC/ETH, enabling the same surface/vol-of-vol modeling as the SPX complex. Sources: [Deribit API docs](https://docs.deribit.com/), [ORATS](https://orats.com/data-api).

---

## Sources

- OCC: [Equity options specs](https://www.theocc.com/clearance-and-settlement/clearing/equity-options-product-specifications) · [Index options](https://www.theocc.com/clearance-and-settlement/clearing/index-options) · [Options Disclosure Document](https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document) · [Standardized Options disclosure / riskstoc (PDF)](https://www.theocc.com/getcontentasset/a151a9ae-d784-4a15-bdeb-23a029f50b70/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/riskstoc.pdf)
- Cboe: [SPX specs](https://www.cboe.com/tradable_products/sp_500/spx_options/specifications/) · [SPX product page](https://www.cboe.com/tradable-products/sp-500/spx-options/) · [SPX fact sheet](https://cdn.cboe.com/resources/spx/spx-fact-sheet.pdf) · [XSP](https://www.cboe.com/tradable-products/sp-500/xsp-options/) · [VIX](https://www.cboe.com/tradable-products/vix/) · [VIX methodology (PDF)](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Cboe_Volatility_Index.pdf) · [Global Trading Hours](https://www.cboe.com/insights/posts/cboe-global-trading-hours/) · [0DTE market impact](https://www.cboe.com/insights/posts/volatility-insights-evaluating-the-market-impact-of-spx-0-dte-options/)
- CME Group: [MES](https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.contractSpecs.html) · [ES](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html) · [NQ](https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html) · [Gold](https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html) · [Crude](https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html) · [10Y T-Note](https://www.cmegroup.com/markets/interest-rates/us-treasury/10-year-us-treasury-note.contractSpecs.html) · [30Y T-Bond](https://www.cmegroup.com/markets/interest-rates/us-treasury/30-year-us-treasury-bond.contractSpecs.html)
- OPRA / data: [OPRA plan](https://www.opraplan.com/) · [Databento OPRA](https://databento.com/microstructure/opra) · [ORATS](https://orats.com/data-api) · [Deribit API](https://docs.deribit.com/)
- FINRA: [Options assignment](https://www.finra.org/investors/insights/trading-options-understanding-assignment)
- ML papers: [Deep Hedging (1802.03042)](https://arxiv.org/abs/1802.03042) · [IV-surface deep hedging (2504.06208)](https://arxiv.org/html/2504.06208v2) · [Simulate option markets (SSRN 3470756)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3470756)

**Keywords:** US options, derivativos, opções americanas, SPX, XSP, índice S&P 500, Cboe, OCC, OPRA, VIX, volatilidade implícita (implied volatility), Greeks / as gregas, delta gamma theta vega, 0DTE, weeklys, covered call (lançamento coberto), iron condor, straddle, trava, CME, E-mini ES MES, Nasdaq NQ MNQ, crude oil, gold, ouro, Treasury futures, deep hedging, vol surface, superfície de volatilidade, Deribit, DVOL, ORATS, Polygon, yfinance, quant, machine learning finanças.
