# India Stock Market — NSE & BSE

> India runs the world's most active equity-derivatives market and one of its fastest-growing retail cash markets: two exchanges (NSE, BSE), a single regulator (SEBI), T+1 settlement (T+0 optional since 2024), and a 2024-25 wave of F&O reforms aimed at a retail-options boom in which ~91% of individual traders lose money. This page is a dense, sourced reference for quants and ML engineers — with a Brazil-access note (acesso via BDR/ETF).

India's cash and derivatives markets are dominated by two exchanges. The **National Stock Exchange (NSE)** is the volume leader and, by number of contracts traded, has repeatedly ranked as the **world's largest derivatives exchange** (Futures Industry Association rankings). The **Bombay Stock Exchange (BSE)**, founded **9 July 1875**, is **Asia's oldest stock exchange**. Both are regulated by the **Securities and Exchange Board of India (SEBI)**. Trading, clearing, settlement, lot sizes, and price bands below are confirmed against exchange/SEBI primary sources (2025-2026).

---

## 1. Exchanges & regulator

| Entity | Role | Founded | Notes |
|---|---|---|---|
| **NSE** (National Stock Exchange of India) | Largest equities + derivatives exchange | 1992 (trading 1994) | Flagship index Nifty 50; world's largest derivatives exchange by contract count. [nseindia.com](https://www.nseindia.com/) |
| **BSE** (Bombay Stock Exchange) | Equities + derivatives; Asia's oldest | **1875** | Flagship index Sensex (30); Asia's oldest exchange. [bseindia.com](https://www.bseindia.com/) |
| **NSE Clearing Ltd (NCL)** | CCP / clearing-settlement for NSE | — | Runs T+1 and optional T+0 cycles. [nseclearing.in](https://www.nseclearing.in/) |
| **Indian Clearing Corp (ICCL)** | CCP for BSE | — | — |
| **SEBI** | Statutory market regulator | 1992 | Sets F&O rules, expiry calendar, STT framework. [sebi.gov.in](https://www.sebi.gov.in/) |
| **NSE Indices Ltd** | Index provider (Nifty family) | — | Owns/maintains Nifty 50, Bank Nifty, etc. [niftyindices.com](https://www.niftyindices.com/) |
| **BSE Index Services / S&P DJI** | Sensex methodology (joint) | — | Sensex maintained jointly with S&P Dow Jones Indices. |

Depositories: **NSDL** and **CDSL** hold securities in dematerialised (demat) form. Stockbrokers (Zerodha, Groww, Angel One, Upstox, ICICI Direct, etc.) provide demat + trading accounts.

---

## 2. Benchmark indices

| Index | Exchange / provider | Constituents | Method | Base | Notes |
|---|---|---|---|---|---|
| **Nifty 50** | NSE Indices | 50 | Free-float market-cap (since 26 Jun 2009) | 3 Nov 1995 = 1000 | Rebalanced semi-annually (cut-offs 31 Jan / 31 Jul). [factsheet PDF](https://www.niftyindices.com/Factsheet/ind_nifty50.pdf) |
| **Nifty Next 50** | NSE Indices | 50 | Free-float market-cap | — | Stocks 51-100 by size; "emerging blue chips". [methodology PDF](https://nsearchives.nseindia.com/content/indices/ind_next50.pdf) |
| **Nifty Bank (Bank Nifty)** | NSE Indices | 12 | Free-float market-cap | — | Most-traded sectoral derivatives underlying. |
| **Nifty Financial Services (FinNifty)** | NSE Indices | ~20 | Free-float market-cap | — | Financials (banks, NBFCs, insurance). |
| **Nifty Midcap Select (MIDCPNIFTY)** | NSE Indices | 25 | Free-float market-cap | — | Midcap derivatives underlying. |
| **BSE Sensex** | BSE / S&P DJI | 30 | Free-float market-cap | 1978-79 = 100 (1 Apr 1979) | India's oldest index; large-cap blue chips. [BSE Indices methodology](https://www.bseindices.com/Downloads/BSE_Indices_Methodology.pdf) |
| **BSE Bankex / BSE Sensex 50** | BSE | 10+ / 50 | Free-float | — | Weekly F&O on these discontinued in the 2024 rationalisation. |

Nifty 50 eligibility (key gates): constituent must be **F&O-eligible**, have **≥1-month listing history**, and meet an **impact-cost ≤0.50%** liquidity test for 90% of observations on a ₹100 million basket. Nifty 50 covers roughly **~54%** of NSE free-float market cap. (Source: [Nifty 50 methodology / factsheet](https://nsearchives.nseindia.com/content/indices/Method_NIFTY_Equity_Indices.pdf))

---

## 3. Market mechanics

### Trading hours (IST)
- **Pre-open auction:** 09:00-09:15
- **Continuous (normal) market:** **09:15-15:30**
- **Closing session:** 15:40-16:00
- Currency/commodity segments differ. (Source: [NSE](https://www.nseindia.com/))

### Settlement cycles
| Cycle | Status | Detail |
|---|---|---|
| **T+1** | **Default** for equity cash (full rollout by Jan 2023) | Trades on T settle T+1; final obligations by ~08:30 on T+1. [NSE settlement cycle](https://www.nseindia.com/static/products-services/equity-market-settlement-cycle) |
| **T+0 (same-day)** | **Optional / beta** since **28 Mar 2024** | Trading window to ~13:30; obligations by ~14:30; payout same day. Being expanded scrip-by-scrip. [NSE T+0 page](https://www.nseindia.com/static/products-services/t0-settlement-cycle) |

India is a global leader on settlement speed: it moved to T+1 ahead of the US/EU and is piloting optional **T+0** and (proposed) instant settlement.

### Price bands & circuit breakers
- **Market-wide index circuit breakers** trigger at **10% / 15% / 20%** moves of Nifty 50 or Sensex (whichever breaches first), halting the whole market for set durations; re-open via a call auction. (Source: [NSE circuit breakers](https://www.nseindia.com/products/content/equities/equities/circuit_breakers.htm))
- **Stock-level daily price bands:** typically **2% / 5% / 10% / 20%** depending on the security; **20%** band for most scrips not in F&O. Stocks in the derivatives segment have **no fixed daily price band** but are subject to dynamic operating ranges / flexible bands. (Source: [NSE price bands](https://www.nseindia.com/static/products-services/equity-market-price-bands))

### Taxes & frictions — **Securities Transaction Tax (STT)**
STT is levied on-exchange and is a major cost driver for high-frequency/options strategies. Rates effective the FY2025-26 / Budget-2026 framework (always re-confirm against the current Finance Act):

| Transaction | STT (on) | Rate |
|---|---|---|
| Delivery equity (buy & sell) | turnover, both sides | **0.10%** each |
| Intraday equity | sell side | **0.025%** |
| Equity **futures** | sell side | **0.02%** (Budget-2026 proposal: rising) |
| Equity **options** | premium, sell side | **0.10%** (post-2024 hike) |
| Options on exercise | settlement (intrinsic) value | **0.125%** |

(Source: [ClearTax STT guide](https://cleartax.in/s/securities-transaction-tax-stt); [NSE Clearing STT](https://www.nseclearing.in/clearing-settlement/equity-derivatives/securities-transaction-tax)). Other costs: exchange transaction charges, SEBI turnover fee, GST, stamp duty, brokerage. Confirm live rates with broker calculators (e.g. Zerodha).

---

## 4. The retail & demat boom (UPI-era)

- NSE added **~8.4 million (84 lakh) new active demat accounts in FY25** (+~20.5% y/y), total active accounts ~**4.92 crore (49.2M)** by Mar 2025. ([source](https://www.deccanherald.com/business/markets/nse-adds-84-lakh-new-demat-accounts-in-fy25-groww-angel-one-drive-growth-3501638))
- NSE total **investor/trading accounts crossed 24 crore (240M)** by Nov 2025, driven by smaller towns; retail ownership in NSE-listed firms hit ~**18.75%** (Sep 2025), a 22-year high. ([source](https://www.angelone.in/news/market-updates/nse-crosses-24-crore-investor-accounts-milestone-amid-surge-from-small-town-india))
- Digital-first brokers (**Groww**, **Angel One**, **Zerodha**, **Upstox**) drive net additions; onboarding is paperless (Aadhaar e-KYC) and funded via **UPI**.

---

## 5. Equity derivatives (F&O) — the centre of gravity

NSE's index options volume makes India the **largest derivatives market by contract count** worldwide. Retail participation is exceptionally high — SEBI reported **~62% of NSE F&O volume is retail-driven** (Apr 2025). The flip side: SEBI studies found **~91% of individual F&O traders lost money in FY25** (net losses ~**₹1.05 lakh crore**, up 41% y/y), and **93% lost over FY22-FY24** (aggregate >**₹1.8 lakh crore**). ([SEBI press release](https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html))

### Index options & the weekly-expiry / 0DTE phenomenon
Weekly-expiring index options (Nifty, Bank Nifty, FinNifty) drove a massive **expiry-day / 0DTE (zero-days-to-expiry)** retail trade — cheap, far-OTM weeklies bought hours before expiry. This concentration of speculative volume on expiry day is exactly what SEBI's 2024-25 reforms target.

### SEBI 2024-2025 F&O tightening (key, dated rules)
| Reform | Effective | Detail |
|---|---|---|
| **One weekly expiry per exchange** | 20 Nov 2024 | Weekly options allowed on only **one benchmark index per exchange**. NSE keeps **Nifty 50** weekly; BSE keeps **Sensex** weekly. NSE discontinued weeklies on Bank Nifty, FinNifty, Midcap Select, Next 50; BSE dropped Bankex & Sensex-50 weeklies. ([Zerodha Z-Connect](https://zerodha.com/z-connect/business-updates/sebis-new-rules-for-index-derivatives-heres-whats-changing)) |
| **Larger contract value (lot-size hike)** | 21 Nov 2024 | Minimum index-derivative contract value raised to **₹15-20 lakh** (from ₹5-10 lakh). |
| **Extreme Loss Margin (ELM)** | 20 Nov 2024 | Extra **2% ELM** on short option positions on expiry day. |
| **Upfront option premium** | 1 Feb 2025 | Buyers must pay **full premium upfront** (removes intraday leverage on long options). |
| **No calendar-spread benefit on expiry** | 10 Feb 2025 | Margin offset removed for spreads where one leg expires that day. |
| **Intraday position-limit monitoring** | 1 Apr 2025 | Position limits checked **multiple times intraday**, not just EOD. |
| **Uniform expiry days** | 1 Sep 2025 | SEBI circular **SEBI/HO/MRD/MRD-TPD-1/P/CIR/2025/76 (26 May 2025)**: all equity-derivative expiries on **Tuesday (NSE)** or **Thursday (BSE)**; expiry changes need SEBI approval. ([Ventura](https://www.venturasecurities.com/blog/changes-in-expiry-nse-and-bse/)) |

### Revised index lot sizes (from 21 Nov 2024)
| Underlying | Old lot | **New lot** |
|---|---|---|
| **NIFTY 50** | 25 | **75** |
| **BANKNIFTY** | 15 | **30** |
| **FINNIFTY** | 25 | **65** |
| **MIDCPNIFTY** | 50 | **120** |
| **SENSEX (BSE)** | 10 | **20** |

(Source: [Zerodha Z-Connect](https://zerodha.com/z-connect/business-updates/sebis-new-rules-for-index-derivatives-heres-whats-changing).) Existing contracts kept old lots until expiry; quarterly/half-yearly contracts transitioned on staggered dates (late Dec 2024).

### Stock futures & options
Single-stock F&O exists for SEBI-eligible large/liquid stocks (the F&O list is periodically revised on liquidity/turnover criteria). Each stock has its **own lot size** (published by NSE/BSE; revised periodically). All equity-derivative single-stock contracts now expire on the exchange's chosen monthly day (last Tuesday/Thursday). Index & stock options are **European-style, cash-settled** (stock options moved to physical/delivery settlement in 2019).

---

## 6. ETFs

India's ETF market is large and growing (passive flows + EPFO mandates into Nifty/Sensex ETFs). Major issuers and example products:

| Issuer | Example ETFs | Underlying |
|---|---|---|
| **Nippon India (Nippon Life AMC)** | Nippon India ETF Nifty 50 BeES, Bank BeES, Gold BeES | Nifty 50, Bank Nifty, gold |
| **SBI Mutual Fund** | SBI Nifty 50 ETF, SBI Sensex ETF | Nifty 50 / Sensex (large EPFO inflows) |
| **ICICI Prudential** | ICICI Pru Nifty 50 ETF, Nifty Next 50 ETF | Nifty family |
| **UTI / HDFC / Mirae** | Nifty 50 / Sensex / Next 50 ETFs | Broad indices |

ETFs trade on NSE/BSE like shares (T+1), priced near iNAV. Gold ETFs and (recently) silver ETFs are also popular.

---

## 7. ML / quant angle for Indian markets

- **Microstructure & expiry dynamics:** the single-weekly-expiry regime (Nifty=Tue, Sensex=Thu) creates strong, *predictable* intraday seasonality and gamma/pin-risk effects around expiry — fertile ground for options-flow, OI (open-interest) and IV-surface modelling. The 2024-25 reforms structurally shifted volume, so models trained pre-Nov-2024 need re-validation (regime break).
- **Options analytics:** Bank Nifty / Nifty weekly chains are liquid enough for live IV-surface fitting, options-greeks ML, and 0DTE PnL distribution studies. STT and the upfront-premium rule materially change strategy economics — bake transaction costs into backtests.
- **Cross-sectional equity factors:** value/momentum/quality/low-vol factor zoo works on Nifty 500 universe; impact-cost and liquidity filters (the same SEBI uses for index inclusion) matter for small/midcaps.
- **Event/regime models:** RBI policy, Union Budget (Feb 1), monsoon, FPI flows, and index-rebalance (Jan/Jul) are exploitable scheduled events.
- **Data hygiene caveats:** corporate-action / split adjustment, survivorship in the F&O-eligible list, and frequent lot-size revisions are the top backtest landmines. Always reconcile to **bhavcopy** (official EOD).

---

## 8. Data & APIs

### Broker APIs (live + historical, retail-accessible)
| API | Provider | Notes |
|---|---|---|
| **Kite Connect** | Zerodha | REST + WebSocket; orders, portfolio, historical OHLC (up to ~10y intraday); ~₹500/mo per API key; free for personal use (2024 change). [zerodha.com/products/api](https://zerodha.com/products/api/) |
| **Upstox API** | Upstox | REST + WebSocket; historical + live; per-order API promos. [upstox.com/developer](https://upstox.com/developer/api-documentation) |
| **SmartAPI** | Angel One | **Free**; orders, portfolio, historical data, WebSocket; ~10 orders/s. [smartapi.angelbroking.com](https://smartapi.angelbroking.com/) |
| **Fyers API** | Fyers | REST + WebSocket; minute OHLCV history; backtest-friendly. [myapi.fyers.in](https://myapi.fyers.in/) |
| **Samco / 5paisa / ICICI Breeze** | resp. brokers | Additional broker REST APIs. |

### Open-source / scraping libraries (Python)
| Library | What it does |
|---|---|
| **`yfinance`** | Yahoo Finance OHLC. Use **`.NS`** suffix for NSE (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`) and **`.BO`** for BSE (`RELIANCE.BO`). Indices: `^NSEI` (Nifty 50), `^BSESN` (Sensex), `^NSEBANK` (Bank Nifty). |
| **`nsepython` / `nsepy`** | NSE EOD/derivatives/option-chain data scraping. |
| **`jugaad-data`** | NSE/BSE bhavcopy, historical stock & derivatives data. |
| **`bhavcopy` (NSE/BSE)** | Official **EOD CSV** of all traded securities — the canonical reference for prices/volumes/settlement. Downloadable from NSE/BSE archives. |

### Ticker / symbol conventions
- **Native symbols:** NSE uses plain symbols (`RELIANCE`, `HDFCBANK`, `NIFTY`, `BANKNIFTY`); BSE uses numeric **scrip codes** (e.g. Reliance = `500325`).
- **yfinance:** append `.NS` (NSE) or `.BO` (BSE). Indices use `^` prefix (`^NSEI`, `^BSESN`).
- **Derivatives:** instrument tokens via broker APIs; option symbols encode underlying+expiry+strike+CE/PE (e.g. `NIFTY25SEP24800CE`).

---

## 9. Access for foreigners & Brazilians (acesso para brasileiros)

- **FPIs (Foreign Portfolio Investors):** foreigners invest in Indian equities via the SEBI **FPI** route (registration, custodian, PAN); direct retail foreign access to NSE/BSE cash is generally not available without FPI/NRI status.
- **ADRs:** several Indian large-caps trade as US ADRs — **Infosys (INFY)**, **Wipro (WIT)**, **HDFC Bank (HDB)**, **ICICI Bank (IBN)** — giving USD exposure without an Indian account.
- **US-listed ETFs:** **iShares MSCI India ETF (INDA)** (≈170+ holdings; top names HDFC Bank, Reliance, ICICI Bank), plus **INDY**, **SMIN**, **EPI**, etc.
- **Brazil (B3):** Brazilians get India exposure without an offshore account via **BDR de ETF** — e.g. **BNDA39**, a Brazilian Depositary Receipt that mirrors **INDA (iShares MSCI India)**, traded in BRL on **B3** with daily liquidity. ([B3 BDR de ETF](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/brazilian-depositary-receipts-bdrs-de-etf.htm); [BNDA39](https://investidor10.com.br/bdrs/bnda39/)). Brazilian brokers also offer global ETFs/ADRs directly via offshore accounts. *Sempre conferir tributação e custos (IR sobre ganhos no exterior, IOF) antes de investir.*

---

## Sources

- NSE — settlement cycle: https://www.nseindia.com/static/products-services/equity-market-settlement-cycle
- NSE — T+0 settlement: https://www.nseindia.com/static/products-services/t0-settlement-cycle
- NSE — price bands: https://www.nseindia.com/static/products-services/equity-market-price-bands
- NSE — circuit breakers: https://www.nseindia.com/products/content/equities/equities/circuit_breakers.htm
- Nifty 50 factsheet/methodology: https://www.niftyindices.com/Factsheet/ind_nifty50.pdf · https://nsearchives.nseindia.com/content/indices/Method_NIFTY_Equity_Indices.pdf
- Nifty Next 50 methodology: https://nsearchives.nseindia.com/content/indices/ind_next50.pdf
- BSE Indices methodology: https://www.bseindices.com/Downloads/BSE_Indices_Methodology.pdf
- SEBI F&O study (93% losses): https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html
- SEBI F&O reforms (Zerodha Z-Connect): https://zerodha.com/z-connect/business-updates/sebis-new-rules-for-index-derivatives-heres-whats-changing
- Expiry-day rationalisation: https://www.venturasecurities.com/blog/changes-in-expiry-nse-and-bse/
- STT (ClearTax): https://cleartax.in/s/securities-transaction-tax-stt · NSE Clearing: https://www.nseclearing.in/clearing-settlement/equity-derivatives/securities-transaction-tax
- Demat boom: https://www.deccanherald.com/business/markets/nse-adds-84-lakh-new-demat-accounts-in-fy25-groww-angel-one-drive-growth-3501638 · https://www.angelone.in/news/market-updates/nse-crosses-24-crore-investor-accounts-milestone-amid-surge-from-small-town-india
- Kite Connect: https://zerodha.com/products/api/ · Angel One SmartAPI: https://smartapi.angelbroking.com/
- INDA ETF: https://www.ishares.com/us/products/239659/ishares-msci-india-etf
- B3 BDR de ETF: https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/brazilian-depositary-receipts-bdrs-de-etf.htm · BNDA39: https://investidor10.com.br/bdrs/bnda39/

**Keywords:** India stock market, NSE, BSE, Nifty 50, Bank Nifty, FinNifty, Sensex, SEBI, F&O derivatives, index options, weekly expiry, 0DTE, lot size, T+1 settlement, T+0, circuit breaker, price band, STT (securities transaction tax), demat, UPI, Kite Connect, SmartAPI, yfinance .NS .BO, bhavcopy, FPI, ADR, ETF INDA, BDR, BNDA39 · *mercado de ações da Índia, bolsa indiana, derivativos, opções de índice, liquidação, leilão, imposto sobre transações, corretora, ETF, BDR, acesso do investidor brasileiro*
