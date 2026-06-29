# US Stock Market — NYSE & Nasdaq (Bolsa Americana)

> The world's deepest equity market: how US exchanges (NYSE, Nasdaq, Cboe, IEX) are wired under Reg NMS, the indices and instruments that matter, how Brazilians can invest, and the ML/quant toolkit for trading and research.

The United States hosts the largest, most liquid equity market on earth. Listings concentrate on two primary venues — the **New York Stock Exchange (NYSE)** and **Nasdaq** — but actual *trading* is fragmented across ~16 equities exchanges (out of ~24 total registered national securities exchanges) plus dozens of off-exchange venues, all stitched together by the **National Market System (NMS)**. This page is a dense reference for data scientists, quants, and investors (especially Brazilian / brasileiros) who need the mechanics straight.

---

## 1. Exchanges & Venues (Bolsas e plataformas)

Trading is dominated by a few exchange *families* plus a notable independent (IEX). Listing is concentrated; trading is fragmented.

| Operator | Exchanges | Role / niche |
|---|---|---|
| **NYSE Group (ICE)** | NYSE, NYSE American, NYSE Arca, NYSE National, NYSE Texas (renamed from NYSE Chicago, launched 31 Mar 2025) | NYSE uses a floor + Designated Market Maker (DMM) parity model; Arca is fully electronic and leads in ETPs; National uses inverted (taker-maker) pricing. ([nyse.com/trade/equities](https://www.nyse.com/trade/equities), [FINRA: NYSE Chicago → NYSE Texas](https://developer.finra.org/news/nyse-chicago-name-change-nyse-texas)) |
| **Nasdaq** | Nasdaq, Nasdaq BX, Nasdaq PSX | Fully electronic, dealer-market heritage; primary listing venue for most tech. |
| **Cboe Global Markets** | BZX, BYX, EDGA, EDGX | One of the largest US equities operators; its four venues together run roughly the low-to-mid teens of *total* US share and around ~20% of *on-exchange* (lit) volume. ([cboe.com/us/equities](https://www.cboe.com/us/equities/)) |
| **IEX (Investors Exchange)** | IEX | Independent; famous "speed bump" (350µs delay, via a coiled fiber loop) to blunt latency arbitrage; small single-digit share of volume. ([Wikipedia: IEX](https://en.wikipedia.org/wiki/IEX)) |

Roughly **half of US share volume executes off-exchange** (wholesalers, ATSs/dark pools, internalizers) — averaging ~45% in 2024 and frequently at or above 50% in late 2024–2025 — with the rest "on-exchange" (lit). ([SIFMA blog](https://www.sifma.org/news/blog/rethinking-trade-through-prohibitions-beware-of-the-market-structure-octopus))

- **Lit venues** publish quotes pre-trade (displayed order books on exchanges).
- **Dark venues** (dark pools / ATSs) match without displaying quotes — used by institutions to move size without signaling. Reg NMS best-price routing fragmented liquidity and pushed block flow into the dark. ([Wikipedia: Regulation NMS](https://en.wikipedia.org/wiki/Regulation_NMS))

---

## 2. Market Structure under Reg NMS (Estrutura de mercado)

**Regulation NMS** (adopted 2005) is the rulebook that makes the ~16 equities exchanges behave as one market. Core pillars ([SEC / Wikipedia](https://en.wikipedia.org/wiki/Regulation_NMS)):

- **Order Protection Rule (Rule 611 / "trade-through rule")** — orders must execute at the best displayed price across all protected venues; you cannot "trade through" a better quote elsewhere.
- **NBBO (National Best Bid and Offer)** — the consolidated best bid and best offer across all exchanges; the reference price for best execution and price improvement.
- **Access Rule (Rule 610)** — fair, non-discriminatory access to quotes; caps access fees.
- **Sub-Penny Rule (Rule 612)** — minimum quoting increments. Under the **2024 Reg NMS amendments** (adopted 18 Sep 2024), NMS stocks priced ≥ $1.00 with a **Time-Weighted Average Quoted Spread (TWAQS) ≤ $0.015** move to a **half-cent ($0.005) tick**; the **access-fee cap drops from 30 mils ($0.003/share) to 10 mils ($0.001/share)**; and round-lot/odd-lot definitions are accelerated. Compliance: Rule 612/610 + round-lot redefinition on the first business day of **November 2025**; odd-lot information on the first business day of **May 2026**. ([SEC press release 2024-137](https://www.sec.gov/newsroom/press-releases/2024-137), [SEC final rule 34-101070](https://www.sec.gov/files/rules/final/2024/34-101070.pdf), [Davis Polk](https://www.davispolk.com/insights/client-update/reg-nms-resized-sec-adjusts-tick-sizes-lowers-access-fees-and-accelerates))

**Market makers & DMMs.** On NYSE, **exactly one DMM is assigned per listed security** (e.g., Citadel Securities, GTS); the DMM commits capital to dampen volatility and to set fair opening/closing prices. Citadel Securities alone is DMM for ~60%+ of NYSE listings. ([nyse.com DMM PDF](https://www.nyse.com/publicdocs/nyse/markets/nyse/designated_market_makers.pdf), [Citadel Securities](https://www.citadelsecurities.com/what-we-do/equities/designated-market-maker-dmm/)) Nasdaq uses competing electronic market makers rather than a single DMM.

**Payment for Order Flow (PFOF).** Retail brokers (Robinhood, Schwab, etc.) route marketable orders to wholesalers (Citadel Securities, Virtu) for a small rebate (fractions of a cent/share); wholesalers profit from the spread while often delivering **price improvement vs. the NBBO**. Disclosed via **SEC Rule 606** (quarterly routing reports) and **Rule 607** (account-opening disclosure). ([CRS / Congress.gov](https://www.congress.gov/crs-product/IF12594), [Wikipedia: PFOF](https://en.wikipedia.org/wiki/Payment_for_order_flow))

---

## 3. Trading Sessions & Auctions (Pregão e leilões)

| Session | Hours (Eastern Time) | Notes |
|---|---|---|
| **Pre-market (pré-mercado)** | ~04:00–09:30 ET | Thinner liquidity, wider spreads. |
| **Core / regular session (pregão regular)** | **09:30–16:00 ET** | Standard session; half-days close 13:00 ET around certain holidays. ([nyse.com/trade/equities](https://www.nyse.com/trade/equities)) |
| **After-hours (after-market)** | ~16:00–20:00 ET | Earnings reactions land here. |

- **Opening Auction (~09:30 ET)** and **Closing Auction (~16:00 ET)** are the single largest liquidity events of the day; NYSE DMMs facilitate them. The closing auction sets official closing prices used by index funds and benchmarks. ([NYSE Auctions Fact Sheet](https://www.nyse.com/publicdocs/nyse/markets/nyse/NYSE_Opening_and_Closing_Auctions_Fact_Sheet.pdf))
- ET = US Eastern Time. Brazil (BRT, UTC−3) has had no DST since 2019, so the gap to ET is 1h when the US is on EST and 2h when the US is on EDT — meaning the US open is ~11:30 in São Paulo during US daylight time and ~11:30/10:30 otherwise; confirm around US DST changes.

**Settlement: T+1.** Since **May 28, 2024** the standard US settlement cycle is **one business day (T+1)** for stocks, ETFs, and most securities — shortened from T+2 (the SEC adopted the rule on 15 Feb 2023) to cut counterparty, market, and liquidity risk. ([Davis Polk](https://www.davispolk.com/insights/client-update/sec-adopts-t-1-settlement-effective-may-2024), [Investor.gov](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/new-t1-settlement-cycle-what-investors-need-know-investor-bulletin))

---

## 4. Indices & Sectors (Índices e setores)

| Index | Provider | Constituents | Weighting |
|---|---|---|---|
| **S&P 500** | S&P Dow Jones Indices | 500 large-cap US firms | **Float-adjusted market cap (FMC)** — only investor-available shares counted. ([SPDJI US Indices Methodology](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-indices.pdf)) |
| **Dow Jones Industrial Average (DJIA)** | S&P Dow Jones Indices | 30 blue chips | **Price-weighted** (divisor-scaled) — high-priced stocks dominate. |
| **Nasdaq-100 (NDX)** | Nasdaq | 100 largest non-financial Nasdaq firms | **Modified market-cap**; capping prevents over-concentration: no single issuer >20%, and if the combined weight of issuers individually above 4.5% reaches ≥48% it is trimmed to 40%. A May 2026 methodology update added quarterly rank-based reviews (Mar/Jun/Sep) and a 3×-float weight cap. ([Nasdaq NDX Methodology PDF](https://indexes.nasdaq.com/docs/Methodology_NDX.pdf), [May 2026 NDX Changes FAQ](https://indexes.nasdaqomx.com/docs/2026_May_NDX_Changes_FAQ.pdf)) |
| **Nasdaq Composite** | Nasdaq | ~all Nasdaq-listed common stocks | Market-cap weighted. |
| **Russell 3000 / 1000 / 2000** | FTSE Russell (LSEG) | ~broad investable US market / large / small cap | **Float-adjusted market-cap**; reconstitution moves to **semi-annual (June & December) starting 2026** (June on the fourth Friday, December on the second Friday) to reduce single-event turnover. ([LSEG: semi-annual move](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2025/russell-us-indexes-move-to-semi-annual-reconstitution), [LSEG Russell Reconstitution](https://www.lseg.com/en/ftse-russell/russell-reconstitution)) |

**GICS sectors** — the standard sector taxonomy (MSCI + S&P), **11 sectors**: Communication Services, Consumer Discretionary, Consumer Staples, Energy, Financials, Health Care, Industrials, Information Technology, Materials, Real Estate, Utilities. Below sit 25 industry groups, ~74 industries, ~163 sub-industries. ([MSCI GICS](https://www.msci.com/indexes/index-resources/gics), [SPDJI GICS](https://www.spglobal.com/spdji/en/landing/topic/gics/))

---

## 5. Share Classes, ADRs & Instruments

- **Share classes (classes de ações).** Many firms list multiple classes with different voting rights — e.g., Alphabet **GOOGL** (Class A, one vote/share) vs **GOOG** (Class C, no vote); the super-voting Class B (10 votes/share) is held by insiders and not publicly traded. Berkshire Hathaway lists **BRK.A** vs **BRK.B**. Index providers (Nasdaq-100, Russell) increasingly aggregate economic size across classes while weighting only *listed* shares. ([Nasdaq NDX Methodology](https://indexes.nasdaq.com/docs/Methodology_NDX.pdf))
- **ADRs (American Depositary Receipts).** US-traded certificates representing foreign shares (e.g., Brazil's Petrobras **PBR**, Vale **VALE**, Nu Holdings **NU**, Itaú **ITUB**). Sponsored Levels I–III mirror the depositary-receipt structure used by BDRs in reverse.
- **ETFs** track the indices above: **SPY/IVV/VOO** (S&P 500), **QQQ** (Nasdaq-100), **DIA** (DJIA), **IWM** (Russell 2000). These are the most liquid instruments for index exposure.

---

## 6. How Brazilians Invest in US Stocks (Como brasileiros investem)

Three practical routes:

**A) BDRs on B3 (Brazilian Depositary Receipts).** Certificates on B3 representing foreign shares — exposure in BRL without opening an offshore account. Program levels ([B3 official](https://www.b3.com.br/en_us/products-and-services/solutions-for-issuers/bdrs-brazilian-depositary-receipts/)):

| Level | Sponsored? | Public offering | Investor access | Reporting |
|---|---|---|---|---|
| Unsponsored I | No | No | Qualified/institutional (historically); CVM rules later opened to retail | Origin-country GAAP |
| Sponsored I | Yes | No | Qualified/institutional | Origin-country GAAP |
| Sponsored II | Yes | No | All investors | IFRS |
| Sponsored III | Yes | Yes | All investors | IFRS |

**B) US-domiciled brokerage accounts.** Brazilian-facing brokers (Avenue, Nomad, BTG, XP International) or direct US brokers (**Interactive Brokers, Charles Schwab**). A **Form W-8BEN** certifies non-US (NRA) status, avoids 24% backup withholding, and is required to hold US securities. **Important:** Brazil and the US have **no income-tax treaty**, so a Brazilian resident **cannot** use the W-8BEN to obtain a reduced treaty rate — US-source dividends are withheld at the **default 30%** rate. (Many other countries enjoy a 15% treaty rate; Brazil does not.) US capital gains are generally not taxed to NRAs at source. ([IRS W-8BEN instructions](https://www.irs.gov/instructions/iw8ben), [PwC: Brazil foreign tax relief & treaties](https://taxsummaries.pwc.com/brazil/individual/foreign-tax-relief-and-tax-treaties))

**C) International ETFs/funds** wrapped locally.

> ⚠️ **US estate tax trap (imposto sucessório).** US-listed stocks are **US-situs assets**. A non-resident alien's US-situs holdings above only **US$60,000** can be exposed to **US estate tax that ramps up to 40%** at death (the rate schedule begins at 18%). Brazil has **no estate-tax treaty** with the US, so only the small $60k filing threshold applies — a key reason some prefer Ireland-domiciled UCITS ETFs or BDRs. ([IRS: nonresidents with US assets](https://www.irs.gov/individuals/international-taxpayers/some-nonresidents-with-us-assets-must-file-estate-tax-returns)) *Not tax advice — consult a cross-border specialist.*

---

## 7. ML / Quant Angle (Aprendizado de máquina aplicado)

US equities are the canonical testbed for quantitative finance because of clean, deep, decades-long data (CRSP) and liquid instruments.

- **Empirical asset pricing via ML.** Gu, Kelly & Xiu (2020, *Review of Financial Studies* 33(5):2223–2273) benchmark linear models, trees, random forests, gradient boosting, and neural nets to predict the cross-section of US stock returns; flexible (nonlinear) models — especially trees and neural nets — dominate, and the most informative predictors are **price trends (momentum), liquidity, and volatility**. ([SSRN 3159577](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577), [Xiu PDF](https://dachxiu.chicagobooth.edu/download/ML.pdf), [Oxford RFS](https://academic.oup.com/rfs/article/33/5/2223/5758276))
- **Deep learning in asset pricing** — autoencoder/IPCA latent-factor models for time-varying betas. ([arXiv 1904.00745](https://arxiv.org/pdf/1904.00745))
- **Common applications:** factor modeling & alpha research; cross-sectional return ranking (learning-to-rank); regime detection (HMM, change-point); NLP on 10-K/10-Q/8-K filings and earnings calls; limit-order-book microstructure prediction (deep LOB models); execution/optimal-routing under T+1 and the NBBO; portfolio construction (mean-variance, hierarchical risk parity).
- **Methodology hygiene:** survivorship bias, look-ahead bias, point-in-time fundamentals, purged/embargoed cross-validation, and transaction-cost/slippage realism are the difference between a backtest and a strategy.

---

## 8. Data & APIs (Dados e APIs)

| Source | Type | Notes / endpoint |
|---|---|---|
| **yfinance** | Free OHLCV/fundamentals | Yahoo Finance wrapper; great for prototyping, no SLA. `pip install yfinance`. |
| **Polygon.io** | Real-time + historical ticks | REST + WebSocket, JSON/CSV; Python client `polygon-api-client`. ([polygon.io](https://polygon.io/), [client-python](https://github.com/polygon-io/client-python)) |
| **Alpaca** | Data **+ commission-free trading** | Place real US-stock orders via API plus market data. |
| **IEX / Cboe data** | Exchange feeds | IEX historical TOPS/DEEP; Cboe One consolidated feed. ([Cboe One](https://www.cboe.com/market_data_services/us/equities/cboe_one/)) |
| **Nasdaq Data Link** (ex-Quandl) | Curated datasets | Free + premium; official Python package. |
| **SEC EDGAR** | Filings & XBRL fundamentals | Official `data.sec.gov` REST APIs: **submissions**, **companyfacts**, **companyconcept**; **no API key**, but a descriptive **User-Agent** header (name + email) is required and the rate limit is **10 requests/second per IP** (across data.sec.gov, efts.sec.gov, www.sec.gov). Full-text search covers filings since 2001. ([SEC Developer Resources](https://www.sec.gov/about/developer-resources), [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [EFTS FAQ](https://www.sec.gov/edgar/search/efts-faq.html)) |
| **CRSP** (Wharton WRDS) | Academic gold standard | Survivorship-bias-free daily/monthly US stock returns since 1925; standard for asset-pricing research. |
| **Compustat** (S&P) | Fundamentals | Standardized financial statements; CRSP/Compustat merge is the canonical research panel. |

Example EDGAR fundamentals pull:
```
GET https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json
Headers: User-Agent: "Your Name your.email@example.com"
```

---

## 9. Quick Reference (Resumo)

- **Primary listing venues:** NYSE, Nasdaq. **Trading:** ~16 equities exchanges + roughly half of volume off-exchange.
- **Glue:** Reg NMS → NBBO + Order Protection Rule (no trade-throughs).
- **Core hours:** 09:30–16:00 ET; pre/after-market extend ~04:00–20:00 ET.
- **Settlement:** T+1 (since 28-May-2024).
- **2024 Reg NMS amendments:** half-cent ($0.005) tick for stocks with TWAQS ≤ $0.015; access-fee cap cut to $0.001/share; phased in Nov 2025 / May 2026.
- **Benchmarks:** S&P 500 (FMC), DJIA (price-weighted), Nasdaq-100 (modified cap, 20% single-issuer cap), Russell 2000/3000 (float cap, semi-annual reconstitution from 2026).
- **Sectors:** 11 GICS sectors.
- **Brazil access:** BDRs (B3) | US brokers + W-8BEN (NRA status only — **30% dividend withholding, no 15% treaty rate**) | watch the **$60k US-situs estate-tax** exposure.

---

**Sources:** [SEC Reg NMS (Wikipedia)](https://en.wikipedia.org/wiki/Regulation_NMS) · [NYSE Equities](https://www.nyse.com/trade/equities) · [NYSE DMM PDF](https://www.nyse.com/publicdocs/nyse/markets/nyse/designated_market_makers.pdf) · [NYSE Auctions Fact Sheet](https://www.nyse.com/publicdocs/nyse/markets/nyse/NYSE_Opening_and_Closing_Auctions_Fact_Sheet.pdf) · [NYSE Texas / NYSE Chicago (FINRA)](https://developer.finra.org/news/nyse-chicago-name-change-nyse-texas) · [Cboe US Equities](https://www.cboe.com/us/equities/) · [SIFMA market structure](https://www.sifma.org/news/blog/rethinking-trade-through-prohibitions-beware-of-the-market-structure-octopus) · [Reg NMS 2024 amendments (SEC PR 2024-137)](https://www.sec.gov/newsroom/press-releases/2024-137) · [Reg NMS final rule (34-101070)](https://www.sec.gov/files/rules/final/2024/34-101070.pdf) · [Reg NMS amendments (Davis Polk)](https://www.davispolk.com/insights/client-update/reg-nms-resized-sec-adjusts-tick-sizes-lowers-access-fees-and-accelerates) · [PFOF (CRS)](https://www.congress.gov/crs-product/IF12594) · [T+1 (Davis Polk)](https://www.davispolk.com/insights/client-update/sec-adopts-t-1-settlement-effective-may-2024) · [T+1 (Investor.gov)](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/new-t1-settlement-cycle-what-investors-need-know-investor-bulletin) · [S&P US Indices Methodology](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-indices.pdf) · [Nasdaq-100 Methodology](https://indexes.nasdaq.com/docs/Methodology_NDX.pdf) · [Nasdaq-100 May 2026 Changes FAQ](https://indexes.nasdaqomx.com/docs/2026_May_NDX_Changes_FAQ.pdf) · [Russell Semi-Annual Reconstitution (LSEG)](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2025/russell-us-indexes-move-to-semi-annual-reconstitution) · [Russell Reconstitution (LSEG)](https://www.lseg.com/en/ftse-russell/russell-reconstitution) · [GICS (MSCI)](https://www.msci.com/indexes/index-resources/gics) · [GICS (S&P)](https://www.spglobal.com/spdji/en/landing/topic/gics/) · [B3 BDRs](https://www.b3.com.br/en_us/products-and-services/solutions-for-issuers/bdrs-brazilian-depositary-receipts/) · [W-8BEN instructions (IRS)](https://www.irs.gov/instructions/iw8ben) · [Brazil foreign tax relief & treaties (PwC)](https://taxsummaries.pwc.com/brazil/individual/foreign-tax-relief-and-tax-treaties) · [IRS NRA estate tax (US-situs assets)](https://www.irs.gov/individuals/international-taxpayers/some-nonresidents-with-us-assets-must-file-estate-tax-returns) · [Gu-Kelly-Xiu (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577) · [Gu-Kelly-Xiu (RFS)](https://academic.oup.com/rfs/article/33/5/2223/5758276) · [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) · [EDGAR Full-Text Search FAQ](https://www.sec.gov/edgar/search/efts-faq.html) · [Polygon.io](https://polygon.io/)

**Keywords:** US stock market, NYSE, Nasdaq, Cboe, IEX, Reg NMS, NBBO, dark pool, payment for order flow, PFOF, DMM, market maker, opening auction, closing auction, T+1 settlement, half-penny tick, access fee cap, S&P 500, Dow Jones, Nasdaq-100, Russell 2000, Russell 3000, GICS, ADR, ETF, share classes, SEC EDGAR, CRSP, Compustat, Polygon, Alpaca, yfinance, machine learning, empirical asset pricing, quant, bolsa americana, ações americanas, BDR, B3, W-8BEN, imposto sucessório, dividendos, retenção 30%, mercado de ações dos EUA, investir nos EUA, aprendizado de máquina, finanças quantitativas
