# Japan & East-Asia Markets — JPX, HKEX, KRX, TWSE

> Reference page on the four developed East-Asian equity markets (ex-mainland China) — Japan, Hong Kong, South Korea, and Taiwan — covering exchanges, indices, regulators, trading mechanics (price limits, settlement, tick rules), and the ML/quant data stack (APIs, libraries, ticker suffixes), with notes on how Brazilian investors (*investidores brasileiros*) reach them via ETFs and BDRs.

This page is part of the AIForge open ML/finance index. It targets quant researchers and ML engineers who need precise, current (2025–2026) market microstructure facts plus working data endpoints. Mainland China (SSE/SZSE, A-shares) is covered separately; here we treat the *developed* East-Asian venues. All non-obvious facts are sourced inline from primary exchange/regulator/index-provider pages.

---

## 1. The four markets at a glance

| Market | Exchange (operator) | Regulator | Main board indices | Settlement | Daily price limit | Currency | yfinance suffix |
|---|---|---|---|---|---|---|---|
| Japan | Tokyo Stock Exchange / Japan Exchange Group (JPX) | FSA (Financial Services Agency) | Nikkei 225, TOPIX | T+2 | none (per-tick "renewal price limits", static) | JPY (¥) | `.T` |
| Hong Kong | Hong Kong Exchanges and Clearing (HKEX) | SFC (Securities and Futures Commission) | Hang Seng Index (HSI), Hang Seng TECH | T+2 | none (no daily limit on equities) | HKD (HK$) | `.HK` |
| South Korea | Korea Exchange (KRX) | FSC / FSS | KOSPI, KOSDAQ | T+2 | ±30% | KRW (₩) | `.KS` (KOSPI) / `.KQ` (KOSDAQ) |
| Taiwan | Taiwan Stock Exchange (TWSE) + Taipei Exchange (TPEx) | FSC (Taiwan) | TAIEX | T+2 | ±10% | TWD (NT$) | `.TW` (TWSE) / `.TWO` (TPEx) |

Note the contrast in volatility controls: Korea and Taiwan use hard daily percentage bands; Japan uses per-order *renewal price limits* (static price-tick caps) rather than a percentage day-band; Hong Kong has no daily price limit at all on ordinary shares.

---

## 2. Japan — JPX / Tokyo Stock Exchange

**Operator:** Japan Exchange Group (JPX), which owns the Tokyo Stock Exchange (TSE) and Osaka Exchange (OSE, derivatives). **Regulator:** Financial Services Agency (FSA, *金融庁*).

### Market segments (2022 restructure)
On **4 April 2022** the TSE replaced its old four-tier structure (First Section, Second Section, Mothers, JASDAQ) with three segments ([JPX – Overview of Market Restructuring](https://www.jpx.co.jp/english/equities/improvements/market-structure/01.html)):

| Segment | Orientation | Initial count (4 Apr 2022) |
|---|---|---|
| **Prime** | Large caps centred on constructive dialogue with global investors; highest governance/liquidity bar | 1,839 |
| **Standard** | Sufficient liquidity & governance to be a public-market investment | 1,466 |
| **Growth** | High growth potential, higher risk | 466 |

Of 3,771 TSE companies at transition, those were the segment selections ([JPX – Results of Market Segment Selection](https://www.jpx.co.jp/english/corporate/news/news-releases/0060/20220111-01.html)). Transitional measures for continued-listing criteria ended in **March 2025**, after which companies must meet the full segment standards or face designation/delisting review.

### Indices
- **Nikkei 225** — *price-weighted* index of 225 highly liquid large caps, maintained by Nikkei Inc.; a company's weight depends on its **share price**, not market cap. Reviewed twice a year ([Nikkei FAQ PDF](https://indexes.nikkei.co.jp/nkave/archives/faq/faq_nikkei_stock_average_en.pdf)).
- **TOPIX** (Tokyo Stock Price Index) — *free-float market-cap weighted*, broad benchmark covering ~95% of Japanese market value, calculated mainly from Prime-market stocks ([JPX – Stock Price Index FAQ](https://www.jpx.co.jp/english/faq/stock_price_index.html)).

### Mechanics
- **Settlement:** **T+2**, live since **16 July 2019** (shortened from T+3) ([JPX – T+2 settlement](https://www.jpx.co.jp/english/equities/clearing-settlement/tplus2-settlement-cycle/index.html)).
- **Trading hours:** morning session 09:00–11:30, afternoon 12:30–15:00 (extended to 15:30 from Nov 2024; verify on JPX). Lunch break between sessions.
- **Price limits:** static **renewal price limits** (daily limit-up/limit-down expressed as fixed yen amounts per base-price band), not a percentage day-band like Korea.
- **Ticker convention:** 4-digit numeric codes (e.g., Toyota `7203`, Sony `6758`). On Yahoo/yfinance append `.T` → `7203.T`.

---

## 3. Hong Kong — HKEX

**Operator:** Hong Kong Exchanges and Clearing (HKEX), running the Stock Exchange of Hong Kong (SEHK). **Regulator:** Securities and Futures Commission (SFC).

### Indices (Hang Seng Indexes Company)
- **Hang Seng Index (HSI)** — flagship blue-chip benchmark, **free-float-adjusted market-cap weighted with an 8% per-constituent cap**; quarterly review/rebalance; constituent count expanded over time (≈82+ as of 2024) ([HSI methodology PDF](https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/methodologies/IM_hsie.pdf)).
- **Hang Seng TECH Index (HSTECH)** — the **30 largest** Hong Kong-listed tech companies, market-cap weighted with an **8% cap**, fixed at 30 constituents, quarterly review ([HSTECH methodology PDF](https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/methodologies/IM_hsteche.pdf)).

### Mechanics
- **Settlement:** **T+2** for SEHK equities ([Clearstream – HK settlement](https://www.clearstream.com/clearstream-en/res-library/market-coverage/settlement-process-hong-kong-1281186)).
- **Board lots vary by stock** — there is no universal 100-share lot; each security defines its own board-lot size (e.g., 100, 200, 500, 1,000, 2,000), so always check the instrument's lot before sizing orders.
- **No daily price limit** on ordinary shares.
- **Stock Connect (southbound):** mainland investors buy eligible HK-listed stocks via the Shanghai/Shenzhen ↔ HK link; eligibility is driven by Hang Seng Composite LargeCap/MidCap constituents and SmallCap names ≥ HK$5bn, plus SEHK-listed shares. Because settlement is T+2, **day-trading of Connect securities is not permitted** ([HKEX – Stock Connect](https://www.hkex.com.hk/Mutual-Market/Stock-Connect?sc_lang=en); [HKEX Stock Connect FAQ PDF](https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Getting-Started/Information-Booklet-and-FAQ/FAQ/FAQ_En.pdf)). Southbound flow is a widely watched sentiment signal.
- **Ticker convention:** numeric codes, typically padded (Tencent `0700`, HSBC `0005`); yfinance uses `0700.HK`.

---

## 4. South Korea — KRX (KOSPI & KOSDAQ)

**Operator:** Korea Exchange (KRX). **Regulators:** Financial Services Commission (FSC) + Financial Supervisory Service (FSS).

### Indices
- **KOSPI** — main-board composite (Korea Composite Stock Price Index), free-float market-cap weighted; **KOSPI 200** is the liquid derivatives/ETF benchmark subset.
- **KOSDAQ** — venture/tech-growth board composite (Korea's analogue to NASDAQ).

### Mechanics ([KRX – Guide to Trading in the Korean Stock Market PDF](https://global.krx.co.kr/contents/GLB/01/0109/0109000000/guide_to_trading_in_the_korean_stock_market.pdf))
- **Daily price limit:** **±30%** of the base (previous close) price, for both KOSPI and KOSDAQ.
- **Settlement:** **T+2**.
- **Regular session:** 09:00–15:30 KST, continuous trading with **no lunch break**; opening single-price call auction 08:30–09:00 and closing call auction 15:20–15:30 set official open/close prices.
- **Tick sizes** (KRW, by price band): <1,000 → 1; 1,000–4,999 → 5; 5,000–9,999 → 10; 10,000–49,999 → 50; 50,000–99,999 → 100; 100,000–499,999 → 500 (KOSPI); ≥500,000 → 1,000 (KOSPI). ETF/ETN/ELW trade on a flat 5-KRW tick.
- **Volatility Interruption (VI):** static/dynamic VI temporarily switches a stock to single-price auction on sharp moves.
- **2026 change:** from **29 June 2026** KRX adds executable pre-market (07:00–08:00) and after-market (16:00–20:00) sessions alongside the regular 09:00–15:30 session — verify the latest rules before building intraday systems.
- **Ticker convention:** 6-digit numeric (Samsung Electronics `005930`); yfinance `005930.KS` (KOSPI) or `.KQ` (KOSDAQ).

---

## 5. Taiwan — TWSE (+ TPEx)

**Operator:** Taiwan Stock Exchange (TWSE), main board; Taipei Exchange (TPEx) runs the OTC/emerging board. **Regulator:** Financial Supervisory Commission (FSC, Taiwan).

### Index
- **TAIEX** (Taiwan Capitalization Weighted Stock Index) — free-float market-cap weighted main-board benchmark; heavily concentrated in semiconductors (TSMC dominant).

### Mechanics ([TWSE – Trading Mechanism](https://www.twse.com.tw/en/products/system/trading.html))
- **Daily price limit:** **±10%** of the opening auction reference price for stocks, ETFs/ETNs, beneficiary certificates, TDRs and convertible bonds (bonds ±5%); **new IPOs have no price limit for the first 5 trading days**.
- **Settlement:** **T+2**.
- **Trading hours:** order entry from 08:30; continuous regular session **09:00–13:30**; opening & closing prices set by call auction; intraday odd-lot trading runs 09:00–13:30 with periodic call auctions (first match 09:10, then every few seconds).
- **Tick sizes** (NT$, by band): 0.01–4.99 → 0.01; 5.00–9.99 → 0.05; 10.00–49.99 → 0.05; 50.00–99.99 → 0.10; rising further at higher price bands.
- **Volatility safeguard:** sharp deviations trigger a brief (~2-minute) matching suspension, then resume via call auction.
- **Ticker convention:** 4-digit numeric (TSMC `2330`); yfinance `2330.TW` (TWSE) or `.TWO` (TPEx).

---

## 6. ML / Quant angle

These markets are attractive — and tricky — for quant/ML work because their microstructure rules directly shape model design:

- **Hard price bands (KRX ±30%, TWSE ±10%)** create *limit-hit* states: returns are truncated, order books thin near the band, and limit-up/limit-down events cluster. Models for next-day returns must treat "locked-limit" days as a censored/absorbing regime — naïve mean-reversion or momentum features mislead here. Classification of *will a name hit its limit?* is itself a tradable signal in Korea/Taiwan.
- **Call-auction open/close (KRX, TWSE)** means the most informative prices are auction-cleared, not continuous; intraday alpha research should model the auction imbalance separately.
- **Japan price-weighted Nikkei 225** creates index-arbitrage and divisor-event effects (stock splits, constituent swaps) that pure market-cap intuition misses; pairs of Nikkei vs TOPIX (price-weight vs cap-weight) is a classic factor-spread study.
- **Southbound Stock Connect flow** (HK) and **foreign-investor net buy/sell** (KRX/TWSE publish daily institutional & foreign flows) are strong, widely used sentiment/flow features for cross-sectional models.
- **Semiconductor concentration** (TWSE/TAIEX ≈ TSMC; KOSPI ≈ Samsung/SK Hynix) means single-name and sector-factor risk dominate — useful for supply-chain/earnings-driven ML, but demands careful neutralization in factor models.
- **Common toolchain:** `pandas`/`polars` for data, `statsmodels`/`scikit-learn` for cross-sectional factor models, gradient-boosting (`lightgbm`/`xgboost`) for return ranking, and `backtrader`/`vectorbt`/`zipline`-style backtesters. Always align timezones (JST, HKT, KST, Taipei) and exchange holiday calendars per market.

---

## 7. Data & APIs

| Source | Market | Type | Endpoint / library | Notes |
|---|---|---|---|---|
| **J-Quants API** (official, JPX) | Japan | Prices, financials, listed info, calendar | `https://api.jquants.com/v1` ; docs at [jpx.co.jp J-Quants](https://www.jpx.co.jp/english/markets/other-data-services/j-quants-api/index.html) | Free tier has **12-week delay**; paid plans (Light/Standard/Premium) reduce delay. Endpoints: `/listed/info`, `/prices/daily_quotes`, `/fins/statements`, `/fins/announcement`, `/markets/trading_calendar`. Register at [jpx-jquants.com](https://jpx-jquants.com/en). |
| **yfinance** (Yahoo Finance) | all four | OHLCV, basics | `pip install yfinance` | Suffixes: `.T` (Tokyo), `.HK` (Hong Kong), `.KS`/`.KQ` (KOSPI/KOSDAQ), `.TW`/`.TWO` (TWSE/TPEx). Indices: `^N225`, `^TPX` (TOPIX via `^TOPX`), `^HSI`, `^KS11` (KOSPI), `^KQ11` (KOSDAQ), `^TWII` (TAIEX). Unofficial; rate-limited. |
| **pykrx** | Korea | KRX/Naver scrape: OHLCV, market cap, fundamentals | [github.com/sharebook-kr/pykrx](https://github.com/sharebook-kr/pykrx) ([PyPI](https://pypi.org/project/pykrx/)) | `get_market_ohlcv`, `get_stock_ticker_list`; add ~1s delay between calls to avoid KRX blocking. |
| **KRX Information Data System** | Korea | Official market data portal | [data.krx.co.kr](https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd?locale=en) | Official statistics, indices, derivatives data. |
| **OpenDART / DART** | Korea | Regulatory filings, financials | [opendart.fss.or.kr](https://engopendart.fss.or.kr/) ; English UI [englishdart.fss.or.kr](https://englishdart.fss.or.kr/) | Free API key; corporate disclosures, BS/PL/CF. Wrappers: `dart-fss`, `opendart`. |
| **FinMind** | Taiwan (+more) | 50+ datasets: prices, financials, institutional flows, news | API `https://api.finmindtrade.com/api/v4/data` ; [finmind.github.io](https://finmind.github.io/) ([GitHub](https://github.com/FinMind/FinMind)) | Python `DataLoader` (`pip install FinMind`); dataset `TaiwanStockPrice` etc. Daily updates. |
| **TWSE Open API** | Taiwan | Official TWSE open data | [openapi.twse.com.tw](https://openapi.twse.com.tw/) (linked from [twse.com.tw](https://www.twse.com.tw/en/)) | Official prices, daily trading, indices, institutional investors. |
| **HKEX market data** | Hong Kong | Official statistics, Stock Connect flows | [hkex.com.hk](https://www.hkex.com.hk/) ; [Stock Connect stats](https://www.hkex.com.hk/Mutual-Market/Stock-Connect/Statistics/Historical-Daily?sc_lang=en) | Southbound/northbound daily flows; HSI data from [hsi.com.hk](https://www.hsi.com.hk/). |

**Suffix / ticker cheat-sheet:** Tokyo numeric+`.T`; Hong Kong numeric (zero-padded)+`.HK`; Korea 6-digit numeric + `.KS`/`.KQ`; Taiwan 4-digit numeric + `.TW`/`.TWO`. Index symbols listed in the table above.

---

## 8. Access for Brazilian investors (acesso para brasileiros)

A *pessoa física* in Brazil can reach these markets without opening a foreign brokerage account:

- **US-listed country ETFs** (the simplest indirect route): **EWJ** — iShares MSCI Japan ([BlackRock EWJ](https://www.blackrock.com/us/individual/products/239665/ishares-msci-japan-etf)); **EWY** — iShares MSCI South Korea ([BlackRock EWY](https://www.blackrock.com/us/individual/products/239681/ishares-msci-south-korea-capped-etf)); **EWT** — iShares MSCI Taiwan; **EWH** — iShares MSCI Hong Kong. Held via a US/global broker or replicated through BDRs of ETFs on B3.
- **BDRs (Brazilian Depositary Receipts)** on **B3** let you buy certificates backed by foreign shares/ETFs in **reais (R$)**, on a Brazilian exchange, without currency conversion or an offshore account ([B3 – BDRs](https://www.b3.com.br/pt_br/produtos-e-servicos/solucoes-para-emissores/bdrs-brazilian-depositary-receipts/); [B3 – BDR explained](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/bdrs/bdr-o-que-e-e-como-funciona/)). BlackRock + B3 list ETF-backed BDRs covering Asian/regional indices, broadening access to these markets.
- **Direct access** (abrir conta no exterior) is possible via international brokers but adds FX (*câmbio*), tax-reporting (*IRPF / carnê-leão*) and custody complexity — for most retail the ETF/BDR route is far simpler.

---

## Sources

- JPX — Overview of Market Restructuring: https://www.jpx.co.jp/english/equities/improvements/market-structure/01.html
- JPX — Results of Market Segment Selection (2022): https://www.jpx.co.jp/english/corporate/news/news-releases/0060/20220111-01.html
- JPX — T+2 Settlement Cycle: https://www.jpx.co.jp/english/equities/clearing-settlement/tplus2-settlement-cycle/index.html
- JPX — Stock Price Index FAQ (Nikkei/TOPIX): https://www.jpx.co.jp/english/faq/stock_price_index.html
- Nikkei Inc. — Nikkei Stock Average FAQ (PDF): https://indexes.nikkei.co.jp/nkave/archives/faq/faq_nikkei_stock_average_en.pdf
- JPX — J-Quants API: https://www.jpx.co.jp/english/markets/other-data-services/j-quants-api/index.html
- J-Quants (registration): https://jpx-jquants.com/en
- HKEX — Stock Connect: https://www.hkex.com.hk/Mutual-Market/Stock-Connect?sc_lang=en
- HKEX — Stock Connect FAQ (PDF): https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Getting-Started/Information-Booklet-and-FAQ/FAQ/FAQ_En.pdf
- Clearstream — Hong Kong settlement: https://www.clearstream.com/clearstream-en/res-library/market-coverage/settlement-process-hong-kong-1281186
- Hang Seng Indexes — HSI methodology (PDF): https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/methodologies/IM_hsie.pdf
- Hang Seng Indexes — HSTECH methodology (PDF): https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/methodologies/IM_hsteche.pdf
- KRX — Guide to Trading in the Korean Stock Market (PDF): https://global.krx.co.kr/contents/GLB/01/0109/0109000000/guide_to_trading_in_the_korean_stock_market.pdf
- KRX Information Data System: https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd?locale=en
- OpenDART (Korea filings): https://engopendart.fss.or.kr/
- TWSE — Trading Mechanism: https://www.twse.com.tw/en/products/system/trading.html
- TWSE — home / open data: https://www.twse.com.tw/en/
- FinMind (Taiwan data API): https://finmind.github.io/ — GitHub: https://github.com/FinMind/FinMind
- pykrx (Korea library): https://github.com/sharebook-kr/pykrx — PyPI: https://pypi.org/project/pykrx/
- BlackRock iShares MSCI Japan (EWJ): https://www.blackrock.com/us/individual/products/239665/ishares-msci-japan-etf
- BlackRock iShares MSCI South Korea (EWY): https://www.blackrock.com/us/individual/products/239681/ishares-msci-south-korea-capped-etf
- B3 — BDRs: https://www.b3.com.br/pt_br/produtos-e-servicos/solucoes-para-emissores/bdrs-brazilian-depositary-receipts/

**Keywords:** Japan stock market, Tokyo Stock Exchange, JPX, Nikkei 225, TOPIX, Prime Standard Growth, HKEX, Hang Seng Index, Hang Seng TECH, Stock Connect southbound, KRX, KOSPI, KOSDAQ, price limit, TWSE, TAIEX, Taipei Exchange, T+2 settlement, J-Quants API, yfinance, pykrx, FinMind, OpenDART, EWJ, EWY, EWT, BDR, quant, machine learning, microstructure — *mercados asiáticos, bolsa de Tóquio, bolsa de Hong Kong, bolsa da Coreia, bolsa de Taiwan, liquidação T+2, limite de oscilação diária, BDR, ETF, quant, aprendizado de máquina, acesso brasileiro, câmbio.*
