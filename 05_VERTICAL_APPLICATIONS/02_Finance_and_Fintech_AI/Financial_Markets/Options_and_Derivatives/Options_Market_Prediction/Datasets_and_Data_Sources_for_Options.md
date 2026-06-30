# Datasets & Data Sources for Options Prediction

> Where to actually *get* options data for prediction research — implied/realized volatility, IV surfaces, option prices/returns, flow, and Greeks — with real URLs, free-vs-paid status, coverage, and the licensing/quality traps that silently break backtests. Global focus, ML/quant oriented, current to 2024–2026, with a Brazil/B3 (*opções*) and crypto lens.

The hard truth up front: **clean historical implied-volatility surfaces are expensive**, and most "free" options data is delayed, snapshot-timed inconsistently, and survivorship-biased (expired/delisted contracts silently vanish). Equity-options prediction is also a brutally **low signal-to-noise** problem constrained by **no-arbitrage** (a forecast that violates put-call parity or a non-negative density is wrong by construction), so the data layer is where most projects quietly fail. Pick your source by what you can *afford to be wrong about*.

---

## 1. Quick decision table — which source for which job

| If you need… | Best free option | Best paid / gold standard |
|---|---|---|
| US equity/index **IV surface** (constant-maturity, by delta) | none truly free | **OptionMetrics IvyDB** (via WRDS) / **ORATS** |
| US **EOD option chains + Greeks/IV** | yfinance, Alpha Vantage `HISTORICAL_OPTIONS` | Polygon.io, Cboe DataShop, ORATS |
| **Intraday / tick** US options | — | Cboe DataShop (LiveVol), Polygon.io flat files |
| **Crypto options** (24/7, deep history) | Deribit API; Tardis.dev 1st-of-month free | Tardis.dev, Amberdata, Laevitas |
| **VIX / VVIX / SKEW / PUT** term-structure | Cboe (free CSV), FRED, datahub.io | Cboe DataShop |
| **India NSE F&O** chains/bhavcopy | NSE bhavcopy, getbhavcopy, Kaggle | NSE paid EOD subscription |
| **Brazil B3 opções** | MetaTrader5 (MT5) Python, opcoes.net.br | OpLab, Nelogica DataSolution, B3 UP2DATA |
| **ML benchmark** (vol from order book) | Kaggle Optiver (free with rules) | — |

---

## 2. Academic / gold-standard sources

These are what referees expect in a published equity-options paper. They solve the two problems retail data does not: **correctly computed IV/Greeks** and **a standardized, interpolated surface** that is comparable across days and names.

| Source | Content | Coverage | Free/Paid | How to access |
|---|---|---|---|---|
| **OptionMetrics IvyDB US** | EOD option prices, **correctly computed IV + Greeks**, and a **standardized constant-maturity IV surface** (by delta × expiry) | US equity & index options, **from Jan 1996** | Paid (institutional) | Via **WRDS** ([wrds-www.wharton.upenn.edu/…/optionmetrics](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/)); vendor: [optionmetrics.com/united-states](https://optionmetrics.com/united-states/) |
| **OptionMetrics IvyDB Europe / Asia / ETF** | Same model for non-US and US ETF options | Europe, parts of Asia, US ETFs | Paid | [optionmetrics.com/data-products](https://optionmetrics.com/data-products/) |
| **ORATS** | Full chains, smoothed IVs, **historical vol-surface** (ATM vol, slope, curvature/derivative), IV rank/percentile, skew/kurtosis, HV forecasts, **options flow** | US options, **EOD back to 2007** (5,000+ symbols); intraday by minute | Paid (API tiers) | [orats.com/data-api](https://orats.com/data-api), docs [orats.com/docs/historical-data-api](https://orats.com/docs/historical-data-api); also resold via **Nasdaq Data Link** ([data.nasdaq.com/databases/OPT](https://data.nasdaq.com/databases/OPT)) |
| **Cboe DataShop (LiveVol)** | EOD summary, **option quotes (NBBO) — DataShop history from 2004**, trade-by-trade, open interest, calcs (Greeks/IV) | US options, multi-asset | Paid (one-off + subscription) | [datashop.cboe.com](https://datashop.cboe.com/) |

Notes:
- **OptionMetrics' surface is its killer feature** for ML: instead of a ragged, inconsistently-strike'd raw chain, you get IV interpolated to fixed deltas and fixed days-to-expiry (e.g. expirations of 30, 60, 91, … 730 calendar days), so a model sees a stationary tensor across the whole panel. ORATS offers a comparable smoothed surface.
- Access to IvyDB is normally bundled in a **WRDS** subscription held by a university/desk; there is no retail tier.

---

## 3. Free / retail US sources

Good for prototyping, paper-trading signals, and learning. **Not** safe for a publishable historical study: bid/ask appear only in market hours, snapshots are ~15 min delayed, and expired contracts are not retained, so you cannot reconstruct a clean point-in-time surface.

| Source | Content | Coverage / history | Free? | How |
|---|---|---|---|---|
| **yfinance** (Yahoo) | Live-ish option chain (calls/puts DataFrames), IV, OI | Spot snapshot only; **no clean history** | Free, no key | `pip install yfinance`; `Ticker.options`, `Ticker.option_chain(exp)`. Docs [ranaroussi.github.io/yfinance](https://ranaroussi.github.io/yfinance/), [PyPI](https://pypi.org/project/yfinance/). Unofficial — personal/research use |
| **Alpha Vantage** | `HISTORICAL_OPTIONS`: full chain + **IV + Greeks** for a given date | US, **back to 2008-01-01** (15+ years); realtime needs Premium | Free tier (rate-limited) | [alphavantage.co/documentation](https://www.alphavantage.co/documentation/) |
| **Polygon.io** | Trades, quotes, candles, **Greeks & IV**, OI; chain snapshot; **flat-file (S3) downloads** | US full market; **quotes back to 2022, trades to 2016** | Paid (Business tier entry; no real free options tier) | REST/WebSocket/S3 — [polygon.io/options](https://polygon.io/options), docs [polygon.io/docs/rest/options](https://polygon.io/docs/rest/options/contracts/contract-overview) |
| **Tradier** | Real-time quotes, option chains; **Greeks/IV courtesy of ORATS** | US; live + limited history | Paid/brokerage (sandbox key) | [docs.tradier.com](https://docs.tradier.com/docs/market-data) |
| **Cboe delayed quotes** | Delayed option quotes & daily market stats | US listed | Free (delayed) | [cboe.com/us/options/market_statistics/historical_data](https://www.cboe.com/us/options/market_statistics/historical_data/) |
| **Nasdaq Data Link** | Marketplace reselling ORATS surfaces (`OPT`), option-vol datasets | Vendor-dependent | Mostly paid | [data.nasdaq.com](https://data.nasdaq.com/) |

Rule of thumb: **free → live signals & teaching; paid → backtests you'd stake money or a paper on.**

---

## 4. Crypto options — free, rich, and 24/7

The best price/quality ratio for *learning IV-surface dynamics* lives in crypto: deep liquid BTC/ETH chains, **markprice + IV streamed per option**, no exchange-redistribution licensing, and continuous (24/7) data with no overnight gap. **Deribit** is the dominant venue and its API is openly documented.

| Source | Content | Coverage | Free? | How |
|---|---|---|---|---|
| **Deribit API** | Full BTC/ETH (+ others) option chain, **`markprice.options` = mark price + IV per option** (data since 2019-10-01), `ticker` with Greeks & OI, trades, order book | Deribit options, deep history via WS v2 | Free API (key optional for public) | [docs.deribit.com](https://docs.deribit.com/) |
| **Tardis.dev** | Tick-level historical: incremental L2 book, options chains, **`markprice.options` IV stream**, quotes, derivative tickers, liquidations; CSV datasets | Deribit, OKX, Bybit, Binance European, HTX options | Paid, but **1st-of-each-month files free without key**; Deribit history via partnership | [tardis.dev](https://tardis.dev/), docs [docs.tardis.dev/historical-data-details/deribit](https://docs.tardis.dev/historical-data-details/deribit) |
| **Amberdata (AD Derivatives)** | Normalized crypto IV (computed where exchange doesn't supply), OHLCV, OI, book events/snapshots, trades, liquidations | Deribit + major venues | Paid (institutional) | [amberdata.io/ad-derivatives](https://www.amberdata.io/ad-derivatives) |
| **Laevitas** | Option chains, **IV by strike** (delta/gamma/theta/vega), Greeks, OI, funding; REST/WebSocket/**MCP**/pay-per-request | Binance, Deribit, OKX, Bybit, Hyperliquid | Paid (free app dashboards) | [laevitas.ch](https://www.laevitas.ch/), API docs [docs.laevitas.ch/options/historical](https://docs.laevitas.ch/options/historical) |
| **CryptoDataDownload** | Free Deribit options/futures/trade CSVs | Deribit | Free | [cryptodatadownload.com/data/deribit](https://www.cryptodatadownload.com/data/deribit/) |

If you want to teach a model the *mechanics* of a volatility surface (smile, term structure, calendar/strike arbitrage), **start on Deribit** — the data is free, clean, and arrives with IV already attached.

---

## 5. India (NSE) — F&O is one of the world's most active options markets by volume

| Source | Content | Coverage | Free? | How |
|---|---|---|---|---|
| **NSE bhavcopy (F&O)** | Daily EOD F&O bhavcopy (option/future settlement, OI, volume) | NSE F&O, clean form from ~2020 onward | Free | [nseindia.com](https://www.nseindia.com/) static reports |
| **getbhavcopy** | Bulk downloader for NSE/BSE EOD (equities, indices, futures) | NSE/BSE | Free tool | [getbhavcopy.com](https://www.getbhavcopy.com/) ([github.com/hemenkapadia/getbhavcopy](https://github.com/hemenkapadia/getbhavcopy)) |
| **`nser` (R)** | `fobhav()` downloads NSE F&O bhavcopy **from 1 Jan 2020**; live data | NSE/BSE | Free (CRAN) | [github.com/nandp1/nser](https://github.com/nandp1/nser/) |
| **Kaggle NSE datasets** | "NSE Daily Bhavcopy" (stocks/options/futures) / daily bhavcopy mirrors | NSE | Free | search Kaggle "Nifty options" / [NSE Daily Bhavcopy](https://www.kaggle.com/datasets/akshaypawar7/nse-daily-bhavcopy) |
| **NSE paid EOD** | Official order/trade EOD with full F&O detail | NSE | Paid | NSE Data Services subscription |

Caveat: NSE's website actively rate-limits/blocks scraping; prefer the maintained wrappers (`nser`, getbhavcopy) over hand-rolled scrapers.

---

## 6. Brazil (B3) — *opções sobre ações e sobre Ibovespa*

B3 options liquidity concentrates in a handful of names (PETR4, VALE3, BOVA11, *opções de Ibovespa*) plus weekly *minicontratos*-adjacent series. There is **no free OptionMetrics-grade surface for B3** — you build IV yourself from chains.

| Source | Content | Coverage | Free? | How |
|---|---|---|---|---|
| **MetaTrader 5 (MT5) Python API** | EOD/intraday quotes for any B3 asset with authorized option series | B3 (via broker feed) | Free (needs broker MT5) | Python `MetaTrader5` package; guide: [Asimov Academy — baixar dados B3 com MT5](https://hub.asimov.academy/blog/como-baixar-dados-da-b3-com-metatrader-5-e-python/) |
| **opcoes.net.br** | Cotações de opções, **volatilidade histórica e implícita** por ação B3 | B3 equity options | Free (web) / paid plans | [opcoes.net.br/acoes](https://opcoes.net.br/acoes) |
| **OpLab** | Ferramentas de opções, IV rank/percentile, séries de vol histórica/implícita, simulador | B3 options | Freemium | [oplab.com.br](https://oplab.com.br/) |
| **Nelogica DataSolution** | Base histórica B3 + DLL tempo real (Profit) | B3 | Paid | [blog.nelogica.com.br/dados-historicos-da-b3](https://blog.nelogica.com.br/dados-historicos-da-b3/) |
| **B3 UP2DATA** | Arquivos oficiais B3 (negócios, instrumentos, séries) | B3 official | Paid | [b3.com.br — UP2DATA dados disponíveis](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/up2data/dados-disponiveis/) |
| **b3-api-dados-historicos** | Wrapper open-source para histórico B3 | B3 | Free (GitHub) | [github.com/cvscarlos/b3-api-dados-historicos](https://github.com/cvscarlos/b3-api-dados-historicos) |

Practical Brazil note: for an ML study, the cleanest free path is **MT5 chains + your own Black-Scholes/binomial IV solver** (using the B3 dividend/selic conventions), cross-checked against opcoes.net.br implied vols. Watch B3's *ajuste* and corporate-action conventions — they distort strikes if ignored.

---

## 7. Volatility & term-structure indices (VIX complex)

For prediction *of* vol or as features, the Cboe index family is the canonical, **free** time series.

| Series | What it is | Free source(s) |
|---|---|---|
| **VIX** | 30-day implied vol of S&P 500 options, **daily 1990→present** | [Cboe VIX historical](https://www.cboe.com/tradable_products/vix/vix_historical_data); [FRED VIXCLS](https://fred.stlouisfed.org/series/VIXCLS); [datahub.io/core/finance-vix](https://datahub.io/core/finance-vix); [github.com/datasets/finance-vix](https://github.com/datasets/finance-vix) |
| **VVIX** | Vol-of-VIX (vol of VIX options) | Cboe global indices |
| **SKEW** | Tail-risk / risk-neutral skewness of S&P 500 | [Cboe SKEW dashboard](https://www.cboe.com/us/indices/dashboard/skew/) |
| **PUT / BXM / other strategy indices** | Put-write / buy-write benchmark indices | [cboe.com/us/indices/indicesproducts](https://www.cboe.com/us/indices/indicesproducts/) |

The GitHub/datahub VIX mirrors expose a **stable hardcodable CSV URL** (`github.com/datasets/finance-vix`, sourced from Cboe's own `VIX_History.csv`) — convenient for reproducible pipelines and AI agents.

---

## 8. ML benchmark / competition datasets

The cleanest, dispute-free training data for *volatility prediction* methodology — order book and trades, not chains, but directly about the quantity options price off.

| Dataset | Task | Data | Where | License/notes |
|---|---|---|---|---|
| **Optiver — Realized Volatility Prediction** (2021) | Predict 10-min **realized vol** per stock from L2 book + trades; metric **RMSPE** | `book.parquet`, `trade.parquet`, `train.csv` (hundreds of millions of rows) | [kaggle.com/competitions/optiver-realized-volatility-prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) | Free under Kaggle comp rules; gold reference for vol features |
| **Optiver — Trading at the Close** (2023) | Predict Nasdaq **closing-auction** price moves vs synthetic index | Closing-auction order book, Sep–Dec 2023 | [kaggle.com/competitions/optiver-trading-at-the-close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) | Microstructure/feature-engineering benchmark |
| **paperswithbacktest (HF)** | Daily/1-min price panels (stocks, ETFs, FX, crypto, indices, news) — underlyings for options research | Multi-asset | [huggingface.co/paperswithbacktest](https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price) | Free dataset cards; full access via subscription |

Strong public solution writeups exist (e.g. [github.com/ZhendongTian/Optiver-Realized-Volatility-Prediction](https://github.com/ZhendongTian/Optiver-Realized-Volatility-Prediction)) — useful for vol-feature engineering even outside the competition.

> HuggingFace caveat: searching "options"/"derivatives" on the Hub returns mostly *underlying* price panels and finance-LLM text corpora — there is **no widely-used, clean historical IV-surface dataset on HF** as of mid-2026. If a card claims one, verify the snapshot timing and contract universe before trusting it.

---

## 9. Data-quality traps that silently break options backtests

| Trap | What goes wrong | Mitigation |
|---|---|---|
| **Survivorship bias** | Expired/delisted contracts (and the worst-blown-up names) dropped → returns look too good | Use point-in-time sources (IvyDB/ORATS/Cboe) that retain dead contracts; never reconstruct history from a *current* chain (yfinance) |
| **Snapshot timing** | Option and underlying quoted at different instants → fake mispricings, garbage IV | Insist on synchronized EOD/intraday snaps; crypto (Deribit/Tardis) timestamps both sides |
| **Stale / wide quotes** | Illiquid strikes show last-trade or huge spreads → nonsensical IV, parity violations | Filter by OI/volume/spread; prefer mid; drop deep OTM |
| **IV computation** | DIY Black-Scholes with wrong rate/dividend/American-exercise → biased surface | Use vendor IV (IvyDB/ORATS) or validate your solver against it; B3/India need local conventions |
| **No-arbitrage violations** | Interpolated/predicted surface admits calendar or butterfly arbitrage | Enforce monotonicity / non-negative density (e.g. arbitrage-free SVI, "deep smoothing") |
| **Licensing/redistribution** | yfinance, Polygon, exchange feeds restrict redistribution & commercial use | Read terms; crypto APIs (Deribit) are the most permissive for open research |
| **Cost reality** | A clean multi-year US equity IV surface = **thousands of USD** (IvyDB/ORATS/DataShop) | Prototype on Deribit/Cboe-free; budget for paid data only once a signal survives |

---

## Sources

- OptionMetrics IvyDB: https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/ • https://optionmetrics.com/united-states/ • https://optionmetrics.com/data-products/
- ORATS: https://orats.com/data-api • https://orats.com/docs/historical-data-api • https://orats.com/near-eod-data • https://data.nasdaq.com/databases/OPT
- Cboe DataShop: https://datashop.cboe.com/ • https://www.cboe.com/us/options/market_statistics/historical_data/
- Cboe indices (VIX/SKEW): https://www.cboe.com/tradable_products/vix/vix_historical_data • https://www.cboe.com/us/indices/dashboard/skew/ • https://www.cboe.com/us/indices/indicesproducts/ • https://fred.stlouisfed.org/series/VIXCLS • https://datahub.io/core/finance-vix • https://github.com/datasets/finance-vix
- yfinance: https://ranaroussi.github.io/yfinance/ • https://pypi.org/project/yfinance/
- Alpha Vantage: https://www.alphavantage.co/documentation/
- Polygon.io: https://polygon.io/options • https://polygon.io/docs/rest/options/contracts/contract-overview
- Tradier: https://docs.tradier.com/docs/market-data
- Nasdaq Data Link: https://data.nasdaq.com/
- Deribit: https://docs.deribit.com/
- Tardis.dev: https://tardis.dev/ • https://docs.tardis.dev/historical-data-details/deribit
- Amberdata: https://www.amberdata.io/ad-derivatives
- Laevitas: https://www.laevitas.ch/ • https://docs.laevitas.ch/options/historical
- CryptoDataDownload: https://www.cryptodatadownload.com/data/deribit/
- NSE / India: https://www.getbhavcopy.com/ • https://github.com/hemenkapadia/getbhavcopy • https://github.com/nandp1/nser/ • https://www.kaggle.com/datasets/akshaypawar7/nse-daily-bhavcopy
- Brazil / B3: https://hub.asimov.academy/blog/como-baixar-dados-da-b3-com-metatrader-5-e-python/ • https://opcoes.net.br/acoes • https://oplab.com.br/ • https://blog.nelogica.com.br/dados-historicos-da-b3/ • https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/up2data/dados-disponiveis/ • https://github.com/cvscarlos/b3-api-dados-historicos
- Kaggle / benchmarks: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction • https://www.kaggle.com/competitions/optiver-trading-at-the-close • https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price • https://github.com/ZhendongTian/Optiver-Realized-Volatility-Prediction

**Keywords:** options data, implied volatility surface, realized volatility, option chain, Greeks, IV surface, OptionMetrics IvyDB, ORATS, Cboe DataShop, VIX, SKEW, VVIX, Deribit, Tardis.dev, crypto options, NSE F&O bhavcopy, B3 opções, yfinance, Polygon.io, Optiver realized volatility, survivorship bias, no-arbitrage, dados de opções, volatilidade implícita, superfície de volatilidade, volatilidade realizada, cadeia de opções, opções B3, volatilidade histórica, mercado de opções
