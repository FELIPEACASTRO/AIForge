# Global Datasets for Financial Markets

> A consolidated, country-agnostic catalog of **datasets** for market machine learning that work on **any market in the world** — equities, ETFs, indices, futures, options, FX, bonds, and crypto. Everything here is general-purpose: a US, Brazilian (B3), European, Indian, or Japanese instrument flows through the same loaders. Where a source is region-locked, it is flagged explicitly. This page complements the repo's **Kaggle — Finance Competitions & Datasets** and **HuggingFace — Finance Datasets & Models** pages.

This is a **data sourcing reference** (catálogo de fontes de dados). It maps where to pull price, fundamental, macro, order-book, options, FX, crypto, and news/alt-data — with the free vs. paid status and a working URL for each. Use it together with the feature-engineering and modeling pages in this same folder.

---

## 1. Equities, ETFs & Indices — Price + Fundamentals

These cover global equities/ETFs/indices (cobertura global de ações, ETFs e índices). "Adj." = split/dividend adjusted. Most provide US plus international tickers; a few are US-only (flagged).

| Source | What it provides | Free / Paid | Link |
|---|---|---|---|
| **Stooq** | Bulk EOD OHLCV for global equities, indices, FX, commodities, crypto; downloadable flat files (no formal API) | Free | https://stooq.com/db/h/ |
| **yfinance** (Yahoo Finance backend) | Python lib for EOD + intraday OHLCV, dividends, splits, fundamentals for global tickers; research-grade, not production-stable | Free (unofficial) | https://github.com/ranaroussi/yfinance |
| **Nasdaq Data Link** (ex-Quandl) | Marketplace of curated financial/economic datasets; many free + premium databases under one API | Free + Paid | https://data.nasdaq.com/ |
| **Tiingo** | EOD prices for 65k+ global stocks/ETFs/mutual funds/ADRs incl. splits & dividends; news API; free tier (rate-limited) | Free + Paid | https://www.tiingo.com/ |
| **EOD Historical Data (EODHD)** | EOD + intraday + fundamentals + options for 70+ exchanges worldwide; deep history on paid plans | Free trial + Paid | https://eodhd.com/ |
| **Sharadar** (via Nasdaq Data Link) | US Core Fundamentals + prices, ~25y history, **point-in-time**, includes delisted tickers (low survivorship bias). US-only | Paid | https://data.nasdaq.com/databases/SF1 |
| **Financial Modeling Prep (FMP)** | Real-time + historical statements, ratios, prices, economic & alt-data via REST; broad global coverage | Free tier + Paid | https://site.financialmodelingprep.com/ |
| **SimFin** | Standardized fundamentals + prices for 5,000+ companies; bulk download, screener, Python SDK | Free + Paid | https://www.simfin.com/ |
| **Alpha Vantage** | Free API: EOD/intraday OHLCV, FX, crypto, fundamentals, technical indicators; global coverage, rate-limited free key | Free + Paid | https://www.alphavantage.co/ |

### Academic / point-in-time fundamentals (institutional)

| Source | What it provides | Free / Paid | Link |
|---|---|---|---|
| **CRSP** | US stock returns, prices, shares, delisting — the academic gold standard; survivorship-bias-free. US-only | Paid (institutional) | https://www.crsp.org/ |
| **Compustat** (S&P) | Global + North America fundamentals, point-in-time financial statements | Paid (institutional) | https://www.spglobal.com/marketintelligence/ |
| **WRDS** (Wharton Research Data Services) | Unified access portal to CRSP, Compustat, OptionMetrics, TAQ, etc. for academics | Paid (institutional) | https://wrds-www.wharton.upenn.edu/ |
| **Kenneth French Data Library** | Fama–French 3/5-factor returns, momentum, industry & sorted portfolios; US + International/Global regions | Free | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html |
| **Global Factor Data (JKP)** | 153 firm characteristics / factor returns in **93 countries + 4 regions** (Jensen, Kelly, Pedersen 2023); updated through 2025 | Free (factors); stock-level via WRDS | https://jkpfactors.com/ |
| **JKP code (bkelly-lab)** | Python/SAS code to rebuild the global factor + characteristics dataset from WRDS | Free (code) | https://github.com/bkelly-lab/jkp-data |
| **AQR Data Sets** | Academic-quality factor/return series (value, momentum, defensive, BAB) across countries & assets | Free | https://www.aqr.com/Insights/Datasets |

---

## 2. Macro & Economic Time Series

Country-agnostic by design — these portals carry indicators for essentially every economy (indicadores de praticamente todos os países).

| Source | What it provides | Free / Paid | Link |
|---|---|---|---|
| **FRED** (St. Louis Fed) | 800,000+ US & international economic series (rates, inflation, FX, money, activity); excellent free API | Free | https://fred.stlouisfed.org/ |
| **World Bank Open Data** | World Development Indicators + Data360 API; 200+ countries, dev/macro/social indicators | Free | https://data.worldbank.org/ |
| **OECD Data Explorer** | National accounts, prices, labor, finance for OECD + partner economies; SDMX API | Free | https://data-explorer.oecd.org/ |
| **IMF Data** | World Economic Outlook, IFS, BOP, government finance; modern data portal + API | Free | https://data.imf.org/ |
| **BIS** (Bank for Intl. Settlements) | Cross-border banking, credit, debt securities, property prices, effective exchange rates | Free | https://data.bis.org/ |
| **Our World in Data** | Curated, downloadable long-run economic & social datasets (CSV + API) | Free | https://ourworldindata.org/ |

> Central-bank series (Fed, ECB, BoE, BoJ, **BCB/Banco Central do Brasil SGS**, etc.) are usually free via their own portals and complement FRED for non-US rates/FX.

---

## 3. Limit-Order-Book & Tick Data (Microstructure)

For HFT/microstructure ML you need depth (L2/L3) and message-level data. These are mostly venue-specific by nature; coverage is global across the listed providers.

| Source | What it provides | Free / Paid | Link |
|---|---|---|---|
| **LOBSTER** | Reconstructed limit order books (any depth) from Nasdaq TotalView-ITCH; message + order-book CSVs. US equities | Free (academic samples) + Paid | https://lobsterdata.com/ |
| **Databento** | Tick-level L1/L2/L3, trades & quotes for US equities, futures, options; historical + live, normalized schemas | Paid (free credits) | https://databento.com/ |
| **Nasdaq TotalView-ITCH** | Raw full-depth message feed (every add/cancel/execute); source format behind LOBSTER | Paid | https://databento.com/datasets/XNAS.ITCH |
| **Kaiko** | Institutional crypto tick + full order-book depth, 100+ exchanges, 35k+ pairs, indices/benchmarks. Crypto | Paid | https://www.kaiko.com/ |
| **Dukascopy Historical Data** | Free FX/CFD/commodity/index **tick** data, 1990s→present; via export tool or Python/Node downloaders | Free | https://www.dukascopy.com/swiss/english/marketwatch/historical/ |

> Open-source helpers: `dukascopy-node` (https://github.com/Leo4815162342/dukascopy-node) and `duka` (https://github.com/giuse88/duka) automate bulk tick downloads.

---

## 4. Options & Volatility Surfaces

| Source | What it provides | Free / Paid | Link |
|---|---|---|---|
| **ORATS** | Historical implied-vol surfaces, greeks, skew, earnings/dividend forecasts; 15+ yrs; REST + flat files. US options | Paid | https://orats.com/ |
| **Cboe DataShop** | Exchange-direct options trades, EOD summaries, greeks, VIX & vol indices over OPRA. US options/ETFs/indices | Paid (some free samples) | https://datashop.cboe.com/ |
| **Deribit** | **Free** API for crypto options (BTC/ETH) order books, trades, IV, greeks, historical vol. Crypto options | Free | https://docs.deribit.com/ |
| **OptionMetrics (IvyDB)** | Academic gold-standard historical IV surfaces + standardized greeks; US + global IvyDB. Via WRDS | Paid (institutional) | https://optionmetrics.com/ |

---

## 5. FX & Rates

| Source | What it provides | Free / Paid | Link |
|---|---|---|---|
| **Dukascopy** | Free FX tick/bar history (majors, minors, exotics); see §3 | Free | https://www.dukascopy.com/swiss/english/marketwatch/historical/ |
| **HistData.com** | Free historical FX 1-minute bars & tick (M1/T) for many pairs; bulk CSV download | Free | https://www.histdata.com/ |
| **FRED (FX & rates)** | Daily FX rates, policy & market rates, yield curves for many countries | Free | https://fred.stlouisfed.org/categories/15 |
| **ECB / central banks** | Reference FX rates and yield curves (e.g. ECB euro reference rates, BCB PTAX for BRL) | Free | https://www.ecb.europa.eu/stats/eurofxref/ |
| **Alpha Vantage FX** | Realtime + historical FX OHLCV via free API | Free + Paid | https://www.alphavantage.co/documentation/ |

---

## 6. Crypto & On-Chain

Crypto is inherently global and 24/7 — these are venue/chain agnostic.

| Source | What it provides | Free / Paid | Link |
|---|---|---|---|
| **CCXT** | Unified Python/JS/PHP library to fetch OHLCV, order books, trades from 100+ exchanges | Free (open source) | https://github.com/ccxt/ccxt |
| **CoinGecko API** | Most-used crypto data API: prices, OHLCV, market cap, categories, DeFi/NFT metrics; generous free tier | Free + Paid | https://www.coingecko.com/en/api |
| **CoinMarketCap API** | Real-time quotes, OHLCV, listings, exchange data; tiered commercial plans | Free + Paid | https://coinmarketcap.com/api/ |
| **Kaiko** | Institutional tick + order-book + indices/benchmarks (see §3) | Paid | https://www.kaiko.com/ |
| **Glassnode** | 800+ on-chain metrics (SOPR, MVRV, flows, supply) across major chains | Free tier + Paid | https://glassnode.com/ |
| **Dune** | SQL over decentralized/on-chain data; query DEX, lending, NFT activity; shareable dashboards | Free + Paid | https://dune.com/ |

---

## 7. News, Sentiment & Alternative Data

Text/NLP datasets for sentiment, NER, QA, and event studies. Most are English; for **Portuguese (PT-BR) finance** models/datasets see the repo's HuggingFace page.

| Dataset / Source | What it provides | Free / Paid | Link |
|---|---|---|---|
| **FNSPID** | 29.7M stock prices + 15.7M news records for 4,775 S&P500 firms (1999–2023); time-series aligned | Free | https://huggingface.co/datasets/Zihan1004/FNSPID |
| **Financial PhraseBank** | 4,840 finance sentences labeled positive/neutral/negative by 16 finance annotators | Free | https://huggingface.co/datasets/takala/financial_phrasebank |
| **FiQA (sentiment/QA)** | Aspect-based financial sentiment + opinion QA benchmark (~1k sentiment samples) | Free | https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification |
| **Twitter Financial News Sentiment** | ~11,900 finance tweets (9,938 train + 2,486 validation) labeled Bullish/Bearish/Neutral | Free | https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment |
| **GDELT** | Global news event + tone database, updated continuously; BigQuery + API, every country/language | Free | https://www.gdeltproject.org/ |
| **RavenPack** (now Bigdata.com) | Commercial structured news/sentiment analytics with entity tagging & event detection | Paid | https://www.ravenpack.com/ |

---

## 8. Benchmarks, Competitions & Curated Lists

The best public **benchmarks** for market prediction come from competition platforms; curated lists keep you current.

| Source | What it provides | Free / Paid | Link |
|---|---|---|---|
| **Kaggle — Jane Street** | Real HFT trading-opportunity data; flagship market-prediction benchmark ($100k) | Free | https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting |
| **Kaggle — Optiver** | Realized-volatility prediction over 10-min windows, hundreds of stocks; microstructure features | Free | https://www.kaggle.com/competitions/optiver-realized-volatility-prediction |
| **Kaggle — Two Sigma / G-Research** | News-to-stock and crypto return-prediction challenges | Free | https://www.kaggle.com/competitions?tagIds=11108-Finance |
| **Numerai** | Obfuscated global-equity feature matrices; predict weekly stock rankings for a market-neutral fund | Free (incentivized) | https://numer.ai/ |
| **Numerai Signals** | Bring-your-own-data tournament for global stock signals | Free (incentivized) | https://signals.numer.ai/ |
| **paperswithbacktest** | Curated datasets + strategy backtests; alt-data and asset-class data hub | Free + Paid | https://paperswithbacktest.com/datasets |
| **HuggingFace finance datasets** | Hub-wide finance datasets (filings, news, time-series, QA); see repo HuggingFace page | Free | https://huggingface.co/datasets?search=finance |
| **awesome-quant** (wilsonfreitas) | Maintained list of quant libraries + **Data Sources** section across all asset classes | Free | https://github.com/wilsonfreitas/awesome-quant |

---

## How to Choose a Dataset (checklist)

Picking market data is mostly about avoiding subtle biases that inflate backtests. Before you commit:

- **License & redistribution.** Yahoo/yfinance and Stooq are research-grade and *unofficial* — fine for study, risky for production or redistribution. Read each provider's terms (termos de uso) before publishing results or shipping a product.
- **Survivorship bias (viés de sobrevivência).** Free price feeds usually drop delisted/bankrupt tickers, biasing returns upward. For honest backtests prefer sources that **include delisted names** (CRSP, Sharadar, JKP stock-level).
- **Point-in-time (PIT) data.** Fundamentals get restated. To avoid look-ahead bias use PIT vintages (Sharadar, Compustat PIT, JKP) rather than today's "as-reported" snapshot.
- **Corporate-action adjustment.** Confirm whether prices are split/dividend **adjusted** and whether total-return vs. price-return. Mixing adjusted and raw series silently corrupts features.
- **Coverage & history depth.** Check the exchange list and start date for *your* market — many "global" APIs are deepest on US tickers and thin on emerging markets (e.g. B3, India, frontier).
- **Frequency match.** EOD for daily strategies; L2/L3 tick (LOBSTER, Databento, Kaiko) only when microstructure actually drives your signal — it is far larger and costlier.
- **Time zone & session alignment.** Aligning a Brazilian equity, a US option, and an FX tick requires consistent UTC timestamps and explicit market-session handling.
- **Free → validate → upgrade.** Start on free tiers, cross-check two sources for agreement, then pay only where you need quota, depth, or PIT guarantees.

---

## Cross-References (this repo)

- **Kaggle — Finance Competitions & Datasets** — flagship quant benchmarks and ready-made stock/options/crypto/FX/sentiment datasets. Path: `Datasets_APIs_and_Data_Vendors/Kaggle_Finance_Competitions_and_Datasets.md`
- **HuggingFace — Finance Datasets & Models** — sentiment/NLP/time-series datasets and models, including **Portuguese (PT-BR)** finance models. Path: `Datasets_APIs_and_Data_Vendors/HuggingFace_Finance_Datasets_and_Models.md`
- **Features & Feature Engineering** and **Models for Markets** — in this same `Models_Features_and_Datasets/` folder.
- **Datasets, APIs & Data Vendors** — broader vendor/API directory.

---

**Sources:** Stooq (https://stooq.com/db/h/) · yfinance (https://github.com/ranaroussi/yfinance) · Nasdaq Data Link (https://data.nasdaq.com/) · Tiingo (https://www.tiingo.com/) · EODHD (https://eodhd.com/) · Sharadar SF1 (https://data.nasdaq.com/databases/SF1) · FMP (https://site.financialmodelingprep.com/) · SimFin (https://www.simfin.com/) · Alpha Vantage (https://www.alphavantage.co/) · CRSP (https://www.crsp.org/) · WRDS (https://wrds-www.wharton.upenn.edu/) · Kenneth French Data Library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) · JKP Global Factor Data (https://jkpfactors.com/) · bkelly-lab/jkp-data (https://github.com/bkelly-lab/jkp-data) · AQR Datasets (https://www.aqr.com/Insights/Datasets) · FRED (https://fred.stlouisfed.org/) · World Bank (https://data.worldbank.org/) · OECD (https://data-explorer.oecd.org/) · IMF (https://data.imf.org/) · BIS (https://data.bis.org/) · Our World in Data (https://ourworldindata.org/) · LOBSTER (https://lobsterdata.com/) · Databento (https://databento.com/) · Nasdaq ITCH (https://databento.com/datasets/XNAS.ITCH) · Kaiko (https://www.kaiko.com/) · Dukascopy (https://www.dukascopy.com/swiss/english/marketwatch/historical/) · ORATS (https://orats.com/) · Cboe DataShop (https://datashop.cboe.com/) · Deribit (https://docs.deribit.com/) · OptionMetrics (https://optionmetrics.com/) · HistData (https://www.histdata.com/) · CCXT (https://github.com/ccxt/ccxt) · CoinGecko (https://www.coingecko.com/en/api) · CoinMarketCap (https://coinmarketcap.com/api/) · Glassnode (https://glassnode.com/) · Dune (https://dune.com/) · FNSPID (https://huggingface.co/datasets/Zihan1004/FNSPID) · Financial PhraseBank (https://huggingface.co/datasets/takala/financial_phrasebank) · FiQA (https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification) · Twitter Financial News Sentiment (https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) · GDELT (https://www.gdeltproject.org/) · RavenPack (https://www.ravenpack.com/) · Kaggle Jane Street (https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) · Kaggle Optiver (https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) · Numerai (https://numer.ai/) · Numerai Signals (https://signals.numer.ai/) · paperswithbacktest (https://paperswithbacktest.com/datasets) · awesome-quant (https://github.com/wilsonfreitas/awesome-quant)

**Keywords:** financial market datasets, global stock data, equities ETFs futures options FX crypto bonds, point-in-time fundamentals, limit order book tick data, implied volatility surface, on-chain data, macroeconomic time series, news sentiment dataset, survivorship bias, backtesting data; dados de mercado financeiro, ações ETFs futuros opções câmbio cripto títulos, dados fundamentalistas, livro de ofertas, volatilidade implícita, séries macroeconômicas, análise de sentimento, viés de sobrevivência, dados para backtest
