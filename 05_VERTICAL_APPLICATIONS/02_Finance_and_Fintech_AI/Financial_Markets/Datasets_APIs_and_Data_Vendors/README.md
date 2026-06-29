# Datasets, APIs & Data Vendors

> Where to get market and reference data for research — free APIs, tick/LOB data, academic databases, and crypto feeds.

## Free / low-cost market-data APIs

| Source | Coverage | Link |
|---|---|---|
| yfinance | Yahoo Finance EOD + intraday (equities, ETFs, FX, crypto) — free, unofficial | https://github.com/ranaroussi/yfinance |
| Alpha Vantage | Stocks, FX, crypto, fundamentals (free tier) | https://www.alphavantage.co/ |
| Polygon.io | US stocks/options/FX/crypto, real-time + historical | https://polygon.io/ |
| Alpaca | Commission-free trading + market data API | https://alpaca.markets/ |
| IEX Cloud / Tiingo | Equities, fundamentals, news | https://www.tiingo.com/ |
| Nasdaq Data Link (Quandl) | Curated economic/financial datasets | https://data.nasdaq.com/ |
| FRED | Macro/rates/economic series (free) | https://fred.stlouisfed.org/ |
| SEC EDGAR | Filings + XBRL fundamentals (free) | https://www.sec.gov/edgar |
| Financial Modeling Prep / SimFin | Fundamentals & statements | https://site.financialmodelingprep.com/ |
| OpenBB | Open-source investment research aggregator | https://openbb.co/ |

## Tick & limit-order-book data

- **LOBSTER** — reconstructed Nasdaq LOB (academic). https://lobsterdata.com/
- **Databento** — normalized historical/real-time market data (MBO/MBP). https://databento.com/
- **Nasdaq TotalView-ITCH**, **CME DataMine/MBO**, Dukascopy (FX tick).

## Academic / institutional databases

- **WRDS** (Wharton Research Data Services) — gateway to CRSP, Compustat, TAQ, IBES, OptionMetrics. https://wrds-www.wharton.upenn.edu/
- **CRSP** (returns, delisting), **Compustat** (fundamentals), **TAQ** (trades & quotes), **OptionMetrics** (Ivy DB options).
- Bloomberg, Refinitiv/LSEG, FactSet (commercial terminals/feeds).

## Crypto data

- **CCXT** (unified exchange API), Binance/Coinbase/Kraken APIs, CoinGecko, CoinMarketCap, Kaiko, Amberdata, Deribit (options). On-chain: Etherscan, Dune, The Graph, Glassnode.

## Benchmark/competition datasets

- Kaggle: *Jane Street Market Prediction*, *Optiver Realized Volatility*, *Two Sigma / G-Research Crypto*, *JPX Tokyo Stock Exchange*. See sibling [`../../../`](../../../) Kaggle solutions and [Banking Datasets](../../Banking/Datasets_and_Benchmarks/).
- Numerai — encrypted tournament dataset. https://numer.ai/

## Notes & caveats

- Always check for **survivorship bias** (delisted names) and **point-in-time** correctness (restated fundamentals). Free APIs have rate limits and adjustment quirks — validate corporate-action adjustments before modeling.

## Related in AIForge
- [Backtesting & Frameworks](../Backtesting_and_Frameworks/) · [Tools & Platforms](../Tools_and_Platforms/) · [Equities & Stock Markets](../Equities_and_Stock_Markets/) · [Market Microstructure & HFT](../Market_Microstructure_and_HFT/)
- [`../../../../03_DATASETS_TOOLS_AND_RESOURCES/`](../../../../03_DATASETS_TOOLS_AND_RESOURCES/)

**Keywords:** market data API, yfinance, Polygon, Alpaca, Tiingo, FRED, SEC EDGAR, LOBSTER, Databento, WRDS, CRSP, Compustat, CCXT, Numerai, financial datasets.
