# Tick & High-Frequency Data Vendors

> The vendor map for raw market microstructure — tick-by-tick trades/quotes, L1/L2/L3 order-book feeds, PCAP, and the storage stack (Parquet, kdb+/q) you need to backtest HFT and microstructure strategies. Brazil 🇧🇷 redistributors included. Clean tick data is the single most expensive, license-encumbered, error-prone dataset class in quant — this page tells you who sells what, how deep, how far back, and what it costs.

## What "tick data" actually means (depth taxonomy)

Vendors and exchanges use a depth nomenclature that maps directly to schema names. Get this right or you will buy the wrong thing.

| Term | Synonym | What you see | Example schema |
|---|---|---|---|
| **L1 / Top-of-book** | BBO, MBP-1 | Best bid/ask + last trade, aggregated size | Databento `mbp-1`, `bbo-1s`; CME Top-of-Book (BBO) |
| **L2 / Depth-of-book** | MBP-N, market depth | Aggregated size at N price levels (e.g. 10) | Databento `mbp-10`; Nasdaq TotalView; CME Market Depth |
| **L3 / Full order book** | MBO, order-by-order | Every individual order: add/modify/cancel/execute, queue position, order IDs | Databento `mbo`; Nasdaq ITCH; CME MBO FIX; LOBSTER |
| **T&S / Time & Sales** | trades, tick-by-tick | Executed trades only (price, size, condition codes, aggressor) | Databento `trades`; CME Time & Sales |
| **PCAP** | packet capture | Raw exchange wire packets, GPS/nanosecond-stamped, pre-normalization | LSEG TH-PCAP; CME Packet Capture; Databento raw |

L3/MBO is the gold standard for microstructure (queue-position models, order-flow imbalance, OFI) — but it is also the largest, costliest, and least uniformly available historically. Most ML/quant work can run on L2 (`mbp-10`) or even L1 + trades. LOBSTER is already covered in [`README.md`](./README.md) (reconstructed Nasdaq L3 for academics — https://lobsterdata.com/).

## Normalized, multi-venue API vendors (the modern default)

These sell *normalized* tick data across many venues through one schema/API — the fastest path from zero to a backtest. This is the category that changed the game post-2020.

| Vendor | Depth | Coverage | History | Free? | API / format | Link |
|---|---|---|---|---|---|---|
| **Databento** | L1/L2/L3 (MBO/MBP), trades, OHLCV, statistics, definitions | 60+ venues: CME/Globex, Nasdaq, NYSE, Cboe, OPRA, IEX, Eurex, ICE; futures, options, equities, options-on-futures | 15+ yrs | **$125 free credits** (6-mo expiry), then usage-based per GB or flat-rate Std/Plus/Unlimited | REST + WebSocket; Python/C++/Rust/Go; **DBN binary, CSV, JSON, Parquet**; batch to S3/FTP | https://databento.com/ · [pricing](https://databento.com/pricing) · [schemas](https://databento.com/docs/schemas-and-data-formats) |
| **dxFeed** | Tick (complete) + aggregated, L1/depth-of-market, options analytics (Greeks/theo) | CME, Nasdaq, NYSE, Cboe, OPRA, Eurex, BIST; equities, futures, options, FX, crypto, indices | Deep (cloud archive) | Paid; custom packages | API + on-demand download; ms-stamped tick storage | https://dxfeed.com/ · [historical](https://dxfeed.com/data-services/historical-data-services/) · [CME feed](https://dxfeed.com/market-data/futures/cme/) |
| **Polygon.io** (now **Massive**) | Tick trades + quotes (nanosecond), minute/day aggregates | US stocks, options, FX, crypto — all major US exchanges + darkpools + SIP | ~20 yrs (stocks) | Free tier (delayed/limited); paid unlocks full + **Flat Files** | REST/WS + **Flat Files** (gzip CSV via S3 `files.massive.com`, bucket `flatfiles`) | https://polygon.io/ · [flat files](https://polygon.io/docs/flat-files/stocks/trades) — *rebranded Massive Oct 2025; legacy api.polygon.io still works* |

Databento is the reference implementation: raw PCAPs normalized into 15 schemas, identical fields across venues, pay-per-GB so a single-name backtest costs cents not five figures. For US equities+options under one roof with cheap bulk dumps, Polygon/Massive Flat Files is the pragmatic choice.

## Classic clean-history specialists (research-ready, survivorship-aware)

The older guard. Their pitch is *cleaned, corrected, survivorship-bias-free, point-in-time* history — the parts free APIs silently get wrong.

| Vendor | Depth | Coverage / history | Free? | Notes | Link |
|---|---|---|---|---|---|
| **AlgoSeek** | Tick TAQ / TANQ / TTOB; NBBO; minute bars | US equities since **1998 (~27,500 securities, survivorship-bias-free, SIP CTA/UTP)**; options (OPRA TAQ + NBBO subset, OI, condition codes); futures + futures options (ms-stamp, aggressor flags) | Paid | Also the data source behind **QuantConnect**; crypto added | https://algoseek.com/ · [datasets](https://algoseek.com/data-sets/list) · [QC mirror](https://www.quantconnect.com/data/algoseek-us-equities) |
| **Tick Data, LLC** (OneMarketData subsidiary) | Tick-by-tick trades + quotes; pre-built 1-min bars | US equities since **1993** (NYSE/AMEX/Nasdaq + all CTA); 180+ futures back to **1974**; US options (OPRA) since **Nov-2020**; 2,000+ FX pairs since **2008**; 60+ cash indices since **1983** | Paid | Reputation for cleanliness; global coverage | https://www.tickdata.com/ |
| **FirstRate Data** | Tick (trade+quote since 2010), 1/5/30-min, 1-h, daily | **16,231 stock tickers (incl. 7,000+ delisted) since 2000**; 130 futures since 2007; 76 crypto (14 exchanges); FX, options | Paid (cheap) + **free samples** | Per-ticker pricing (~$99/yr updates), bundles $59–99/mo; great value for individuals | https://firstratedata.com/ · [tick](https://firstratedata.com/tick-data) · [free](https://firstratedata.com/free-tick-data) |
| **Kibot** | Tick (NBBO since 2009), 1-min since 1998, daily since 1962 | 18,000+ US stocks, 6,900+ ETFs, 1,160+ FX, 83 futures | Paid + **free samples** (IBM/OIH 1-min; IVE/WDC tick; all-stock EOD) | On-demand HTTP API by symbol/date | https://www.kibot.com/ · [free](https://www.kibot.com/free_historical_data.aspx) |

Honesty note: "survivorship-bias-free" and "point-in-time" are the load-bearing claims here — free APIs (yfinance etc.) drop delisted names and silently restate, which inflates backtests. AlgoSeek/Tick Data/FirstRate explicitly retain delisted tickers; verify the *as-of* corporate-action handling before trusting a vendor's split/dividend adjustments.

## Institutional tick archives & databases (enterprise)

| Vendor / product | Depth | Coverage / history | Access | Link |
|---|---|---|---|---|
| **LSEG Tick History** (ex-Refinitiv **TRTH**) | L1 + L2 market depth; **TH-PCAP** (nanosecond, GPS-synced) | **30+ yrs, back to Jan-1996; 580+ venues; 80+ PB / 100T+ rows**; OTC + listed | REST API; normalized/raw/**Parquet**/CSV/delta; on-prem, AWS/GCP, Databricks Marketplace | https://www.lseg.com/en/data-analytics/market-data/data-feeds/tick-history · [REST API](https://developers.lseg.com/en/api-catalog/refinitiv-tick-history/refinitiv-tick-history-rth-rest-api) |
| **OneTick** (OneMarketData → **merged into KX, Sept 2025**) | Full tick DB + streaming analytics + surveillance | Multi-asset; used by banks, exchanges, hedge funds, market makers | Proprietary time-series DB engine + OneTick Cloud | https://kx.com/products/onetick-market-data/ · [merger](https://kx.com/news-room/kx-and-onetick-merge-to-unite-capital-markets-data-analytics-ai-and-surveillance-on-one-platform/) |
| **Bloomberg / FactSet** | Tick via B-PIPE / data feeds | Global, all asset classes | Terminal + enterprise feeds (paid, expensive) | (see [`README.md`](./README.md)) |

## Direct from the exchange (authoritative, you do the parsing)

Cheaper per-byte than redistributors but you parse the binary spec yourself and pay exchange license fees.

| Source | Product | Depth | History / format | Free? | Link |
|---|---|---|---|---|---|
| **CME Group** | **CME DataMine** | MBO FIX (L3, since **2017**), Market Depth (L2), Top-of-Book BBO, Time & Sales, Packet Capture, EOD, block trades, OI | Subscription; API / SFTP / **auto-S3** / browser; packages by data type | Paid | https://www.cmegroup.com/market-data/datamine-historical-data.html · [datasets wiki](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457088742/CME+DataMine+Datasets) |
| **Nasdaq** | **TotalView-ITCH 5.0** | **L3/MBO** — every add/modify/cancel/execute; nanoseconds since midnight | Direct feed (SoupBinTCP/MoldUDP64); historical files (T+1 SFTP, archive back to ~2014, ≈18 TB) via Nasdaq Data Link `NTV` & TradingPhysics | Paid | https://data.nasdaq.com/databases/NTV · [spec PDF](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf) |
| **NYSE** | **Daily TAQ** (via WRDS) | All trades + NBBO quotes, US National Market System | **Daily TAQ: µs-precision since 10-Sep-2003**; 10,000+ issues, 16 exchanges; **Monthly TAQ sec-stamp 1993–2014** | Paid (WRDS institutional) | https://www.nyse.com/market-data/historical/daily-taq · [WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/) |
| **IEX** | **DEEP / TOPS via HIST** | DEEP = L2 aggregated depth; TOPS = L1 top-of-book + last sale | **FREE, T+1, trailing 12 mo on site; full archive ≈17 TB**; PCAP (IEX-TP) | **Free** | https://iextrading.com/trading/market-data/ · [parsers](https://github.com/WojciechZankowski/iextrading4j-hist) |

IEX HIST is the best *free* real-exchange L2/PCAP dataset on the planet — no registration, full nanosecond PCAP, parseable with open-source tools ([IEXTools](https://pypi.org/project/IEXTools/)). Caveat: IEX is one venue (~2–3% of US volume), so it is not NBBO/consolidated — great for learning ITCH-style parsing and order-book reconstruction, not for execution research.

## FX tick data — free & cheap

| Source | Depth | Coverage / history | Free? | API / tool | Link |
|---|---|---|---|---|---|
| **Dukascopy** | Tick bid/ask (Swiss bank feed) | 1,000s of FX + CFDs/metals/indices; deep history | **Free** | Web export; `duka`, `dukascopy-node`, `dukascopy-python` | https://www.dukascopy.com/swiss/english/marketwatch/historical/ · [dukascopy-node](https://github.com/Leo4815162342/dukascopy-node) |
| **TrueFX** | Tick-by-tick, ms-stamp, fractional-pip, multi-bank | **15 major pairs since 2009** | **Free** (free registration) | CSV downloads | https://www.truefx.com/truefx-historical-downloads/ |
| **HistData.com** | M1 bars + tick (1-sec) bid/ask | Major + minor FX pairs, by pair/year/month | **Free** | CSV (MT4/MT5/NinjaTrader ready); [FX-1-Minute-Data](https://github.com/philipperemy/FX-1-Minute-Data) | https://www.histdata.com/download-free-forex-data/ |

FX is OTC, so there is no single consolidated tape — Dukascopy/TrueFX prices are *one liquidity provider's* aggregated quotes, not a market-wide NBBO. Fine for research, but spreads/fills will differ from your broker.

## Crypto tick & order-book vendors

Crypto is the one asset class where deep L2/L3 tick history is cheap-to-free because exchanges expose public WebSockets that vendors archive. Builds on CCXT/Kaiko/Amberdata already noted in [`README.md`](./README.md).

| Vendor | Depth | Coverage / history | Free? | API / format | Link |
|---|---|---|---|---|---|
| **Tardis.dev** | L2 incremental book updates, book snapshots (top 25/5), trades, quotes, options chains, OI, funding, liquidations | **50+ exchanges (Binance, Deribit, OKX, Bybit, Coinbase…) since 2019-03-30** | Paid (Solo/Academic/Pro/Business); some free samples | Replay API (exchange-native) + CSV; Python/Node clients; `tardis-machine` (self-host) | https://tardis.dev/ · [docs](https://docs.tardis.dev/) |
| **Kaiko** | L1 (trades) + L2 (order books), tick & snapshots, indices, DeFi | **100+ CEX/DEX, 5+ yrs**; institutional/regulatory grade | Paid | REST + WS + CSV + Snowflake | https://www.kaiko.com/products/l1-l2-data |
| **CoinDesk Data** (ex-**CryptoCompare/CCData**) | Tick trades, OHLCV, **L2 snapshots, nanosecond stamps, CCSEQ gapless sequence** | **300+ exchanges, 300k+ pairs, 10+ yrs tick (Enterprise)** | Paid only (free API tier **retired 21-May-2026**); FCA-authorized | REST/WS API | https://developers.coindesk.com/ |
| **CoinAPI** | Real-time + historical **L2/L3**, replayable archives, FIX streaming | Hundreds of exchanges (native L3 for Bitso/Coinbase; FIX-linked for more); flat files via S3/Snowflake | Free tier + paid | REST/WS/FIX + Flat Files | https://www.coinapi.io/ |
| **Crypto Lake** | Tick trades + **20-level book snapshots**, OHLCV, OI, funding | 10 exchanges, top tokens + alts | Paid + **free-data tier** | **Parquet on S3** (partitioned exchange/symbol/day); `lake-api` Python | https://crypto-lake.com/ · [lake-api](https://github.com/crypto-lake/lake-api) |
| **CryptoTick** (CoinAPI) / **CryptoDataDownload** | Flat-file trades/quotes/book/OHLCV / free OHLCV CSVs | Hundreds of exchanges / 20+ exchanges, 1,500+ instruments | Paid / **Free (OHLCV, no registration)** | S3/Snowflake gzip CSV / direct CSV | https://www.coinapi.io/products/flat-files · https://www.cryptodatadownload.com/ |

Crypto Lake is the quant-friendliest format (native Parquet, S3-partitioned, parallel Python loader). Tardis.dev is the depth/coverage leader. For free OHLCV, CryptoDataDownload; for free L2 raw, replay exchange WebSockets yourself with CCXT Pro.

## Brazil 🇧🇷 — B3 tick & market-data routes

There is **no free consolidated B3 tick tape**. Real-time/tick access goes through B3's authorized-vendor model; for backtesting most retail quants reconstruct ticks from a broker terminal (MetaTrader5 / ProfitDLL) — see [`Brazil_B3_Market_Data_APIs.md`](./Brazil_B3_Market_Data_APIs.md).

| Source | Content | Depth | Free/Paid | How | Link |
|---|---|---|---|---|---|
| **B3 Market Data Platform** (PUMA/UMDF) | Official real-time feed via authorized vendors | L1/L2; FIX/FAST UMDF, UMDF Conflated, Binary UMDF | Paid (vendor license) | Distributed only via authorized vendors/sub-vendors | https://www.b3.com.br/en_us/market-data-and-indices/data-services/market-data/market-data-platform/ |
| **B3 UP2DATA** | Official **EOD** distribution (equities, DI futures, swaps, vol surfaces, tick-by-tick fixed income, corp actions) | EOD + FI tick | Paid (tiered) | Client software / cloud; TXT/CSV/XML/JSON | [about](https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/about-up2data/) |
| **Cedro Technologies** (Market Data Cloud) | Leading B3 redistributor; tick-by-tick service | L1/L2 + **tick-by-tick** (ASSET\|TIME\|PRICE\|QTY\|BUY/SELL BROKER) | Paid (7-day free trial) | REST/JSON, WebSocket, Socket; daily FTP file for tick | https://www.marketdatacloud.com.br/ · [tick API](https://www.marketdatacloud.com.br/api-tick-by-tick/) |
| **Nelogica** (ProfitChart / DataSolution / DLL) | Authorized B3 vendor; ProfitChart feed | Tick (Times & Trades), intraday, EOD | Paid platform | RTD/DDE to Excel; **DLL Real Time** (Python/C#/C++/Delphi); CSV export (daily/intraday/tick) | https://store.nelogica.com.br/data-solution |
| **B3 free reports** | COTAHIST / séries históricas, daily market-data reports | EOD only | **Free** | Direct download | (B3 site) |

Tip: For WIN/WDO intraday research without a vendor contract, `MetaTrader5` Python (broker terminal) or Nelogica **ProfitDLL** are the standard retail routes — history depth depends on the broker, timestamps are broker-timezone, and 1-min bars often cover only ~1 year.

## Storage & format — what the pros actually use

Clean tick data is large (a single US-equities trading day of full OPRA TAQ is tens of GB). The storage choice is part of the dataset decision.

| Format / engine | Why it's used for ticks | Notes |
|---|---|---|
| **kdb+/q** (KX) | Column-store time-series DB; STAC-M3 benchmark record-holder; the HFT/bank standard | `kdb+ tick` architecture = Tickerplant → RDB (in-mem) → HDB (splayed/partitioned). Now natively reads **Parquet**; Fusion connectors to Arrow/Avro/HDF5. https://kx.com/products/kdb/ |
| **Apache Parquet** | Open columnar; cheap on object storage; default for Databento/LSEG/Crypto Lake batch | Query with DuckDB/Trino/Spark/Polars; pay compute only on query |
| **Apache Arrow** | In-memory columnar interchange | Zero-copy between Python/C++/Rust |
| **ArcticDB** (Man Group, source-available BSL 1.1) | Pandas-native time-series store over S3; built for tick/research at scale | Versioned, bitemporal; successor to Arctic; https://github.com/man-group/ArcticDB |
| **ClickHouse / QuestDB** | Open columnar OLAP/TSDB; popular cheaper kdb+ alternatives for tick | QuestDB ingests Databento live feeds directly |

## Buyer's caveats (read before you wire money)

- **Cost reality**: full historical L3/MBO for a broad universe is a five-to-six-figure annual spend (exchange license + vendor fee). Scope to the symbols/dates you actually need — Databento/Polygon per-GB/flat-file models exist precisely to avoid the big-bang buy.
- **Look-ahead / point-in-time**: confirm corporate-action and symbol-mapping are *as-of*, not restated. Restated splits/dividends and silently re-symboled tickers create look-ahead bias.
- **Survivorship**: insist on delisted-name retention (AlgoSeek/Tick Data/FirstRate keep them; most free APIs do not).
- **Consolidated vs single-venue**: NBBO/SIP (TAQ, AlgoSeek) ≠ a single direct feed (IEX, one ITCH venue). FX/crypto have **no** official consolidated tape — every "best bid/ask" is one source's view.
- **Clock & latency**: PCAP with GPS/nanosecond stamps (LSEG TH-PCAP, CME PCAP, Databento raw) matters for latency-sensitive research; normalized feeds may carry vendor-side timestamp jitter.
- **Licensing/redistribution**: exchange data licenses restrict redistribution and often per-user display fees — non-display/research use is a separate (cheaper) license tier. Read the agreement.

**Sources:** [Databento](https://databento.com/) · [Databento pricing](https://databento.com/pricing) · [Databento schemas](https://databento.com/docs/schemas-and-data-formats) · [dxFeed](https://dxfeed.com/) · [dxFeed historical](https://dxfeed.com/data-services/historical-data-services/) · [Polygon/Massive flat files](https://polygon.io/docs/flat-files/stocks/trades) · [AlgoSeek](https://algoseek.com/) · [AlgoSeek datasets](https://algoseek.com/data-sets/list) · [Tick Data LLC](https://www.tickdata.com/) · [FirstRate Data](https://firstratedata.com/) · [Kibot](https://www.kibot.com/) · [LSEG Tick History](https://www.lseg.com/en/data-analytics/market-data/data-feeds/tick-history) · [KX/OneTick merger](https://kx.com/news-room/kx-and-onetick-merge-to-unite-capital-markets-data-analytics-ai-and-surveillance-on-one-platform/) · [CME DataMine](https://www.cmegroup.com/market-data/datamine-historical-data.html) · [Nasdaq TotalView-ITCH](https://data.nasdaq.com/databases/NTV) · [ITCH 5.0 spec](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf) · [NYSE Daily TAQ](https://www.nyse.com/market-data/historical/daily-taq) · [WRDS TAQ](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/) · [IEX market data](https://iextrading.com/trading/market-data/) · [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) · [TrueFX](https://www.truefx.com/truefx-historical-downloads/) · [HistData](https://www.histdata.com/download-free-forex-data/) · [Tardis.dev](https://tardis.dev/) · [Kaiko L1/L2](https://www.kaiko.com/products/l1-l2-data) · [CoinDesk Data](https://developers.coindesk.com/) · [CoinAPI](https://www.coinapi.io/) · [Crypto Lake](https://crypto-lake.com/) · [CryptoDataDownload](https://www.cryptodatadownload.com/) · [B3 Market Data Platform](https://www.b3.com.br/en_us/market-data-and-indices/data-services/market-data/market-data-platform/) · [Cedro Market Data Cloud](https://www.marketdatacloud.com.br/) · [Nelogica DataSolution](https://store.nelogica.com.br/data-solution) · [kdb+](https://kx.com/products/kdb/) · [ArcticDB](https://github.com/man-group/ArcticDB)

**Keywords:** tick data, high-frequency data, HFT, market microstructure, limit order book, LOB, L1 L2 L3, MBO market-by-order, MBP market-by-price, BBO NBBO, time and sales, PCAP packet capture, Databento, dxFeed, Polygon Massive flat files, AlgoSeek, Tick Data LLC, FirstRate Data, Kibot, LSEG Tick History TRTH Refinitiv, OneTick KX kdb+, CME DataMine, Nasdaq TotalView ITCH, NYSE Daily TAQ WRDS, IEX DEEP TOPS HIST, LOBSTER, Dukascopy, TrueFX, HistData, Tardis.dev, Kaiko, CoinDesk Data CryptoCompare, CoinAPI, Crypto Lake, Parquet, Arrow, ArcticDB, ClickHouse QuestDB, survivorship bias, point-in-time, look-ahead, consolidated tape, B3 UP2DATA, Cedro Market Data Cloud, Nelogica ProfitChart ProfitDLL — dados tick a tick, alta frequência, microestrutura de mercado, livro de ofertas, profundidade de mercado, negócios e ofertas, viés de sobrevivência, dados ponto-no-tempo, redistribuidor de market data, dados históricos intradiários, feed em tempo real B3, Cedro, Nelogica
