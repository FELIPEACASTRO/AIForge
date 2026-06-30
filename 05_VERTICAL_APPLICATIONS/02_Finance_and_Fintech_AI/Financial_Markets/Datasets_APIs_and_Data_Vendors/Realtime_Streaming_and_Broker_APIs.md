# Real-Time Streaming & Broker Data APIs

> The live-data layer: WebSocket/streaming market-data feeds and broker/execution APIs that push ticks, quotes, order books and fills in real time — the plumbing for live ML inference, paper trading, and automated execution. US + global + Brazil 🇧🇷. Honest on latency, entitlements, paper sandboxes, rate limits, and the difference between *broker data* (free with an account, license-bound) and *vendor data* (paid, redistributable).

This page is about **getting data while the market is open** — and, where the same API also routes orders, about the broker/execution side. It is deliberately distinct from the bulk/historical vendors in [`Tick_and_HFT_Data_Vendors.md`](./Tick_and_HFT_Data_Vendors.md) and the regional REST catalogs ([`Brazil_B3_Market_Data_APIs.md`](./Brazil_B3_Market_Data_APIs.md), [`Asia_and_India_Market_Data_APIs.md`](./Asia_and_India_Market_Data_APIs.md)). REST polling (yfinance, FRED, basic Polygon REST, Alpha Vantage) is covered elsewhere — here the focus is **push**: WebSocket, SSE, FIX-lite, raw TCP, and DLL/socket bridges.

**Two mental models, get them straight:**
- **Broker-fed data** (Alpaca, IBKR, Schwab, Tradier, Tastytrade, MT5, Cedro-via-broker, Nelogica) — usually *free or cheap with a funded account*, but **entitlement-locked** (you must sign exchange agreements; redistribution is forbidden; non-professional vs professional matters for fees). Great for live trading, not for building a product you resell.
- **Vendor data** (Polygon/Massive, Databento Live, Finnhub, Twelve Data, dxFeed) — *paid, licensable, redistributable on higher tiers*, no brokerage account needed. Use when the data **is** the product.

---

## 1. Streaming market-data vendors (WebSocket / SSE / raw socket)

No brokerage account required. These sell or give away a live push feed you consume directly.

| Provider | Data (streaming) | Coverage | Latency / transport | Free? | Auth & how | Link |
|---|---|---|---|---|---|---|
| **Polygon.io** (rebranded **Massive**, Oct 2025) | Trades, quotes, aggregates (per-second/minute), options, indices, FX, crypto | US stocks/options/indices + global FX & crypto | WebSocket (real-time on paid; delayed on lower tiers) | Free tier (delayed/limited); real-time + WS on higher plans (Starter ~$29/mo delayed-ish, real-time tiers ~$199/mo) | API key; `wss://socket.polygon.io/{stocks\|options\|crypto\|forex}` (legacy host still works) | [polygon.io](https://polygon.io/) · [WS docs](https://massive.com/docs/websocket/crypto/trades) · [pricing](https://massive.com/pricing) |
| **Databento Live** | MBO/MBP-1/MBP-10 (L3/L2), trades, BBO/TBBO, OHLCV, statistics, definitions — many schemas | 60+ licensed venues: CME Globex, Nasdaq, NYSE, Cboe, OPRA, IEX, Eurex, ICE | **Raw TCP socket** (lower overhead than WS/SSE); PTP nanosecond stamps; client-side normalization | $125 sign-up credits (historical only); **Live billed per-message + pass-through exchange license fees** (no flat monthly base) — CME license from ~$32.65/mo (non-pro) | API key; Python/C++/Rust client wraps raw API | [databento.com/live](https://databento.com/live) · [API ref](https://databento.com/docs/api-reference-live) · [pricing](https://databento.com/pricing) |
| **Finnhub** | Real-time trades (`trade` channel), news | US stocks, FX, crypto (real-time US; intl on paid) | WebSocket `wss://ws.finnhub.io` | **Free: 60 req/min, WS up to 50 symbols**; paid lifts symbol cap | API token query param | [finnhub.io](https://finnhub.io/) · [rate limits](https://finnhub.io/docs/api/rate-limit) · [pricing](https://finnhub.io/pricing) |
| **Twelve Data** | Real-time price stream (1 WS credit per subscribed symbol) | 50+ countries, stocks/ETF/FX/crypto/commodities | WebSocket `wss://ws.twelvedata.com/v1/quotes/price` | Trial: 8 WS credits + 800 REST/day; paid plans: Grow $79/mo, Pro $229/mo, Ultra $999/mo (higher WS-credit caps on each tier) | API key | [twelvedata.com](https://twelvedata.com/) · [docs](https://twelvedata.com/docs) · [pricing](https://twelvedata.com/pricing) |
| **dxFeed** | L1/depth-of-market tick + options analytics (Greeks/theo/IV) | CME, Nasdaq, NYSE, Cboe, OPRA, Eurex, BIST; equities/futures/options/FX/crypto | dxLink WebSocket protocol; QD binary | Paid; custom packages (retail "dxFeed Free" sandbox exists) | API key; dxLink/QD | [dxfeed.com](https://dxfeed.com/) |
| **Alpaca Market Data** | Trades, quotes, bars (stocks); trades/quotes/orderbook/bars (crypto); trades/quotes (options) | US stocks (IEX free / SIP paid), crypto, options (indicative free / OPRA paid) | WebSocket; msgpack for options | **Free: IEX stream + indicative options + crypto**; SIP/OPRA on Algo Trader Plus | API key/secret; `wss://stream.data.alpaca.markets/v2/{iex\|sip}`, `/v1beta3/crypto/us`, `/v1beta1/{opra\|indicative}` | [alpaca.markets/data](https://alpaca.markets/data) · [WS docs](https://docs.alpaca.markets/docs/streaming-market-data) |

> ⚠️ **IEX Cloud is dead.** IEX Group retired all IEX Cloud products on **31 Aug 2024** (SSE streaming included). Any tutorial pointing at `cloud.iexapis.com` is stale — migrate to Polygon/Massive, Databento, Finnhub, FMP or Alpha Vantage. ([alphavantage.co migration note](https://www.alphavantage.co/iexcloud_shutdown_analysis_and_migration/), [Massive migration guide](https://massive.com/blog/iex-cloud-migration-guide))

**Why transport matters:** SSE (the old IEX model) and WebSocket carry framing/text overhead; Databento's raw-TCP design and dxFeed's binary QD are measurably lower-latency. For ML inference that tolerates 100–500 ms, WebSocket is fine; for execution-sensitive microstructure, the socket/binary feeds (or colocation) win.

---

## 2. US broker / execution APIs (data + order routing)

These give live data **and** trading. Most have a **paper/sandbox** environment — invaluable for testing ML strategies without capital. Market-data entitlements (exchange agreements, pro vs non-pro) gate what you can see.

| Broker | Streaming data | Paper/sandbox | Auth | Rate limits / notes | Link |
|---|---|---|---|---|---|
| **Interactive Brokers** | L1 top-of-book, L2 deep book, tick-by-tick, real-time bars — via **TWS API** (socket to TWS/IB Gateway) or **Client Portal Web API** (REST + WebSocket) | Paper account (full API) | TWS/Gateway login; CP Web API OAuth/session | ~**50 msg/s** API pacing; 100 concurrent market-data lines min (more by commissions/equity); 100 free snapshots/mo; **per-exchange market-data subscriptions required** for real-time | [TWS API](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/) · [Web API](https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1/) · [market-data pricing](https://www.interactivebrokers.com/en/pricing/market-data-pricing.php) |
| **Alpaca** | See §1 (same WS feeds); trade updates stream | **Paper trading** (`paper-api.alpaca.markets`) | API key/secret | REST 200 req/min (free); commission-free US equities/options/crypto | [alpaca.markets](https://alpaca.markets/) · [docs](https://docs.alpaca.markets/) |
| **Charles Schwab (Trader API)** | WebSocket **Streamer**: `LEVELONE_EQUITIES/OPTIONS/FUTURES/FOREX`, `NYSE_BOOK`, `NASDAQ_BOOK`, `OPTIONS_BOOK`, time-of-sale, chart | No public sandbox (live account) | **OAuth 2.0**, app registered at developer.schwab.com + linked brokerage acct | **Replaces the retired TD Ameritrade API (shut 10 May 2024)**; quotes, chains, history | [developer.schwab.com](https://developer.schwab.com/) |
| **Tradier** | WebSocket `wss://ws.tradier.com/v1/markets/events` (trade/quote/summary/timesale/tradex); also HTTP streaming | **Sandbox** (paper, 15-min delayed) | Bearer token (separate live/sandbox); short-lived `sessionid` (5-min) to open stream | One market + one account stream at a time; API free for account holders | [docs.tradier.com](https://docs.tradier.com/) · [streaming](https://documentation.tradier.com/brokerage-api/overview/streaming) |
| **Tastytrade** | Real-time quotes/Greeks/candles via **DXLink** (dxFeed) WebSocket — fetch quote-token from API, then connect to dxLink streamer | **Sandbox** (developer.tastyworks.com) | OAuth/session; DXLink token | Options-focused; great Greeks/IV stream | [developer.tastytrade.com](https://developer.tastytrade.com/streaming-market-data/) |
| **Tradovate** (futures) | WebSocket `wss://live.tradovateapi.com/v1/websocket` — quotes, DOM/L2, fills | **Demo** `demo.tradovateapi.com` (2-week full API) | Bearer token (`accesstokenrequest`) | CME non-pro data ~$12/mo L1, ~$41/mo L2 (platform); **API/WS market data needs a CME sub-vendor ILA + a separate API add-on subscription**; REST+WS framed protocol | [api.tradovate.com](https://api.tradovate.com/) |
| **E*TRADE** | Streaming **order-status** push (Comet-based); quotes are REST (real-time after signing Market Data Agreement) | **Sandbox** (no real fills) | **OAuth 1.0a**; Market Data Agreement required | No true streaming *quote* feed — REST quotes w/ 5 field-sets | [developer.etrade.com](https://developer.etrade.com/home) |

### Unofficial / reverse-engineered (use with eyes open)
| Library | Broker | Status | Caveat | Link |
|---|---|---|---|---|
| **robin_stocks** / **pyrh** | Robinhood | Active community libs | Robinhood has **no official equities/options API** (only an official *Crypto Trading API*); ToS breach risk; endpoints break without notice | [robin_stocks](https://github.com/jmfernandes/robin_stocks) · [pyrh](https://github.com/robinhood-unofficial/pyrh) · [official crypto API](https://docs.robinhood.com/crypto/trading/) |
| **webull / unofficial-webull** | Webull | Community | Webull now ships an **official OpenAPI** (US market: stocks/options/futures/crypto, real-time data via MQTT streaming, individual + institutional accounts); the python `webull` lib is unofficial | [official OpenAPI docs](https://developer.webull.com/apis/docs/) · community libs on PyPI/GitHub |

> 🪦 **TD Ameritrade API is gone.** Endpoints shut after close **10 May 2024**; the `tda-api` library now targets Schwab. Migrate to the [Schwab Trader API](https://developer.schwab.com/) (`schwabdev`, `schwab-py`). ([tda-api transition](https://tda-api.readthedocs.io/en/latest/schwab.html))

**Python client libraries (battle-tested, 2024-2026):**
- `ib_async` — **the maintained successor to `ib_insync`** (original is on a legacy TWS API, no longer updated after Ewald de Wit's passing). Sync/async, event-driven. [ib-api-reloaded/ib_async](https://github.com/ib-api-reloaded/ib_async)
- `schwabdev` (Tyler Bowers) and `schwab-py` (Alex Golec) — auto token refresh, full Streamer support. [schwabdev](https://pypi.org/project/schwabdev/) · [schwab-py](https://schwab-py.readthedocs.io/)
- `tastytrade` (tastyware) — typed async SDK incl. DXLinkStreamer. [tastyware/tastytrade](https://github.com/tastyware/tastytrade)
- `alpaca-py` (official) — replaces the deprecated `alpaca-trade-api`. [docs](https://docs.alpaca.markets/)

---

## 3. FX / forex broker APIs

| Provider | Streaming | Practice/demo | Auth | Notes | Link |
|---|---|---|---|---|---|
| **OANDA v20 REST** | **PricingStream** endpoint streams live bid/ask; transaction stream | **Free practice (demo) account** | Personal access token (AMP → Manage API Access) | Clean REST+streaming, widely used for FX ML; `oandapyV20` python wrapper | [developer.oanda.com](https://developer.oanda.com/rest-live-v20/introduction/) |
| **FXCM** | **ForexConnect API** (streaming prices, all order types; C++/C#/Java/.NET/Python) + FIX; legacy REST/`fxcmpy` | Demo account | API token | ForexConnect is the supported path; legacy O2G2/REST partly deprecated — confirm current status | [ForexConnect](https://github.com/fxcm/ForexConnectAPI) · [API trading](https://www.fxcm.com/markets/algorithmic-trading/api-trading/) |
| **MetaTrader 4/5** | See §4 — broker-agnostic, covers FX + (in 🇧🇷) B3 | Broker demo | Broker login in terminal | Largest retail FX install base globally | [MQL5 docs](https://www.mql5.com/en/docs/python_metatrader5) |

---

## 4. MetaTrader 4/5 — the universal retail bridge (FX + 🇧🇷 B3)

MetaTrader is the most common live-data path for retail quants worldwide, and **the single most important one for Brazil** because XP, Clear, Rico, Genial, Modal, Órama and others offer MT5 with the B3 feed.

| Path | What | Latency/transport | OS | Notes | Link |
|---|---|---|---|---|---|
| **`MetaTrader5` (PyPI)** | Official Python terminal API — `copy_rates_*` (OHLCV bars), `copy_ticks_*` (bid/ask/last ticks), live `symbol_info_tick`, order send | Local IPC to a **running, logged-in MT5 terminal** | **Windows only** (win_amd64 wheels) | History depth = broker-dependent; timestamps in **broker timezone**; the de-facto WIN/WDO intraday route in 🇧🇷 | [PyPI](https://pypi.org/project/MetaTrader5/) · [copy_ticks_from](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksfrom_py) |
| **MQL4 / MQL5 (native EA)** | Expert Advisors run *inside* the terminal; `OnTick()` event-driven; can bridge to Python via socket/file/ZeroMQ | In-process, lowest latency for MT | Cross (terminal) | For true co-resident strategies; ZeroMQ bridges popular | [mql5.com](https://www.mql5.com/en/docs) |
| **MetaApi (cloud)** | Cloud gateway exposing **MT4 & MT5 over REST + WebSocket** — no local Windows terminal needed; CopyFactory copy-trading | Cloud WS; managed | Any (cloud) | Free tier; `metaapi-cloud-sdk` (Python/JS) | [metaapi.cloud](https://metaapi.cloud/) · [PyPI](https://pypi.org/project/metaapi-cloud-sdk/) |
| **PyTrader connector** | Drag-n-drop EA ↔ Python connector for MT4/MT5 | Socket | Cross | Open-source live+paper bridge | [GitHub](https://github.com/TheSnowGuru/PyTrader-python-mt4-mt5-trading-api-connector-drag-n-drop) |

```python
# MetaTrader5: live tick + 1-min bars (terminal must be open & logged in)
import MetaTrader5 as mt5
mt5.initialize()
tick = mt5.symbol_info_tick("WINQ25")          # bid/ask/last for a B3 mini-index future
bars = mt5.copy_rates_from_pos("PETR4", mt5.TIMEFRAME_M1, 0, 500)  # last 500 1-min bars
mt5.shutdown()
```

---

## 5. Crypto exchange WebSockets (free, no account for market data)

Crypto venues publish *public* market-data WS feeds **without authentication** — the easiest real-time data on Earth for ML prototyping. (Trading channels still need API keys.)

| Exchange | Public WS endpoint | Channels | Limits | Link |
|---|---|---|---|---|
| **Binance** | `wss://stream.binance.com:9443` (spot); `wss://fstream.binance.com` (USDⓈ-M futures) | aggTrade, trade, depth (diff/partial L2), kline, bookTicker, markPrice | Spot ≤5 msg/s in, ≤1024 streams/conn, 300 conn/5min/IP; futures ≤10 msg/s; ping every 20s (spot)/3min (futures) | [spot WS](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams) · [futures WS](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams) |
| **Coinbase** (Advanced Trade) | `wss://advanced-trade-ws.coinbase.com` | `level2`, `ticker`, `ticker_batch`, `market_trades`, `candles`, `status` | Most channels now **no-auth**; auth (CDP JWT, 2-min expiry) recommended for reliability; one channel per subscribe msg | [WS overview](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-overview) |
| **Kraken** | `wss://ws.kraken.com/v2` | `ticker` (L1), `book` (L2, depths 10/25/100/500/1000 w/ CRC32 checksum), `level3` (MBO, auth), `trade`, `ohlc` | CRC32 book checksum for integrity; FIX for institutional | [WS v2 book](https://docs.kraken.com/api/docs/websocket-v2/book/) · [feeds blog](https://blog.kraken.com/product/api/unlocked-3-the-market-data-feeds-systematic-traders-use) |

> Crypto derivatives venues (Deribit, Bybit, OKX, BitMEX) are covered in the crypto-derivatives material; their WS schemas mirror the above (orderbook/trades/funding/mark-price channels). For *normalized* multi-exchange crypto WS in one schema, see open-source `cryptofeed` and `ccxt.pro`.

---

## 6. 🇧🇷 Brazil — B3 real-time & broker feeds

Brazil's live-data market is dominated by **redistributors** (Cedro, Nelogica) who buy the B3 feed and resell it, plus broker terminals. There is **no free real-time B3 WebSocket** equivalent to crypto — real-time B3 always costs money and/or requires signing B3's market-data terms (professional vs non-professional, *Sinacor*/cadastro). Free community APIs are **delayed or snapshot REST**, not streaming.

| Source | Streaming? | Data | Real-time? | How / auth | Link |
|---|---|---|---|---|---|
| **Cedro Technologies** (Market Data Cloud) | ✅ **WebSocket, Socket (TCP/UDP), REST** | Quotes, order book (L2), trades/times-and-sales, FX, futures (WIN/WDO/DOL/IND) | ✅ Real-time (B3 redistributor) — also delayed tier | API credentials; XML/JSON; "Crystal Data Feed" / "Web Feeder" | [marketdatacloud.com.br](https://www.marketdatacloud.com.br/) · [cedrotech.com](https://www.cedrotech.com/) |
| **Nelogica ProfitDLL / DLL Real Time** | ✅ Callback streaming via DLL | L1+L2 book, time & trades, tick-by-tick; **90 days tick history** | ✅ Real-time (BM&F + Bovespa) | `ProfitDLL.dll` bound from Python/C#/Delphi; needs Nelogica Data Solution subscription | [ProfitDLL docs](https://ajuda.nelogica.com.br/hc/pt-br/articles/22396517026203-Ecossistema-ProfitDLL-e-primeiros-passos) · [Data Solution](https://www.nelogica.com.br/data-solution) |
| **Profit (Nelogica) DDE/RTD** | ✅ DDE/RTD into Excel | Quotes, book | ✅ (with terminal) | Legacy Excel DDE/RTD link from the Profit terminal | [nelogica.com.br](https://www.nelogica.com.br/) |
| **OpLab API** | ✅ REST + WebSocket (data via Cedro) | **Options-focused**: option chains, IV, Greeks, B3 equities/options real-time | ✅ Real-time (paid plans) | API token; options analytics | [oplab.com.br](https://oplab.com.br/) · [planos](https://oplab.com.br/planos/) |
| **MetaTrader 5 (XP/Clear/Rico/Genial/Modal)** | ✅ via `MetaTrader5` PyPI (§4) | OHLCV + ticks for B3 equities & futures | ✅ Real-time (broker feed) | Broker MT5 login; Windows terminal | [MQL5 Python](https://www.mql5.com/en/docs/python_metatrader5) |
| **B3 *for Developers* / UP2DATA** | ❌ (file/Cloud, not streaming) | EOD + reference data; market-data APIs catalog | ❌ **D0 end-of-day**, not intraday stream | SAS keys via UP2DATA API; CSV/TXT/JSON/XML | [B3 for Developers](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/b3-for-developers/) · [developers.b3.com.br](https://developers.b3.com.br/) · [UP2DATA](https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/about-up2data/) |
| **brapi.dev** | ❌ REST only (**no WebSocket**) | Quotes, dividends, fundamentals, crypto, FX | ⚠️ Free ~30-min delay; Startup ~15-min, Pro ~5-min (all still polling) | Bearer token; poll periodically | [brapi.dev](https://brapi.dev/) · [pricing](https://brapi.dev/pricing) · [FAQ delay](https://brapi.dev/faq) |
| **BTG Pactual / Banco Inter (Open Finance)** | ⚠️ REST (banking/investments, not a tick feed) | Account, custody, positions, payments — **not** market-data streaming | n/a | OAuth; Open Finance Brasil standard | [BTG developers](https://developer.btgpactual.com/) · [Open Finance BR](https://openfinance.dev.br/) |

> 🇧🇷 **Reality check:** for live B3 ticks in code you realistically choose between (a) **MetaTrader5 + a broker** (cheapest, Windows, intraday WIN/WDO), (b) **Cedro Market Data Cloud** (proper WS/socket, costs money, redistributor terms), or (c) **Nelogica ProfitDLL** (deepest book + 90-day tick history, DLL integration). brapi and B3 UP2DATA are *not* streaming sources — brapi is delayed REST polling, UP2DATA is end-of-day files. Bank "investment APIs" (BTG/Inter) are for **portfolio/custody**, not market data.

---

## 7. Cross-cutting gotchas (read before you build)

- **Entitlements & professional status.** Broker feeds (IBKR, Schwab, Cedro, Nelogica) require signed exchange agreements. Declaring yourself a **professional** user multiplies fees 5–50×. Get classification right. Redistribution of broker/vendor real-time data is almost always prohibited without a separate license.
- **Paper ≠ production data.** Tradier/Alpaca sandbox and many demos serve **15-min delayed** or synthetic data and simplified fills; latency, slippage and partial fills won't match live. Validate on a small live allocation before trusting backtested edge.
- **No look-ahead in live loops.** A live WS bar is *forming* until its window closes — acting on a partial candle is a classic look-ahead bug. Key off the bar-close/`is_final` flag (Binance `k.x`, Schwab/IBKR real-time-bar events).
- **Reconnect & gap-fill.** Every WS drops. Implement heartbeat/ping handling (Binance ping cadence above), exponential-backoff reconnect, and **REST snapshot + book checksum** resync (Kraken CRC32, Binance depth-snapshot+diff) or you'll silently desync the order book.
- **Rate limits bite in messages, not just requests.** Inbound-message caps (Binance 5–10 msg/s), connection caps, and IBKR's ~50 msg/s pacing terminate sessions on abuse. Batch subscriptions.
- **Clock & timezone.** MT5 timestamps are in **broker time**; B3 is BRT/America/Sao_Paulo; crypto is UTC. Normalize to UTC with explicit tz, store nanoseconds where the feed provides them (Databento PTP, Polygon nanosecond).
- **Free real-time is regional.** US equities have cheap/free real-time (IEX via Alpaca, Finnhub). **B3 real-time is never free.** Crypto real-time is free everywhere. Budget accordingly.
- **Survivorship/point-in-time** is a *historical* concern — but if you snapshot live data to build your own history, you inherit it: log delistings, symbol changes and corporate actions as they happen or your home-grown dataset rots.

**Sources:** [Polygon/Massive](https://polygon.io/) · [Massive pricing](https://massive.com/pricing) · [Databento Live](https://databento.com/live) · [Databento pricing](https://databento.com/pricing) · [Finnhub rate limits](https://finnhub.io/docs/api/rate-limit) · [Twelve Data docs](https://twelvedata.com/docs) · [dxFeed](https://dxfeed.com/) · [Alpaca data](https://alpaca.markets/data) · [Alpaca WS docs](https://docs.alpaca.markets/docs/streaming-market-data) · [Alpaca options data](https://docs.alpaca.markets/docs/real-time-option-data) · [IBKR TWS API](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/) · [IBKR Web API](https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1/) · [IBKR market-data pricing](https://www.interactivebrokers.com/en/pricing/market-data-pricing.php) · [ib_async](https://github.com/ib-api-reloaded/ib_async) · [Schwab developer](https://developer.schwab.com/) · [schwab-py streaming](https://schwab-py.readthedocs.io/en/latest/streaming.html) · [schwabdev](https://pypi.org/project/schwabdev/) · [tda-api Schwab transition](https://tda-api.readthedocs.io/en/latest/schwab.html) · [Tradier streaming](https://documentation.tradier.com/brokerage-api/overview/streaming) · [Tradier WS market session](https://documentation.tradier.com/brokerage-api/streaming/create-market-session) · [Tastytrade streaming](https://developer.tastytrade.com/streaming-market-data/) · [tastyware/tastytrade](https://github.com/tastyware/tastytrade) · [Tradovate API](https://api.tradovate.com/) · [E*TRADE developer](https://developer.etrade.com/home) · [robin_stocks](https://github.com/jmfernandes/robin_stocks) · [Robinhood crypto API](https://docs.robinhood.com/crypto/trading/) · [Webull OpenAPI](https://developer.webull.com/apis/docs/) · [OANDA v20](https://developer.oanda.com/rest-live-v20/introduction/) · [FXCM ForexConnect](https://github.com/fxcm/ForexConnectAPI) · [MetaTrader5 PyPI](https://pypi.org/project/MetaTrader5/) · [MQL5 Python](https://www.mql5.com/en/docs/python_metatrader5) · [MetaApi](https://metaapi.cloud/) · [Binance spot WS](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams) · [Binance futures WS](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams) · [Coinbase WS](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-overview) · [Kraken WS v2](https://docs.kraken.com/api/docs/websocket-v2/book/) · [Cedro Market Data Cloud](https://www.marketdatacloud.com.br/) · [Nelogica ProfitDLL](https://ajuda.nelogica.com.br/hc/pt-br/articles/22396517026203-Ecossistema-ProfitDLL-e-primeiros-passos) · [OpLab](https://oplab.com.br/) · [B3 for Developers](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/b3-for-developers/) · [B3 UP2DATA](https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/about-up2data/) · [brapi.dev](https://brapi.dev/) · [BTG developers](https://developer.btgpactual.com/) · [IEX Cloud shutdown](https://www.alphavantage.co/iexcloud_shutdown_analysis_and_migration/)

**Keywords:** real-time market data, streaming API, WebSocket, broker API, execution API, paper trading, market data entitlements, order book, Level 2, tick data, low latency, FIX, SSE, raw socket, Polygon Massive, Databento Live, Alpaca, Interactive Brokers, IBKR, ib_async, Charles Schwab Trader API, schwabdev, Tradier, Tastytrade, DXLink, dxFeed, Tradovate, E*TRADE, Robinhood, Webull, OANDA, FXCM, ForexConnect, MetaTrader 5, MetaTrader5 Python, MetaApi, Binance WebSocket, Coinbase Advanced Trade, Kraken WebSocket v2, crypto streaming, Finnhub, Twelve Data; 🇧🇷 dados em tempo real, cotações ao vivo, streaming, book de ofertas, corretora, API de corretora, conta de teste/paper, Cedro Market Data Cloud, Crystal, Nelogica ProfitDLL, Profit, DDE/RTD, OpLab, B3 for Developers, UP2DATA, brapi, BTG Pactual, Banco Inter, Open Finance, WIN, WDO, mini-índice, mini-dólar, mercado financeiro, dados de mercado, latência, reconexão, look-ahead.
