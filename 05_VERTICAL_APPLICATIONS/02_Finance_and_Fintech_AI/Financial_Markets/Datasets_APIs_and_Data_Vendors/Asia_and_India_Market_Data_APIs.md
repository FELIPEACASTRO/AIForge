# Asia & India Market Data — APIs and Datasets

> How to actually GET Asian and Indian equity/derivatives data for ML/quant research — primary exchanges, market mechanics (settlement, price limits, lot sizes), and concrete free/paid APIs with working URLs, ticker-suffix conventions, and runnable Python. Audience is Brazil-heavy: English with Portuguese terms in parentheses, plus how Brazilians reach these markets via BDRs/ETFs.

Asia is the most fragmented data region on earth: each market has its own regulator, settlement cycle, price-limit band (*banda de oscilação*), lot convention, and ticker namespace. There is no single Bloomberg-style free firehose. This page maps the working sources market-by-market and ends with a master comparison table and code snippets.

---

## 1. The exchanges and their mechanics (2025–2026)

| Exchange | Country | Code/MIC | Trading hours (local) | Settlement | Daily price limit | Notes |
|---|---|---|---|---|---|---|
| NSE | India | NSE / XNSE | 09:15–15:30 IST | T+1 (default); optional T+0 beta for top-500 phased Jan–May 2025 | Index-based circuit (10/15/20%); per-stock bands | F&O hub; weekly index expiries |
| BSE | India | BSE / XBOM | 09:15–15:30 IST | T+1; T+0 optional | Per-stock circuit filters | Sensex weekly options |
| SSE | China | SSE / XSHG | 09:30–11:30, 13:00–15:00 CST | T+1 securities, T+0 cash (A-share) | Main board ±10%; **STAR ±20%** (STAR/IPO: no limit first 5 days) | A-shares (CNY), foreign access via Stock Connect |
| SZSE | China | SZSE / XSHE | 09:30–11:30, 13:00–15:00 CST (continuous to 14:57, closing auction 14:57–15:00) | T+1 | Main ±10%; **ChiNext ±20%** (no limit first 5 days); main-board ST stocks ±5%→±10% (from 6 Jul 2026) | ChiNext = growth board |
| BSE (Beijing) | China | BSE / BJSE | as SSE/SZSE | T+1 | ±30% (first day no limit) | SME-focused |
| JPX (TSE) | Japan | XJPX / XTKS | 09:00–11:30, 12:30–15:30 JST (close extended to 15:30 since 5 Nov 2024) | T+2 | Absolute-yen renkahaba bands | Prime/Standard/Growth segments |
| HKEX | Hong Kong | XHKG | 09:30–12:00, 13:00–16:00 HKT | T+2 (cash); Northbound A-shares T+0 sec | No daily % limit (VCM cooling on select stocks) | Board lot varies by stock |
| KRX | Korea | XKRX (KOSPI), XKOS (KOSDAQ) | 09:00–15:30 KST | **T+2** | **±30%** | Nextrade (NXT) ATS pre/after sessions since 4 Mar 2025 |
| TWSE | Taiwan | XTAI | 09:00–13:30 CST | T+2 | ±10% | TPEx (MIC **ROCO**) for OTC stocks |
| SGX | Singapore | XSES | 09:00–17:00 SGT | T+2 | No fixed % limit | REITs/derivatives hub |
| HOSE/HNX | Vietnam | XSTC/HSTC | 09:00–14:45 ICT | **T+1.5** (roadmap to T+1, Q3 2026) | HOSE ±7%, HNX ±10% | Frontier market |

Sources: [NSE T+0 page](https://www.nseindia.com/static/products-services/t0-settlement-cycle), [SEBI T+0 circular](https://www.sebi.gov.in/legal/circulars/mar-2024/introduction-of-beta-version-of-t-0-rolling-settlement-cycle-on-optional-basis-in-addition-to-the-existing-t-1-settlement-cycle-in-equity-cash-markets_82455.html), [SSE trading mechanism](https://english.sse.com.cn/start/trading/mechanism/), [STAR Market](https://en.wikipedia.org/wiki/Shanghai_Stock_Exchange_STAR_Market), [KRX trading guide (PDF)](https://global.krx.co.kr/contents/GLB/01/0109/0109000000/guide_to_trading_in_the_korean_stock_market.pdf), [FSC Korea release](https://www.fsc.go.kr/eng/pr010101/83967), [HKEX securities hours](https://www.hkex.com.hk/Services/Trading-hours-and-Severe-Weather-Arrangements/Trading-Hours/Securities-Market?sc_lang=en).

### Key mechanics quirks (load-bearing for backtests)
- **China A-share T+1**: shares bought today cannot be sold until the next session — intraday round-trips on the same line are blocked. Cash is T+0. This breaks naive intraday backtests ported from US data.
- **India F&O lot sizes** are revised periodically by NSE to keep contract value inside SEBI's notional band (₹5–10 lakh). Index-derivative lot sizes were revised effective **28 Oct 2025 EOD** (e.g. NIFTY 75→65, BANKNIFTY 35→30, FINNIFTY 65→60, MIDCPNIFTY 140→120) with existing weekly/monthly lots applying until the 30 Dec 2025 expiry, and quarterly/half-yearly contracts revised from **30 Dec 2025 EOD** ([NSE F&O circular No. 176/2025 PDF](https://nsearchives.nseindia.com/content/circulars/FAOP70616.pdf)). Always re-fetch the current lot file before sizing.
- **Korea ±30%** price band is far wider than China's ±10% — tail-risk models calibrated on A-shares under-state Korean daily vol.
- **STAR (±20%) / ChiNext (±20%)** vs main-board (±10%): regime-split your limit-hit indicators by board. Since the **Feb 2023 full registration-system reform**, newly listed stocks on **all** boards (STAR, ChiNext, Beijing, **and the SSE/SZSE main boards**) have **no price limit for their first 5 trading days**, then revert to the board's normal band (main ±10%, STAR/ChiNext ±20%). The legacy ±44%/−36% day-1 cap belonged to the old approval-based system and no longer applies.

---

## 2. INDIA — broker APIs + open-source + official files

India is the richest retail-broker API ecosystem in Asia. Most retail data flows through **broker APIs** (you need a funded account) plus free **bhavcopy** end-of-day files.

| Source | Type | Auth | What you get | URL |
|---|---|---|---|---|
| Zerodha **Kite Connect** v3 | Broker REST + WebSocket | API key + daily login token | Quotes, OHLC historical candles, instruments dump, orders, F&O, MF | https://kite.trade/docs/connect/v3/ |
| **Upstox** API v2 | Broker REST + WS | OAuth | Quotes, historical, orders | https://upstox.com/developer/api-documentation/ |
| **Angel One SmartAPI** | Broker REST + WS | TOTP + API key | Free historical + live, orders | https://smartapi.angelone.in/ |
| **Fyers** API v3 | Broker REST + WS | OAuth | Quotes, history, orders | https://myapi.fyers.in/docsv3 |
| **Dhan** API v2 | Broker REST + WS | access token | Quotes, history, orders, options chain | https://dhanhq.co/docs/v2/ |
| **breeze-connect** (ICICI Direct) | Broker SDK | API key/secret + session | Quotes, history, orders | https://github.com/Idirect-Tech/Breeze-Python-SDK |
| **NSEpy / nsepython / jugaad-data** | Open-source scrapers | none | EOD, F&O, indices, bhavcopy | https://github.com/jugaad-py/jugaad-data |
| NSE official reports | Bulk files | none | Daily bhavcopy, F&O, indices | https://www.nseindia.com/all-reports |
| BSE bhavcopy | Bulk files | none | EOD equities | https://www.bseindia.com/markets/MarketInfo/BhavCopy |
| `yfinance` `.NS` / `.BO` | Delayed EOD | none | Adjusted OHLCV | https://github.com/ranaroussi/yfinance |

**Kite Connect pricing (2025):** order-placement and account/portfolio APIs are **free** for personal use since Mar 2025; the **₹500/month per API key** Connect plan adds live market data + historical candle data at no extra charge ([Zerodha charges article](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/what-are-the-charges-for-kite-apis), [pykiteconnect](https://github.com/zerodha/pykiteconnect)). Segments use the `EXCHANGE:TRADINGSYMBOL` convention, e.g. `NSE:INFY`, `NFO:NIFTY25...CE`, `BSE:RELIANCE`.

**Ticker suffixes for Yahoo:** NSE = `.NS` (e.g. `RELIANCE.NS`, `INFY.NS`), BSE = `.BO` (e.g. `RELIANCE.BO`, `BSE.NS` for the exchange-operator stock).

```python
# Zerodha Kite Connect — historical candles
from kiteconnect import KiteConnect
kite = KiteConnect(api_key="xxx")
kite.set_access_token("daily_token")          # from login flow
# instrument_token from kite.instruments("NSE")
df = kite.historical_data(
    instrument_token=738561,                  # RELIANCE on NSE
    from_date="2025-01-01", to_date="2025-06-29",
    interval="day")
```

```python
# Open-source EOD via jugaad-data (no account, NSE bhavcopy)
from jugaad_data.nse import stock_df
from datetime import date
df = stock_df(symbol="SBIN", from_date=date(2025,1,1),
              to_date=date(2025,6,29), series="EQ")
```

---

## 3. CHINA — Tushare, AkShare, baostock (+ Stock Connect for foreigners)

A-share data is dominated by community Python libraries fed off vendor/exchange feeds. None require a Chinese brokerage account.

| Source | Auth | Coverage | Free/Paid | URL |
|---|---|---|---|---|
| **Tushare Pro** | token (points system) | A-share EOD/min, fundamentals, money-flow, indices, HK | Freemium (points; contribute or pay for high-freq) | https://tushare.pro/ |
| **AkShare** | none | Huge breadth: A-share, HK, US, futures, bonds, macro, fund | Free, open-source | https://akshare.akfamily.xyz/ • https://github.com/akfamily/akshare |
| **baostock** | none (anon login) | A-share daily/min, adjusted, valuation | Free | http://baostock.com/ |
| `yfinance` `.SS`/`.SZ` | none | Delayed EOD | Free | — |
| Wind / Choice (东方财富) | commercial license | Institutional terminal + API | Paid | — |

**Suffixes:** SSE = `.SS` (`600519.SS` Kweichow Moutai), SZSE = `.SZ` (`000001.SZ` Ping An Bank). Tushare uses `600519.SH` / `000001.SZ`; AkShare uses bare 6-digit codes (`600519`) and prefixed forms (`sh600519`, `sz000001`).

```python
import tushare as ts
pro = ts.pro_api("YOUR_TOKEN")
df = pro.daily(ts_code="600519.SH", start_date="20250101", end_date="20250629")
```
```python
import akshare as ak
df = ak.stock_zh_a_hist(symbol="600519", period="daily",
                        start_date="20250101", end_date="20250629", adjust="qfq")  # qfq=fwd-adj
```

> Foreign quant access to A-shares in practice goes through **Stock Connect** (Northbound). Northbound A-shares settle securities T+0; trades follow SSE/SZSE hours ([HKEX Stock Connect calendar](https://www.hkex.com.hk/Mutual-Market/Stock-Connect/Reference-Materials/Trading-Hour,-Trading-and-Settlement-Calendar?sc_lang=en)).

---

## 4. JAPAN — J-Quants API (the official JPX research feed)

[**J-Quants**](https://jpx-jquants.com/en) is JPX's own data service for individuals: historical prices, financial statements, TOPIX, margin/short data via REST. Authentication migrated from the token method to the **API-key method** in the **V2 API**; the **V1 API was retired on 1 Jun 2026** (registrations from 22 Dec 2025 are V2-only) ([spec](https://jpx-jquants.com/en/spec), [V1→V2 migration](https://jpx-jquants.com/en/spec/migration-v1-v2), [JPX page](https://www.jpx.co.jp/english/markets/other-data-services/j-quants-api/index.html)).

Core **V2** endpoints (note the restructured paths vs retired V1): `/v2/equities/master` (listed info), `/v2/equities/bars/daily` (daily quotes), `/v2/fins/details` (financial statements), `/v2/indices/bars/daily/topix`, `/v2/equities/investor-types` (trades by investor), `/v2/markets/short-ratio`, `/v2/markets/margin-interest`. Plans (**Free / Light / Standard / Premium**) differ by **history depth and data lag** — the Free tier intentionally lags recent data, paid tiers reduce the lag and extend history.

**Yahoo suffix:** `.T` (e.g. `7203.T` Toyota, `6758.T` Sony). Official Python client: [`jquants-api-client-python`](https://github.com/J-Quants/jquants-api-client-python) (PyPI `jquants-api-client`); quick-start notebooks at [`jquants-api-quick-start`](https://github.com/J-Quants/jquants-api-quick-start).

```python
# J-Quants V2 uses API-key authentication (set the key from the dashboard)
import jquantsapi
cli = jquantsapi.Client(refresh_token="...")   # or configure the V2 API key
df = cli.get_prices_daily_quotes(code="7203", from_yyyymmdd="20250101", to_yyyymmdd="20250629")
```

---

## 5. KOREA — pykrx, KRX OpenAPI, DART filings

| Source | Auth | Coverage | URL |
|---|---|---|---|
| **pykrx** | none (scrapes KRX/Naver) | KOSPI/KOSDAQ/KONEX OHLCV, market cap, fundamentals, index | https://github.com/sharebook-kr/pykrx |
| **KRX OpenAPI** (Data Marketplace) | free API key (approval) | Official market data | https://openapi.krx.co.kr/ |
| KRX Data Marketplace | web/CSV | EOD + reference | https://data.krx.co.kr/ |
| **OpenDART** (FSS) | free API key | Corporate filings (*divulgações*), financial statements, ownership | https://opendart.fss.or.kr/ |
| `yfinance` `.KS`/`.KQ` | none | Delayed EOD | — |

**Suffixes:** KOSPI = `.KS` (`005930.KS` Samsung Electronics), KOSDAQ = `.KQ`. Remember Korea is **±30% price band, T+2**.

```python
from pykrx import stock
df = stock.get_market_ohlcv("20250101", "20250629", "005930")   # Samsung Electronics
tickers = stock.get_market_ticker_list("20250629", market="KOSPI")
```

---

## 6. TAIWAN — FinMind + TWSE open API

[**FinMind**](https://finmind.github.io/) exposes 50+ Taiwan-centric datasets via one endpoint `https://api.finmindtrade.com/api/v4/data` (Bearer token; ~300 req/hr anon, 600/hr verified) — prices, tick, PER/PBR, institutional (chip) flows, financials ([GitHub](https://github.com/FinMind/FinMind), [site](https://finmindtrade.com/)). TWSE/TPEx also publish free open-data JSON/CSV directly.

**Suffixes:** TWSE = `.TW` (`2330.TW` TSMC), TPEx/OTC = `.TWO`.

```python
from FinMind.data import DataLoader
api = DataLoader(); api.login_by_token(api_token="YOUR_TOKEN")
df = api.taiwan_stock_daily(stock_id="2330", start_date="2025-01-01", end_date="2025-06-29")
```

---

## 7. SOUTHEAST ASIA & VIETNAM

| Market | Yahoo suffix | Open-source / official |
|---|---|---|
| Singapore (SGX) | `.SI` (`D05.SI` DBS) | SGX official data products; `yfinance` |
| Indonesia (IDX) | `.JK` (`BBRI.JK`) | `yfinance`, IDX site |
| Thailand (SET) | `.BK` (`PTT.BK`) | `yfinance`, SET SMART |
| Malaysia (Bursa) | `.KL` (`1155.KL`) | `yfinance` |
| Philippines (PSE) | `.PS` (`BDO.PS`) | `yfinance` |
| Vietnam (HOSE/HNX) | `.VN` | **vnstock** (best free Vietnam lib) — https://github.com/thinh-vu/vnstock |

```python
from vnstock import Vnstock
df = Vnstock().stock(symbol="VNM", source="VCI").quote.history(start="2025-01-01", end="2025-06-29")
```

---

## 8. PAN-ASIA / COMMERCIAL vendors

| Vendor | Asia coverage | Free tier | Note |
|---|---|---|---|
| Refinitiv / LSEG | Full pan-Asia incl. A-share, BSE Beijing | No | Eikon/Workspace + APIs |
| Bloomberg | Full | No | Terminal/BLPAPI |
| FactSet | Full | No | Institutional |
| **EOD Historical Data (eodhd.com)** | Broad incl. NSE/BSE, SSE/SZSE, JPX, KRX, TWSE, SET, VN | Limited free | Cheap REST EOD/fundamentals |
| **Nasdaq Data Link** | Selected | Some free tables | Quandl successor |
| **Alpha Vantage** | Global incl. India, some Asia | Free key (rate-limited) | Daily/intraday |
| **Polygon.io** | Mostly US; thin Asia | Free tier US | Limited Asia |

---

## 9. MASTER comparison table

| Source | Country | Coverage | Free/Paid | How-to / endpoint |
|---|---|---|---|---|
| Kite Connect | India | NSE/BSE/NFO eq+F&O, history | ₹500/mo data | `kiteconnect`; `kite.historical_data()` |
| jugaad-data / nsepython | India | NSE/BSE EOD, F&O, bhavcopy | Free | scrapes nseindia.com |
| NSE all-reports | India | Daily bhavcopy files | Free | https://www.nseindia.com/all-reports |
| Tushare | China | A-share + HK, fundamentals | Freemium | `pro.daily(ts_code="600519.SH")` |
| AkShare | China | A/HK/US/futures/macro | Free | `ak.stock_zh_a_hist()` |
| baostock | China | A-share adj. daily/min | Free | `bs.query_history_k_data_plus()` |
| J-Quants | Japan | JPX prices, fins, TOPIX | Freemium | V2 REST (`/v2/equities/bars/daily`) |
| pykrx | Korea | KOSPI/KOSDAQ OHLCV | Free | `stock.get_market_ohlcv()` |
| OpenDART | Korea | filings/financials | Free key | https://opendart.fss.or.kr/ |
| FinMind | Taiwan | 50+ TW datasets | Freemium | `api/v4/data` Bearer |
| vnstock | Vietnam | HOSE/HNX prices/fins | Free | `Vnstock().stock(...)` |
| yfinance | Pan-Asia | Delayed EOD (suffix-based) | Free | `yf.download("7203.T")` |
| EOD Historical Data | Pan-Asia | EOD + fundamentals | Paid (cheap) | REST `/eod/{TICKER.EX}` |

---

## 10. ML / quant angle

- **Cross-market factor models**: A-shares (T+1, ±10%) vs Korea (±30%) vs Japan (yen bands) need regime-specific limit-hit and gap features; do not pool raw returns naively.
- **Microstructure / limit-up modeling**: China's price-limit mechanism is a canonical RL/sequence-modeling testbed (limit-hit → next-day continuation). STAR/ChiNext (±20%) vs main board (±10%) is a clean natural split.
- **Alt/chip data**: Taiwan (FinMind institutional flows) and Korea (foreign ownership via KRX/DART) provide *fluxo institucional* features rare in Western retail data.
- **F&O / vol surfaces**: India's deep weekly-options market (NSE Nifty/Bank Nifty, BSE Sensex) is among the world's largest by contracts — rich for vol-forecasting and gamma-exposure studies; re-fetch lot sizes each revision.
- **Survivorship & adjustment**: AkShare/baostock `qfq`/`hfq` forward/backward adjustment flags matter for label leakage; J-Quants ships explicit adjustment factors.
- **Calendars**: each market has distinct holidays + China's lunch break and Korea's Nextrade extended sessions — align bars on exchange-local calendars, not UTC.

---

## 11. Brazil access (acesso a partir do Brasil)

Brazilian retail investors generally cannot directly trade NSE/SSE/KRX. Practical routes:
- **BDRs na B3** (*Brazilian Depositary Receipts*): a handful of Asia-linked names trade as BDRs; coverage is thin vs US names.
- **ETFs**: broad/EM ETFs give exposure — e.g. emerging-markets funds (SPEM and similar) holding China/India/Brazil; India- and China-specific ETFs trade on US exchanges (accessible via international brokers). Some local B3 ETFs track global/EM indices.
- **Corretoras internacionais** (Interactive Brokers etc.) give direct access to HKEX, SGX, JPX, and India NSE/BSE for qualified accounts; A-shares via Stock Connect.
- **Data-only** (no trading): all the open-source libs above (`yfinance`, AkShare, J-Quants Free, FinMind, pykrx, vnstock) work from Brazil for research without any local-market account.

---

**Sources:** [Kite Connect v3](https://kite.trade/docs/connect/v3/) · [Zerodha API charges](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/what-are-the-charges-for-kite-apis) · [NSE T+0](https://www.nseindia.com/static/products-services/t0-settlement-cycle) · [SEBI T+0 circular](https://www.sebi.gov.in/legal/circulars/mar-2024/introduction-of-beta-version-of-t-0-rolling-settlement-cycle-on-optional-basis-in-addition-to-the-existing-t-1-settlement-cycle-in-equity-cash-markets_82455.html) · [NSE F&O lot circular](https://nsearchives.nseindia.com/content/circulars/FAOP70616.pdf) · [NSE all-reports](https://www.nseindia.com/all-reports) · [BSE bhavcopy](https://www.bseindia.com/markets/MarketInfo/BhavCopy) · [Tushare](https://tushare.pro/) · [AkShare](https://github.com/akfamily/akshare) · [baostock](http://baostock.com/) · [J-Quants](https://jpx-jquants.com/en) · [J-Quants spec](https://jpx-jquants.com/en/spec) · [J-Quants V1→V2 migration](https://jpx-jquants.com/en/spec/migration-v1-v2) · [jquants-api-client-python](https://github.com/J-Quants/jquants-api-client-python) · [pykrx](https://github.com/sharebook-kr/pykrx) · [KRX OpenAPI](https://openapi.krx.co.kr/) · [OpenDART](https://opendart.fss.or.kr/) · [KRX trading guide](https://global.krx.co.kr/contents/GLB/01/0109/0109000000/guide_to_trading_in_the_korean_stock_market.pdf) · [FSC Korea](https://www.fsc.go.kr/eng/pr010101/83967) · [FinMind](https://finmind.github.io/) · [SSE mechanism](https://english.sse.com.cn/start/trading/mechanism/) · [STAR Market](https://en.wikipedia.org/wiki/Shanghai_Stock_Exchange_STAR_Market) · [HKEX hours](https://www.hkex.com.hk/Services/Trading-hours-and-Severe-Weather-Arrangements/Trading-Hours/Securities-Market?sc_lang=en) · [HKEX Stock Connect calendar](https://www.hkex.com.hk/Mutual-Market/Stock-Connect/Reference-Materials/Trading-Hour,-Trading-and-Settlement-Calendar?sc_lang=en) · [vnstock](https://github.com/thinh-vu/vnstock) · [Yahoo suffix list](https://help.yahoo.com/kb/SLN2310.html)

**Keywords:** Asia market data API, India NSE BSE data, Kite Connect, Tushare, AkShare, baostock, J-Quants JPX, pykrx KRX OpenDART, FinMind TWSE, vnstock Vietnam, A-share T+1, price limit band, F&O lot size, yfinance suffix, ticker .NS .BO .SS .SZ .T .KS .TW .SI, BDR ETF — dados de mercado asiático, ações da Índia e China, derivativos, banda de oscilação, ciclo de liquidação, fluxo institucional, corretora internacional, ETF emergentes, quant financeiro
