# Brazil & B3 Market Data — APIs and Datasets

> Practical, source-verified guide to *getting* Brazilian market data (B3 equities, futures, public bonds, funds, macro) for quantitative research — free and paid APIs, code snippets, and the local quirks (.SA suffix, lote padrão, proventos) that break naive pipelines.

Brazil's single exchange, **B3 — Brasil, Bolsa, Balcão** (formed by the 2017 merger of BM&FBOVESPA and CETIP), lists equities, ETFs, BDRs, real‑estate funds (FIIs), and one of the most liquid retail futures complexes in the world (mini index and mini dollar). For researchers (*pesquisadores*), quants (*quants*), and investors (*investidores*), the data landscape splits into four layers: **free community APIs**, **broker‑fed terminals**, **official regulator/government open data**, and **commercial vendors**. This page maps each with working endpoints.

---

## 1. Free & community APIs (the fast path)

### brapi.dev — the de-facto free B3 REST API

[brapi.dev](https://brapi.dev/) ([docs](https://brapi.dev/docs)) serves quotes, dividends, fundamentals, crypto, FX and economic indicators for B3-listed assets as JSON over REST. Authentication is a Bearer token (`Authorization: Bearer YOUR_TOKEN`), or `?token=` query param for no-code tools. Four tickers — **PETR4, VALE3, ITUB4, MGLU3** — work without a token for testing ([docs](https://brapi.dev/docs/acoes)).

```bash
# Latest quote (no token needed for the 4 test tickers)
curl "https://brapi.dev/api/quote/PETR4,VALE3"

# 1-year daily history + fundamentals (needs token)
curl "https://brapi.dev/api/quote/WEGE3?range=1y&interval=1d&fundamental=true&dividends=true" \
     -H "Authorization: Bearer YOUR_TOKEN"

# List every available ticker
curl "https://brapi.dev/api/available"
```

The response is `{ "results": [ { "symbol": ..., "regularMarketPrice": ..., "historicalDataPrice": [...] } ], "requestedAt": ..., "took": ... }`. Beyond equities, brapi also documents coverage for **Tesouro Direto, crypto and currency (FX)** endpoints ([docs](https://brapi.dev/docs)). Pricing/rate limits are on the [plans page](https://brapi.dev/pricing); the free tier is usable for research, paid tiers lift request caps.

### yfinance — Yahoo Finance with the `.SA` suffix

The single most important quirk: **every B3 ticker on Yahoo Finance carries the `.SA` suffix** (São Paulo). `PETR4` → [`PETR4.SA`](https://finance.yahoo.com/quote/PETR4.SA/). Fractional-market codes append `F` *before* `.SA` (e.g. `PETR4F.SA`).

```python
import yfinance as yf
df = yf.download(["PETR4.SA", "VALE3.SA", "^BVSP"], start="2015-01-01", auto_adjust=True)
# ^BVSP = Ibovespa index. auto_adjust=True applies split/dividend adjustment (proventos)
```

yfinance is free and convenient but unofficial, rate-limited, and prone to silent gaps — validate against a second source for production research.

---

## 2. Broker-fed terminals (real-time & intraday)

### MetaTrader 5 — Python package via a Brazilian broker

The official **`MetaTrader5`** Python package pulls real-time and historical OHLCV directly from a broker's data feed, including B3 equities and futures, once the MT5 terminal is running and logged in ([MQL5 docs: `copy_rates_range`](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py)). MT5 is supported by many Brazilian brokers (XP, Clear, Rico, Genial, Modal, etc.).

```python
import MetaTrader5 as mt5
from datetime import datetime
mt5.initialize()  # MT5 terminal must be open and logged in
rates = mt5.copy_rates_range("PETR4", mt5.TIMEFRAME_M5,
                             datetime(2024,1,1), datetime(2024,12,31))
import pandas as pd
df = pd.DataFrame(rates); df['time'] = pd.to_datetime(df['time'], unit='s')
```

Caveats ([MQL5](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py)): history depth depends on the broker; 1-minute bars often cover only ~1 year while 5/10-minute bars go back further; timestamps are in the **broker's timezone**, not local. This is the most common path for intraday WIN/WDO research without paying a vendor. **Nelogica Profit / ProfitDLL** is the other broker-terminal route widely used in Brazil (C/Python DLL bindings) for tick data.

---

## 3. Official regulator & government open data (free, authoritative)

### Banco Central do Brasil (BCB) — SGS, IPCA, Selic via `python-bcb`

The BCB's **SGS** (Sistema Gerenciador de Séries Temporais) is the canonical source for Brazilian macro/rate series. Raw REST endpoint ([Dados Abertos BCB](https://dadosabertos.bcb.gov.br/dataset/11-taxa-de-juros---selic)):

```
https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json&dataInicial=01/01/2010&dataFinal=31/12/2016
https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/{N}?formato=json
```

The **`python-bcb`** library by Wilson Freitas ([docs](https://wilsonfreitas.github.io/python-bcb/sgs.html) · [PyPI](https://pypi.org/project/python-bcb/) · [GitHub](https://github.com/wilsonfreitas/python-bcb)) wraps SGS plus the OData services (PTAX FX, FOCUS/Expectativas, interest rates):

```python
from bcb import sgs
df = sgs.get({'Selic': 11, 'IPCA': 433, 'IGP-M': 189}, start='2010-01-01')
# Series codes: Selic over (taxa over, % a.d.) = 11, IPCA mensal = 433, IGP-M = 189, CDI = 12
```

| Series | SGS code | Notes |
|---|---|---|
| Selic (taxa over, % a.d.) | 11 | daily realized over rate; **meta Selic (% a.a.) = 432** |
| CDI | 12 | daily (% a.d.) |
| IPCA (inflation, monthly %) | 433 | headline consumer inflation |
| IGP-M | 189 | rent/contract index (monthly %) |

(Codes per [python-bcb SGS docs](https://wilsonfreitas.github.io/python-bcb/sgs.html) and [BCB Dados Abertos](https://dadosabertos.bcb.gov.br/dataset/11-taxa-de-juros---selic). Series 11 is the realized **Selic over** rate as % a.d.; the policy **target/meta** is series 432, expressed % a.a. Note the BCB API caps some series at ~10-year windows per request — page by date range.)

### Tesouro Direto — government bonds (títulos públicos)

Historical price/rate series for every Tesouro Direto bond (Tesouro Selic / IPCA+ / Prefixado) is published as open data via Tesouro Transparente (CKAN) — the file `PrecoTaxaTesouroDireto.csv` with columns *Tipo Título, Data Vencimento, Data Base, Taxa Compra Manhã, Taxa Venda Manhã, PU Compra Manhã, PU Venda Manhã, PU Base Manhã* ([Tesouro Transparente CKAN](https://www.tesourotransparente.gov.br/ckan/dataset?res_format=CSV&tags=Tesouro+Direto) · [Tesouro Nacional APIs](https://www.gov.br/tesouronacional/pt-br/central-de-conteudo/apis)).

The live quote JSON is at `https://www.tesourodireto.com.br/json/br/com/b3/tesourodireto/service/api/treasurybondsinfo.json`, but it sits behind **Cloudflare** which 403s non-browser clients ([B3 developers — Tesouro Direto API](https://developers.b3.com.br/apis/tesouro-direto)). For programmatic access prefer the CKAN CSV (history) or brapi's Tesouro Direto endpoint. B3's official Tesouro Direto API is **B2B only**, not offered to individuals.

### CVM — funds & listed-company filings (Dados Abertos)

The **Portal de Dados Abertos da CVM** ([dados.cvm.gov.br](https://dados.cvm.gov.br/)) is the regulator's free catalog: fund daily reports (**Informe Diário**), portfolio composition (**CDA**), and listed-company filings (**DFP** annual / **ITR** quarterly).

```python
import pandas as pd
url = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202506.zip"
df = pd.read_csv(url, sep=';', encoding='latin-1', compression='zip')
# columns: CNPJ_FUNDO, DT_COMPTC, VL_QUOTA, VL_PATRIM_LIQ, CAPTC_DIA, RESG_DIA, NR_COTST
```

Informe Diário covers the last 12 months and is refreshed regularly ([fi-doc-inf_diario dataset](https://dados.cvm.gov.br/dataset/fi-doc-inf_diario)). All files are semicolon-delimited CSV, Latin-1 encoded — a frequent gotcha.

### B3 official — UP2DATA & market data feeds

**UP2DATA** is B3's official end-of-day data distribution platform ([about](https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/about-up2data/) · [available data](https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/available-data/)). It covers equities, indices, currencies, interest rates (DI futures, swaps), fixed income, commodities, crypto-asset derivatives, corporate actions, yield curves, volatility surfaces, debentures, CRI/CRA, securities lending, and fixed-income tick-by-tick — delivered as TXT/CSV/XML/JSON via client software or cloud. It is **contracted/paid**; pricing tiers depend on internal-use vs. distribution ([pricing FAQ](https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/faq/contracting-and-prices.htm)). B3 also publishes free daily reports (séries históricas / COTAHIST, market data reports) on its site.

---

## 4. Commercial vendors

| Vendor | What it adds |
|---|---|
| **Economatica** | Long-standing LatAm fundamentals + adjusted prices terminal, used heavily in academic finance in Brazil. |
| **Comdinheiro** | Web/API platform for prices, fundamentals, fund analytics, fixed income; popular with asset managers. |
| **Nasdaq Data Link** (ex-Quandl) | Hosts BCB time series; e.g. [`BCB/7` = Bovespa Index](https://data.nasdaq.com/data/BCB/7-bovespa-index), accessible via the `nasdaqdatalink` Python client. CSV/JSON/XML. |
| **B3 UP2DATA** | Official, authoritative end-of-day & reference data (see §3). |

```python
import nasdaqdatalink
nasdaqdatalink.ApiConfig.api_key = "YOUR_KEY"
bvsp = nasdaqdatalink.get("BCB/7")  # Bovespa index series mirrored from BCB SGS
```

---

## 5. Fundamentals by scraping (Status Invest / Fundamentus)

When no clean API exists, the community scrapes **Status Invest**, **Fundamentus**, **Investidor10** and **InvestSite** for P/L, P/VP, dividend yield, ROE, net-debt/EBITDA, etc. ([Stock-Screener-bovespa](https://github.com/cammneto/Stock-Screener-bovespa) · [pyFundamentus](https://github.com/alexcamargos/pyFundamentus) / [PyPI](https://pypi.org/project/pyfundamentus/) · [statusinvest scraper](https://github.com/lfreneda/statusinvest)). Typical stack: `requests` + `BeautifulSoup` + `pandas`; some pages expose hidden JSON. Respect robots.txt and terms of use — these are courtesy sources, not licensed feeds.

---

## 6. Contract specs & trading mechanics (the quirks that break pipelines)

| Item | Spec | Source |
|---|---|---|
| Equity round lot (lote padrão) | **100 shares**; multiples only | [B3 trading rules](https://www.b3.com.br/en_us/products-and-services/trading/equities/cash-equities/b3-trading-characteristics-and-rules.htm) |
| Fractional market (mercado fracionário) | 1–99 shares; ticker + **`F`** (e.g. `VALE3F`) | [Bora Investir / B3](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/acoes/o-que-e-o-mercado-fracionario-como-investir-nele/) |
| Equity continuous session | ~10:00–16:55 (BRT); closing call to ~17:00; after-market ~17:30–18:00 | [B3 equities trading hours](https://www.b3.com.br/en_us/solutions/platforms/puma-trading-system/for-members-and-traders/trading-hours/equities/) |
| Ticker convention | ON = `…3`, PN = `…4`, Units = `…11`, FII = `…11`, BDR = `…34/35/32/39` | B3 listing convention |
| Mini Ibovespa future (WIN) | R$ **0.20** per index point; tick **5 points**; expires even months, Wed nearest the 15th; cash-settled | [Mini Ibovespa Futures](https://www.b3.com.br/en_us/products-and-services/trading/equities/mini-ibovespa-futures.htm) |
| Mini USD future (WDO) | contract **USD 10,000**; quoted BRL per USD 1,000; tick **BRL 0.50 per USD 1,000**; cash-settled, expires 1st business day of month | [Mini U.S. Dollar Futures](https://www.b3.com.br/en_us/products-and-services/trading/exchange-rates/mini-u-s-dollar-futures.htm) |

**Adjustment for proventos (corporate actions):** Brazilian raw close prices are *not* adjusted for dividends, JCP (juros sobre capital próprio), splits (desdobramentos) or bonuses (bonificações). Always backtest on an **adjusted** series — `yf.download(..., auto_adjust=True)`, brapi's adjusted history, or Economatica's adjusted prices — or your returns will show false jumps on ex-dates.

---

## 7. ML / quant angle

Machine learning on B3 data is constrained by the same realities as any single emerging market: ~10–20 years of clean daily history, a heavy-tailed currency (BRL), and structural breaks (2008, 2015–16 recession, 2020 COVID, fiscal cycles). Common, defensible uses:

- **Volatility & risk forecasting** — GARCH/HAR baselines vs. LSTM/Temporal Fusion Transformers on Ibovespa and USD/BRL; the liquid **WIN/WDO** mini futures give clean intraday OHLCV (via MT5) for realized-volatility and microstructure models.
- **Cross-sectional equity factors** — building value/quality/momentum signals from **CVM DFP/ITR** filings + adjusted prices; gradient-boosted trees (LightGBM/XGBoost) for return ranking with strict **purged, embargoed walk-forward CV** to avoid leakage on overlapping fundamentals.
- **Macro nowcasting** — **BCB SGS** (Selic, IPCA, IGP-M, CDI) and **FOCUS/Expectativas** (via `python-bcb`) as features for regime models and fixed-income (Tesouro Direto) curve forecasting.
- **NLP** — CVM filings, Banco Central minutes (Copom), and earnings releases (largely Portuguese) for sentiment/event signals; multilingual or PT-tuned transformers.

Watch for **survivorship bias** (delisted tickers vanish from free APIs), **low liquidity** outside the Ibovespa names (wide spreads invalidate fill assumptions), and **timezone/holiday calendars** (B3 has its own holiday set — use `pandas-market-calendars` or B3's calendar).

---

## 8. Data & APIs — quick comparison

| Source | Coverage | Free / Paid | How to access |
|---|---|---|---|
| [brapi.dev](https://brapi.dev/docs) | Equities, FIIs, ETFs, BDR, crypto, FX, Tesouro | Free tier + paid | REST + Bearer token |
| [yfinance](https://finance.yahoo.com/quote/PETR4.SA/) | Equities, indices, FX (`.SA`) | Free (unofficial) | `pip install yfinance` |
| [MetaTrader5](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py) | Equities + futures, real-time/intraday | Free (needs broker) | `pip install MetaTrader5` |
| Nelogica Profit / ProfitDLL | Tick & intraday via broker | Paid platform | DLL / Python bindings |
| [python-bcb](https://wilsonfreitas.github.io/python-bcb/) / [BCB SGS](https://dadosabertos.bcb.gov.br/) | Macro, rates, FX, FOCUS | Free | `pip install python-bcb` / REST |
| [Tesouro Transparente CKAN](https://www.tesourotransparente.gov.br/ckan/dataset?res_format=CSV&tags=Tesouro+Direto) | Gov bonds, price/rate history | Free | CSV download |
| [CVM Dados Abertos](https://dados.cvm.gov.br/) | Funds (Informe Diário, CDA), DFP/ITR | Free | CSV (zip), Latin-1 |
| [B3 UP2DATA](https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/about-up2data/) | Official EOD + reference, all markets | Paid (contract) | Client / cloud, TXT/CSV/XML/JSON |
| [Nasdaq Data Link](https://data.nasdaq.com/data/BCB/7-bovespa-index) | BCB/macro mirrors, alt-data | Free + paid | `nasdaqdatalink` API |
| Economatica | Adjusted prices + fundamentals | Paid | Terminal / export |
| Comdinheiro | Prices, fundamentals, funds | Paid | Web / API |
| [Status Invest](https://github.com/lfreneda/statusinvest) / [Fundamentus](https://github.com/alexcamargos/pyFundamentus) | Fundamentals (scraped) | Free (scraping) | `requests`+`BeautifulSoup` |

---

**Sources:** [brapi.dev docs](https://brapi.dev/docs) · [brapi pricing](https://brapi.dev/pricing) · [Yahoo PETR4.SA](https://finance.yahoo.com/quote/PETR4.SA/) · [MQL5 copy_rates_range](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py) · [python-bcb SGS](https://wilsonfreitas.github.io/python-bcb/sgs.html) · [BCB Dados Abertos](https://dadosabertos.bcb.gov.br/dataset/11-taxa-de-juros---selic) · [Tesouro Nacional APIs](https://www.gov.br/tesouronacional/pt-br/central-de-conteudo/apis) · [Tesouro Transparente CKAN](https://www.tesourotransparente.gov.br/ckan/dataset?res_format=CSV&tags=Tesouro+Direto) · [B3 developers — Tesouro Direto](https://developers.b3.com.br/apis/tesouro-direto) · [CVM Dados Abertos](https://dados.cvm.gov.br/) · [CVM Informe Diário](https://dados.cvm.gov.br/dataset/fi-doc-inf_diario) · [B3 UP2DATA](https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/about-up2data/) · [B3 Mini Ibovespa Futures](https://www.b3.com.br/en_us/products-and-services/trading/equities/mini-ibovespa-futures.htm) · [B3 Mini USD Futures](https://www.b3.com.br/en_us/products-and-services/trading/exchange-rates/mini-u-s-dollar-futures.htm) · [B3 equities trading hours](https://www.b3.com.br/en_us/solutions/platforms/puma-trading-system/for-members-and-traders/trading-hours/equities/) · [Nasdaq Data Link BCB/7](https://data.nasdaq.com/data/BCB/7-bovespa-index)

**Keywords:** B3 market data API, dados de mercado B3, brapi.dev, yfinance .SA, cotações ações brasileiras, MetaTrader5 Python B3, python-bcb SGS, Selic IPCA CDI API, Banco Central dados abertos, Tesouro Direto API títulos públicos, CVM dados abertos fundos informe diário, UP2DATA B3, Economatica, Comdinheiro, Status Invest Fundamentus scraping, WIN WDO mini índice mini dólar, lote padrão mercado fracionário, ajuste por proventos, Ibovespa quant machine learning, séries temporais financeiras Brasil.
