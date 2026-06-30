# Macro & Economic Data for Markets

> Authoritative, current (2024-2026) catalog of macroeconomic & economic data beyond basic FRED — big research panels (FRED-MD/QD), vintage/point-in-time archives (ALFRED), free aggregators (DBnomics), official-statistics APIs (World Bank, OECD, IMF, BIS, Eurostat), economic-calendar APIs, official nowcasts & recession models, and Brazil 🇧🇷 sources (BCB SGS, IBGE SIDRA, IPEAData). Heavy on APIs, free/paid status, and look-ahead/point-in-time honesty.

This page complements `Models_Features_and_Datasets/Global_Datasets_for_Markets.md` (which lists FRED, World Bank, OECD, IMF, BIS at a one-line level) and `Brazil_B3_Market_Data_APIs.md`. Here we go **deep** on macro: the research-grade panels, the *vintage* archives that prevent look-ahead bias, the free aggregators that unify hundreds of providers, the paid terminals, and the official real-time nowcast/recession signals. Everything below was confirmed to exist and be current as of 2026-06.

---

## 1. Research-grade macro panels (ready-made feature matrices)

These are curated, **transformation-ready** panels built for factor models, forecasting, and ML — far more convenient than scraping individual FRED series.

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **FRED-MD** (St. Louis Fed, McCracken & Ng 2016) | ~134 monthly US series in 8 groups (Output & income, Labor, Housing, Consumption/orders/inventories, Money & credit, Interest rates & spreads, Prices, Stock market); **header row carries transformation codes** | US, 1959→present | Free | Monthly CSV (`current.csv`) + **monthly vintage CSVs** under `monthly/` | [research.stlouisfed.org/econ/mccracken/fred-databases](https://research.stlouisfed.org/econ/mccracken/fred-databases/) |
| **FRED-QD** (St. Louis Fed, McCracken & Ng) | ~248 quarterly US series in 14 groups (NIPA, Industrial Production, Employment, Housing, Inventories/Orders/Sales, Prices, Earnings & Productivity, Interest Rates, Money & Credit, Household Balance Sheets, Exchange Rates, Other, Stock Markets, Non-Household Balance Sheets); same tcode header design | US, 1959:Q1→present | Free | Quarterly CSV + vintages | [stlouisfed.org/.../fred-qd](https://www.stlouisfed.org/publications/review/2021/01/14/fred-qd-a-quarterly-database-for-macroeconomic-research) |
| **Transformation codes** | tcode 1=level, 2=Δ, 3=Δ², 4=ln, 5=Δln, 6=Δ²ln, 7=Δ(x/x₋₁−1) — applied per-column to induce stationarity | — | Free | First data row in each CSV | [FRED-MD description (PDF)](https://users.ssc.wisc.edu/~behansen/econometrics/FRED-MD_description.pdf) |
| **`fbi` / `fredmd` R pkgs, `pyfredapi`** | Loaders + factor tools for FRED-MD/QD | — | Free | R/Python | [SSRN FRED-QD paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3587692) |

> **Point-in-time note:** the `current.csv` is the *latest-vintage* panel (revised data → look-ahead bias for backtests). For honest backtesting use the **dated vintage files** in the `monthly/`/`quarterly/` folders (e.g. `2020-03.csv`), which reflect what was actually published that month.

---

## 2. Vintage / point-in-time (real-time) data — avoid look-ahead bias

The single most overlooked issue in macro backtesting: most series are **revised** for years after first release. Vintage archives store *what was known on each date*.

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **ALFRED** (ArchivaL FRED, St. Louis Fed) | Every vintage of FRED series; each obs carries `realtime_start`/`realtime_end` | US + intl, full FRED history | Free | FRED API with `realtime_start`/`vintage_dates` params; `fredapi`, `pyfredapi` | [alfred.stlouisfed.org](https://alfred.stlouisfed.org/) · [API docs](https://fred.stlouisfed.org/docs/api/fred/alfred.html) |
| **Philadelphia Fed Real-Time Data Set (RTDSM)** | Vintage NIPA/macro panels purpose-built for real-time research; Greenbook forecasts; SPF | US, 1965→present | Free | CSV/Excel + Excel add-in | [philadelphiafed.org/.../real-time-data](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research) |
| **OECD MEI Original Release & Revisions** | First-release vs revised macro indicators | OECD | Free | OECD SDMX API | [data-explorer.oecd.org](https://data-explorer.oecd.org/) |
| **`alfred`/`fredr` (R), `fredapi` (Py)** | Pull arbitrary vintages programmatically | — | Free | `get_series_as_of_date()` / `realtime_*` | [github.com/mortada/fredapi](https://github.com/mortada/fredapi) |

> 🇧🇷 **Brazil PIT:** BCB SGS does **not** expose true vintages for most series; for point-in-time SELIC/IPCA market views use the **BCB Expectativas (Focus)** OData API (weekly snapshots of forecasts, dated). IBGE revisions are documented per-survey but not vintage-archived in one feed. For paid Brazil/EM point-in-time, **CEIC now ships a Point-in-Time (PIT) dataset** (long historical revision vintages) — see §5.

---

## 3. Free aggregators (one API, hundreds of providers)

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **DBnomics** | Unified mirror of **80+ providers** (INSEE, Eurostat, IMF, OECD, World Bank, BIS, ECB, Fed, BLS, national stats) into one data model | Millions of series, global | **Free, no key** | REST `https://api.db.nomics.world/v22/`; Python `dbnomics`, R `rdbnomics`, Julia, Stata, MATLAB | [db.nomics.world](https://db.nomics.world/) · [docs](https://docs.db.nomics.world/web-api/) |
| **EconDB** | Macro + **shipping/trade & energy** data; curated cross-country indicators | Global | Free (keyless basic) + Paid | REST (JSON/CSV); Python clients `prognosis` / `inquisitor` (token for non-basic use) | [econdb.com/api](https://www.econdb.com/api/) |
| **Nasdaq Data Link (ex-Quandl)** | Marketplace incl. free macro databases (e.g. Rate/FX, World Bank mirrors) + premium | Global | Free + Paid | `nasdaqdatalink` Python / REST, key required | [data.nasdaq.com](https://data.nasdaq.com/) |
| **`pandas-datareader`** | Single Python interface to FRED, World Bank, OECD, Eurostat, Stooq, MOEX, Fama-French | Multi-provider | Free | `pandas_datareader.data` | [pandas-datareader docs](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html) |
| **`pandaSDMX` / `sdmx1`** | Generic SDMX 2.1/3.0 client for **20+** orgs (World Bank, BIS, ILO, ECB, Eurostat, OECD, UN, UNICEF) | Multi-provider | Free | Python | [pandasdmx.readthedocs.io](https://pandasdmx.readthedocs.io/en/master/) |
| **Our World in Data** | Long-run curated macro/development series with sources | Global | Free | CSV/Grapher API | [ourworldindata.org](https://ourworldindata.org/) |

> **Why aggregators win:** they normalize SDMX dimension codes, currency/units, and frequency across providers, so you skip per-agency quirks. DBnomics is the standout — free, keyless, and a clean Python client.

---

## 4. Official statistics APIs (national / international)

All free; almost all SDMX-based. `pandaSDMX` or DBnomics can front-end any of them.

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **FRED API** | 800k+ US & intl series; categories, releases, tags | US + intl | Free (key) | REST + `fredapi`/`pyfredapi`; ALFRED via same key | [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/fred/) |
| **World Bank Indicators / Data360** | WDI + thousands of datasets; SDMX REST on Data360 | 200+ countries | Free | REST (XML/JSON), SDMX; `wbdata`, `wbgapi` | [data360.worldbank.org/en/api](https://data360.worldbank.org/en/api) |
| **OECD Data Explorer** | National accounts, prices, labor, finance, MEI | OECD + partners | Free | SDMX-JSON/ML REST | [data-explorer.oecd.org](https://data-explorer.oecd.org/) |
| **IMF Data** | WEO, IFS, BOP, GFS, DOTS | Global | Free | **SDMX 2.1 & 3.0** APIs | [data.imf.org/.../IMF-API](https://data.imf.org/en/Resource-Pages/IMF-API) |
| **BIS Data Portal** | Credit to non-financial sector, **credit-to-GDP gaps**, property prices (RPP/CPP), EER, debt service ratios, OTC derivatives | Global | Free | **SDMX REST v2.1** (JSON/XML/CSV) + bulk zip | [data.bis.org](https://data.bis.org/) · [API doc](https://stats.bis.org/api-doc/v2/) |
| **Eurostat** | EU macro/social/trade; JSON-stat + SDMX 3.0 | EU/EEA | Free | REST (JSON-stat, TSV, SDMX); R `eurostat`, `restatapi` | [Eurostat API intro](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-introduction) |
| **ECB Data Portal** | Euro-area rates, money, FX, yield curve | Euro area | Free | SDMX REST `data-api.ecb.europa.eu` | [data.ecb.europa.eu](https://data.ecb.europa.eu/) |
| **US BLS** | CPI, PPI, employment (CES/CPS), JOLTS, ECI | US | Free (key) | REST v2 JSON | [bls.gov/developers](https://www.bls.gov/developers/) |
| **US BEA** | GDP/NIPA, personal income, trade, regional | US | Free (key) | REST JSON | [apps.bea.gov/api](https://apps.bea.gov/API/signup/) |
| **US Census** | Trade, business formation, retail, ACS | US | Free (key) | REST | [census.gov/data/developers](https://www.census.gov/data/developers.html) |
| **UN Comtrade** | Bilateral merchandise/services trade | Global | Free + Paid (premium) | REST; `comtradeapicall` | [comtradeplus.un.org](https://comtradeplus.un.org/) |

> **SDMX tip:** the IMF/OECD/BIS/Eurostat/ECB all speak SDMX, so one `pandaSDMX`/`sdmx1` workflow covers them; for the others (FRED/BLS/BEA/Census) use their native REST. DBnomics mirrors most and removes key management.

---

## 5. Paid macro terminals & vendors

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **Haver Analytics** | ~200M+ time series, the buy-side macro standard; cleans on official + proprietary aggregations | Global | **Paid** | DLX, Haver API, Python/R, Excel | [haver.com](https://www.haver.com/) |
| **LSEG Datastream** (ex-Refinitiv) | Deep macro+market history — LSEG advertises 46M+ economic/financial indicators, 215 countries/regions, up to 120 yrs (600M+ time series) | Global | **Paid** | Datastream Web Service (Python/R/Matlab/EViews), Workspace | [lseg.com/.../datastream-and-macroeconomics](https://www.lseg.com/en/data-analytics/datastream-and-macroeconomics) |
| **Trading Economics** | ~20M indicators/series, calendar, forecasts, ratings | 196 countries | **Paid** (trial: 100k points / 100 reqs) | REST + `tradingeconomics` Python/Node | [tradingeconomics.com/api](https://tradingeconomics.com/api/) · [docs](https://docs.tradingeconomics.com/) |
| **Macrobond** | Curated macro/financial DB (300M+ series, 150M+ point-in-time series), analytics workbook | Global | **Paid** | Desktop + Web/Data API, Python | [macrobond.com](https://www.macrobond.com/) |
| **CEIC** (ISI Markets) | Emerging-market-deep macro (22M+ series, 200+ countries; China, Asia, LatAm incl. 🇧🇷); also a **Point-in-Time** dataset | Global, EM-strong | **Paid** | REST API, Python (`ceic_api_client`/PyCEIC), R, Excel, Snowflake | [ceicdata.com](https://www.ceicdata.com/) |
| **Moody's / S&P Global / Oxford Economics** | Forecasts, scenarios, country risk | Global | **Paid** | API/feeds | [oxfordeconomics.com](https://www.oxfordeconomics.com/) |

> Academic access: many of the above are available via **WRDS** ([wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu/)) at universities — check your institution before buying.

---

## 6. Economic-calendar APIs (event/release scheduling + actual vs consensus)

Critical for event-driven strategies and for aligning data to *release timestamps* (not reference periods).

| Source | Content | Free/Paid | API / How | Link |
|---|---|---|---|---|
| **Finnhub** | Global economic calendar (actual/estimate/prior), macro indicators | Free tier + Paid | REST, key | [finnhub.io/docs/api/economic-calendar](https://finnhub.io/docs/api/economic-calendar) |
| **Financial Modeling Prep (FMP)** | Economic releases calendar + macro datasets (GDP, CPI, rates) | Free tier + Paid | REST | [FMP economics calendar](https://site.financialmodelingprep.com/developer/docs/stable/economics-calendar) |
| **Trading Economics** | Real-time calendar + streaming updates | Paid (trial) | REST/WebSocket | [docs.tradingeconomics.com/economic_calendar](https://docs.tradingeconomics.com/economic_calendar/snapshot/) |
| **Econoday** | Institutional-grade release calendar w/ consensus | Paid | Feeds/widgets | [econoday.com](https://www.econoday.com/) |
| **investing.com calendar** | Widely used web calendar (no official API; scraping fragile/ToS-restricted) | Free (web) | HTML; community libs `investpy` *(deprecated, unreliable)* | [investing.com/economic-calendar](https://www.investing.com/economic-calendar/) |
| **Nasdaq Data Link / FRED releases** | FRED `release_dates` endpoint gives official US release schedule | Free | FRED API `fred/releases` | [FRED releases](https://fred.stlouisfed.org/docs/api/fred/releases.html) |

> **Release-timestamp note:** A CPI print for reference month *May* is released in *June*. For PIT-correct features, key features off the **release date**, not the reference period — the FRED `release_dates` and TE/Finnhub calendars give you that mapping.

---

## 7. Official nowcasts (real-time GDP / inflation trackers)

Model-based estimates published by central banks before official data — usable as live macro features. All free.

| Source | Tracks | Cadence | Free/Paid | How / Link |
|---|---|---|---|---|
| **Atlanta Fed GDPNow** | Current-quarter US real GDP growth | Updated ~6–7×/month | Free | Page + spreadsheet; **FRED `GDPNOW`** | [atlantafed.org/research-and-data/data/gdpnow](https://www.atlantafed.org/research-and-data/data/gdpnow) |
| **NY Fed Staff Nowcast** | US GDP via dynamic factor model + data-flow impacts (relaunched Sep 2023 as "Nowcast 2.0" after a 2021–2023 suspension) | Weekly (Fridays) | Free | XLSX downloads | [newyorkfed.org/research/policy/nowcast](https://www.newyorkfed.org/research/policy/nowcast) |
| **Cleveland Fed Inflation Nowcasting** | Daily CPI & PCE (monthly + current-quarter) | Daily | Free | Page + data | [clevelandfed.org/.../inflation-nowcasting](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting) |
| **St. Louis Fed (FRED) nowcast tag** | Mirror of `GDPNOW`, `PCENOW`, etc. | varies | Free | FRED API | [fred.stlouisfed.org/tags/series?t=nowcast](https://fred.stlouisfed.org/tags/series?t=nowcast) |
| **Dallas Fed Weekly Economic Index (WEI)** | High-frequency US activity index | Weekly | Free | FRED `WEI` | [dallasfed.org](https://www.dallasfed.org/research/wei) |

> 🇧🇷 **Brazil nowcast:** **IBRE/FGV Monitor do PIB** (monthly GDP tracker) and **BCB IBC-Br** (Índice de Atividade Econômica, a GDP proxy, FRED-style monthly) are the closest equivalents — IBC-Br is on BCB SGS (series 24364/24363). [FGV IBRE](https://portalibre.fgv.br/).

---

## 8. Recession & financial-conditions signals (ready models / series)

| Signal | What it is | Source/Series | Free/Paid | Link |
|---|---|---|---|---|
| **Sahm Rule** | Recession start when 3M-avg U3 unemployment rises ≥0.50pp over its prior-12M low | FRED `SAHMCURRENT` (latest data) & `SAHMREALTIME` (real-time/PIT) | Free | [FRED SAHMREALTIME](https://fred.stlouisfed.org/series/SAHMREALTIME) |
| **Yield-curve recession prob.** | NY Fed model: 10Y–3M spread → 12-month recession probability | NY Fed term-spread model; FRED `T10Y3M`, `T10Y2Y` | Free | [newyorkfed.org/.../yield-curve](https://www.newyorkfed.org/research/capital_markets/ycfaq) |
| **Excess Bond Premium (EBP)** | Gilchrist–Zakrajšek spread component capturing credit risk appetite; leads recessions ~12M | Fed FEDS Notes (updated dataset; direct CSV `ebp_csv.csv`) | Free | [Fed EBP update](https://www.federalreserve.gov/econres/notes/feds-notes/updating-the-recession-risk-and-the-excess-bond-premium-20161006.html) |
| **Chicago Fed NFCI / ANFCI** | National Financial Conditions Index (weekly) | FRED `NFCI`, `ANFCI` | Free | [chicagofed.org/nfci](https://www.chicagofed.org/research/data/nfci/current-data) |
| **Chicago Fed CFNAI** | Activity index, 85-indicator composite | FRED `CFNAI` | Free | [chicagofed.org/cfnai](https://www.chicagofed.org/research/data/cfnai/current-data) |
| **ADS Index (Aruoba-Diebold-Scotti)** | High-frequency business-conditions index | Philadelphia Fed | Free | [philadelphiafed.org/.../ads](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ads) |
| **NBER recession dates** | Official US business-cycle dating | FRED `USREC`, `USRECD` | Free | [nber.org/research/business-cycle-dating](https://www.nber.org/research/business-cycle-dating) |
| **OECD CLI** | Composite Leading Indicators, many countries (incl. 🇧🇷) | OECD SDMX | Free | [OECD CLI](https://data-explorer.oecd.org/) |

> **Honesty:** Sahm/yield-curve signal late or with lead-time variance; `SAHMREALTIME` exists *precisely* because the unrevised real-time print differs from `SAHMCURRENT`. For backtests, use the real-time/PIT variant and lag to the actual release date.

---

## 9. 🇧🇷 Brazil macro & economic data (APIs + Python)

| Source | Content | Free/Paid | API / How | Link |
|---|---|---|---|---|
| **BCB SGS** (Sistema Gerenciador de Séries Temporais) | Thousands of series: SELIC (11/4390), IPCA, credit, FX, BoP, IBC-Br (24364), public finance | Free, no key | REST `https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json`; Python **`python-bcb`** (`bcb.sgs.get`) | [python-bcb SGS](https://wilsonfreitas.github.io/python-bcb/sgs.html) |
| **BCB Expectativas (Focus)** | Weekly market expectations (~130 institutions) for SELIC, IPCA, GDP, FX — OData, **dated snapshots = PIT-friendly** | Free | OData; `python-bcb` `Expectativas` class | [Focus dataset](https://dadosabertos.bcb.gov.br/dataset/expectativas-mercado) · [python-bcb docs](https://wilsonfreitas.github.io/python-bcb/expectativas.html) |
| **BCB OData (PTAX, TaxaJuros, IFDATA, MercadoImobiliario)** | FX PTAX, lending rates, bank data, real-estate | Free | OData; `python-bcb` | [python-bcb OData](https://wilsonfreitas.github.io/python-bcb/odata.html) |
| **IBGE SIDRA** | Aggregated official stats: IPCA, PNAD (labor), PIM (industry), PMC (retail), PMS (services), GDP | Free | REST tables; Python **`sidrapy`** (`get_table`) | [github.com/AlanTaranti/sidrapy](https://github.com/AlanTaranti/sidrapy) |
| **IPEAData** | Curated macro/social/regional series (mirrors many sources, OData) | Free | OData `http://ipeadata.gov.br/api/odata4/`; Python **`ipeadatapy`** | [ipeadatapy on PyPI](https://pypi.org/project/ipeadatapy/) |
| **BCB Dados Abertos portal** | Catalog of all BCB open datasets/APIs | Free | Web + APIs | [dadosabertos.bcb.gov.br](https://dadosabertos.bcb.gov.br/) |
| **FGV IBRE** | IGP-M/IGP-DI inflation, business/consumer confidence (Sondagens), Monitor do PIB | Free + Paid | Web/portal | [portalibre.fgv.br](https://portalibre.fgv.br/) |
| **Tesouro Transparente / Tesouro Direto** | Fiscal data + sovereign bond reference prices | Free | CSV/API | [tesourotransparente.gov.br](https://www.tesourotransparente.gov.br/) |

> 🇧🇷 **Workflow tip:** `python-bcb` + `sidrapy` + `ipeadatapy` cover ~all public Brazilian macro. For unified access, **DBnomics also mirrors BCB and IBGE** providers — handy for cross-country joins. Selic series 11 (daily) vs 4390 (monthly accumulated) vs 432 (Selic Meta target) are *different* — pick deliberately.

---

## 10. Practical notes: licensing, survivorship, PIT, look-ahead

- **Look-ahead / revisions:** Revised macro (GDP, payrolls, retail sales) can move materially after first print. Use **ALFRED / Philly Fed RTDSM / FRED-MD vintages / `SAHMREALTIME`** for honest backtests. Latest-vintage CSVs are convenient but *leak the future*.
- **Release vs reference date:** Always model features off the **publication timestamp** (use FRED `release_dates`, or calendar APIs), not the period the data describes.
- **Frequency mixing:** Nowcasting/MIDAS handles daily→quarterly mixes (Cleveland Fed inflation, NY Fed nowcast are templates). FRED-MD/QD give clean aligned panels.
- **Licensing:** Official-statistics APIs are free/open but redistribution terms vary (OECD/Eurostat generally CC-BY; check per-dataset). Haver/LSEG/Macrobond/CEIC/TE are **commercially licensed — no redistribution**.
- **Survivorship:** Less of an issue for macro than for equities, but *discontinued/renamed series* (methodology breaks, e.g. CPI rebasing, GDP benchmark revisions) create structural breaks — flag and splice carefully.
- **Keys & rate limits:** FRED/BLS/BEA/Census/TE need keys; DBnomics/BCB SGS/IBGE are **keyless**. Respect rate limits (FRED ~120 req/min).

**Sources:** FRED-MD/QD ([databases](https://research.stlouisfed.org/econ/mccracken/fred-databases/), [FRED-QD review](https://www.stlouisfed.org/publications/review/2021/01/14/fred-qd-a-quarterly-database-for-macroeconomic-research), [description PDF](https://users.ssc.wisc.edu/~behansen/econometrics/FRED-MD_description.pdf), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3587692)) · ALFRED ([site](https://alfred.stlouisfed.org/), [API](https://fred.stlouisfed.org/docs/api/fred/alfred.html)) · [Philly Fed real-time data](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research) · [fredapi](https://github.com/mortada/fredapi) · DBnomics ([site](https://db.nomics.world/), [API](https://docs.db.nomics.world/web-api/)) · [EconDB API](https://www.econdb.com/api/) · [Nasdaq Data Link](https://data.nasdaq.com/) · [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html) · [pandaSDMX](https://pandasdmx.readthedocs.io/en/master/) · [FRED API](https://fred.stlouisfed.org/docs/api/fred/) · [World Bank Data360 API](https://data360.worldbank.org/en/api) · [OECD Data Explorer](https://data-explorer.oecd.org/) · [IMF API](https://data.imf.org/en/Resource-Pages/IMF-API) · BIS ([portal](https://data.bis.org/), [API](https://stats.bis.org/api-doc/v2/)) · [Eurostat API](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-introduction) · [ECB Data Portal](https://data.ecb.europa.eu/) · [BLS](https://www.bls.gov/developers/) · [BEA](https://apps.bea.gov/API/signup/) · [Census](https://www.census.gov/data/developers.html) · [UN Comtrade](https://comtradeplus.un.org/) · [Haver](https://www.haver.com/) · [LSEG Datastream](https://www.lseg.com/en/data-analytics/datastream-and-macroeconomics) · [Trading Economics API](https://tradingeconomics.com/api/) · [Macrobond](https://www.macrobond.com/) · [CEIC](https://www.ceicdata.com/) · [Finnhub calendar](https://finnhub.io/docs/api/economic-calendar) · [FMP calendar](https://site.financialmodelingprep.com/developer/docs/stable/economics-calendar) · [Econoday](https://www.econoday.com/) · [GDPNow](https://www.atlantafed.org/research-and-data/data/gdpnow) · [NY Fed Nowcast](https://www.newyorkfed.org/research/policy/nowcast) · [Cleveland Fed inflation nowcast](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting) · [FRED SAHMREALTIME](https://fred.stlouisfed.org/series/SAHMREALTIME) · [Fed EBP](https://www.federalreserve.gov/econres/notes/feds-notes/updating-the-recession-risk-and-the-excess-bond-premium-20161006.html) · [Chicago Fed NFCI](https://www.chicagofed.org/research/data/nfci/current-data) · [Philly Fed ADS](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ads) · [NBER cycle dating](https://www.nber.org/research/business-cycle-dating) · [python-bcb](https://wilsonfreitas.github.io/python-bcb/) · [sidrapy](https://github.com/AlanTaranti/sidrapy) · [ipeadatapy](https://pypi.org/project/ipeadatapy/) · [BCB Dados Abertos](https://dadosabertos.bcb.gov.br/) · [FGV IBRE](https://portalibre.fgv.br/)

**Keywords:** macroeconomic data, economic data API, FRED-MD, FRED-QD, ALFRED, vintage data, point-in-time data (dados ponto-no-tempo), real-time macro, DBnomics, SDMX, World Bank API, OECD, IMF, BIS, Eurostat, ECB, BLS, BEA, economic calendar API (calendário econômico), nowcasting, GDPNow, NY Fed Nowcast, Cleveland Fed inflation nowcast, Sahm rule (regra de Sahm), yield curve recession, excess bond premium, NFCI, CFNAI, ADS index, recession indicators (indicadores de recessão), look-ahead bias (viés de antecipação), survivorship, Trading Economics, Haver Analytics, LSEG Datastream, Macrobond, CEIC, Banco Central do Brasil, BCB SGS, IBC-Br, Focus (Expectativas de Mercado), IBGE SIDRA, IPEAData, python-bcb, sidrapy, ipeadatapy, SELIC, IPCA, dados macroeconômicos Brasil.
