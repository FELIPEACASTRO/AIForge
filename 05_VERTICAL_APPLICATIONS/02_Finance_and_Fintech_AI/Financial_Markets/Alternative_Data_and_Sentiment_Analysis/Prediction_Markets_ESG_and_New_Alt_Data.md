# Prediction Markets, ESG & New Alternative-Data Sources

> A dense, current (2025-2026) field guide to **new alternative-data sources** for financial markets — prediction-market probabilities, ESG/climate-finance feeds, news/event macro data, satellite/geospatial, web/consumer signals, and crypto on-chain analytics — with real URLs, API access notes, and point-in-time/licensing caveats. Audience: quants and researchers (Brazil-heavy; English body, termos em português entre parênteses).

This page catalogs sources **not already indexed elsewhere** in this repo. Equity exchanges and venues (US NYSE/Nasdaq, Brazil B3, India NSE/BSE, China SSE/SZSE, Japan/HK/Korea/Taiwan, ASEAN), generic global price APIs, Kaggle/HuggingFace data pulls, and arXiv q-fin are covered in their own pages and are only referenced here in passing. The focus below is **alternative data (dados alternativos)** — the non-price signals that increasingly drive systematic strategies and nowcasting.

---

## 1. Prediction markets as data / signal (mercados de previsão)

Prediction markets convert crowd money into **real-time probabilities** for discrete events (elections, rate decisions, CPI prints, earnings, geopolitics). Treated as a live macro signal, the marginal price is a calibrated forecast of `P(event)`. In 2025 the asset class went institutional: **Intercontinental Exchange (parent of NYSE) committed up to US$2bn to Polymarket**, valuing it ~US$8-9bn, and pledged to **distribute Polymarket's data** through ICE — i.e., prediction-market odds are becoming a licensed market-data product alongside equities and futures ([Fortune, Oct 2025](https://fortune.com/crypto/2025/10/07/polymarket-2-billion-intercontinental-exchange-new-york-stock-exchange-9-billion/); [Polymarket — Wikipedia](https://en.wikipedia.org/wiki/Polymarket)). Rival **Kalshi** raised >US$1bn at a ~US$22bn valuation, with 2025 industry nominal volumes near US$30bn ([TechCrunch, Oct 2025](https://techcrunch.com/2025/10/10/kalshi-hits-5b-valuation-days-after-rival-polymarket-gets-2b-nyse-backing-at-8b/)).

| Platform | What it is | Regulation | Free? | API / how to access | Link |
|---|---|---|---|---|---|
| **Polymarket** | Largest market by volume; on-chain CLOB on Polygon, USDC-settled | Offshore crypto; UMA oracle resolution | Yes (public read) | **Gamma API** (`gamma-api.polymarket.com`, market metadata), **CLOB API** (`clob.polymarket.com/markets`, order book + OHLC), **Data API** (positions/PnL); Polygon RPC for raw on-chain; `py-clob-client` | [docs.polymarket.com](https://docs.polymarket.com/) |
| **Kalshi** | CFTC-regulated US event exchange; cash-settled binary contracts | **CFTC DCM** (regulated) | Yes (public read) | REST + WebSocket (`trading-api.kalshi.co/trade-api/v2`); markets, candlesticks, order book; `kalshi-python` client | [docs.kalshi.com](https://docs.kalshi.com/) |
| **Metaculus** | Forecasting platform (not money) — aggregated expert/crowd distributions | N/A (research) | Yes | Public REST/Swagger API (`metaculus.com/api/`); CSV question download; commercial/research tiers | [metaculus.com/api](https://www.metaculus.com/api/) |
| **Manifold Markets** | Play-money (mana) markets; huge breadth, fast resolution | N/A (play-money) | Yes (very open) | Full REST API (`docs.manifold.markets/api`); `manifoldpy`/`PyManifold`; ideal for prototyping signal pipelines | [docs.manifold.markets/api](https://docs.manifold.markets/api) |
| **PredictIt** | US-politics real-money market (academic-exempt) | CFTC no-action (winding down legally contested) | Yes | Read-only `predictit.org/api/marketdata/all/` (no auth; ~1 req/min) | [predictit.org](https://www.predictit.org/) |
| **CFTC / federal** | Regulatory backdrop for US event contracts | — | Yes | CRS primer "Prediction Markets: Policy Issues for Congress" (IF13187) | [congress.gov](https://www.congress.gov/crs-product/IF13187) |

**How event probabilities feed models.** (1) *Macro nowcasting*: Kalshi/Polymarket contracts on "Fed cuts in Sep", "CPI > 3%", "recession in 2026" give a market-implied path that can be differenced against survey medians or futures-implied curves to extract surprise. (2) *Event-study triggers*: probability jumps mark information arrival timestamps for high-frequency reaction studies. (3) *Cross-asset overlays*: election/geopolitics odds as exogenous regressors for FX/rates vol. **Caveats**: thin books → wide spreads and stale mid-prices; favorite-longshot bias; resolution/oracle risk (UMA disputes on Polymarket); play-money venues (Manifold) are *not* dollar-calibrated. Curated index: [`spfunctions/awesome-prediction-markets`](https://github.com/spfunctions/awesome-prediction-markets).

**Free aggregation / research tooling.** Beyond the per-venue APIs, two patterns help researchers: (a) unified feeds such as `simplefunctions.dev` (REST changes feed across venues) and the curated [`awesome-prediction-markets`](https://github.com/spfunctions/awesome-prediction-markets) list; (b) joining odds to **free macro series** — FRED, World Bank, Eurostat (`ec.europa.eu/eurostat/api`), OECD SDMX, and DBnomics (`db.nomics.world`) — to build market-implied-vs-consensus spreads. Polymarket's full on-chain history is also queryable via **The Graph / Bitquery** subgraphs and **Dune** (see §6), which avoids rate limits on the hosted APIs.

**Brazil access**: Polymarket/Kalshi are not CVM-registered for Brazilian retail; Polymarket trading is geofenced in several jurisdictions. Use the public read APIs for **data/signal** only — do not assume legal trading access from Brazil. Domestically, regulated *bolsas de apostas* (fixed-odds betting, Lei 14.790/2023, effective 2025) are a sports/event-odds source but are **not** general prediction markets and lack open data APIs.

---

## 2. ESG & climate-finance data (dados ESG e de clima)

Commercial ESG ratings consolidated and **closed their free public score databases in 2025** (MSCI, Sustainalytics), while the EU ESG-rating-provider regime entered into force Jan 2025 (most obligations from 2026). TCFD was disbanded (Oct 2023); its recommendations are now absorbed into the **ISSB IFRS S1/S2** standards ([IFRS — ISSB and TCFD](https://www.ifrs.org/sustainability/tcfd/)). Net effect: paid ratings got harder to access freely, but **open emissions/climate-policy datasets exploded**.

### Commercial ESG ratings & data

| Provider | Coverage | Free? | API / how | Link |
|---|---|---|---|---|
| **MSCI ESG** | Ratings + 21,000+ data points, climate metrics | Paid (public DB removed 2025) | **ESG Data API v3.0**, OAuth 2.0 | [developer.msci.com/apis](https://developer.msci.com/apis) |
| **Morningstar Sustainalytics** | ESG Risk Ratings, controversies | Paid (Global Access/Datafeeds/API) | Swagger/OpenAPI ESG API | [sustainalytics.com/esg-data](https://www.sustainalytics.com/esg-data) |
| **LSEG (Refinitiv) ESG** | ~16,000 companies, point-in-time history | Paid | Data Platform / Workspace + WRDS | [lseg.com](https://www.lseg.com/en/data-analytics/sustainable-finance) |
| **Bloomberg ESG** | Disclosure + scores, climate | Paid (Terminal/BBG Data License) | `BLP`/`blpapi`, Data License | [bloomberg.com](https://www.bloomberg.com/professional/products/data/enterprise-catalog/esg/) |
| **CDP** | Self-reported corporate climate/water/forest disclosures | Partly free (scores public; full data paid) | Bulk data via CDP + WRDS | [cdp.net](https://www.cdp.net/) |

### Free / open climate & emissions data

| Source | What | Free? | API / how | Link |
|---|---|---|---|---|
| **EDGAR** (EU JRC) | Global GHG/air-pollutant emissions by country, 1970-2024, 0.1°×0.1° grids, up to hourly | **Free** | CSV/NetCDF bulk download; Fast-Track v8/2025 | [edgar.jrc.ec.europa.eu](https://edgar.jrc.ec.europa.eu/) |
| **Climate TRACE** | Asset-level emissions inventory, **352M+ assets** (power, steel, ships, refineries) | **Free** | Open download + REST API | [climatetrace.org](https://climatetrace.org/) |
| **Climate Policy Radar** | Full text of **30,000+ climate laws/policies** + climate-finance projects, AI search | **Free** (open data, nonprofit) | Open database + API | [climatepolicyradar.org](https://www.climatepolicyradar.org/) |
| **IFRS / ISSB** | S1/S2 standards (supersede TCFD) for disclosure structure | Free (standards) | Document downloads | [ifrs.org/sustainability](https://www.ifrs.org/sustainability/tcfd/) |
| **Urgentem** | Corporate carbon-footprint / Scope 1-3, financed-emissions data | Paid (commercial) | Datafeed/API, distribution partners | [via Datarade / provider listings](https://datarade.ai/) |

**Caveats**: ESG scores are **provider-divergent** ("rate the raters" shows low cross-vendor correlation); restatement and **backfill** create look-ahead bias — insist on point-in-time vintages. Self-reported (CDP) data has selection bias; satellite-derived (Climate TRACE) and statistical (EDGAR) estimates have methodology-version breaks.

**Brazil note**: B3 publishes the **ISE B3** (Índice de Sustentabilidade Empresarial) and **ICO2** (carbon-efficient) indices; ESG-themed BDRs/ETFs give local exposure but are not a substitute for raw provider data.

---

## 3. News / event macro alternative data (notícias e eventos)

| Source | Category | Free? | API / how | Link |
|---|---|---|---|---|
| **GDELT 2.0** | Global news events (100+ languages), tone, GKG; updates every 15 min | **Free** | DOC 2.0 full-text API (rolling 3 months); event CSVs at `data.gdeltproject.org`; **BigQuery** public dataset; `gdelt` (PyPI) | [gdeltproject.org](https://www.gdeltproject.org/) |
| **Wikimedia Pageviews** | Attention proxy per article/project (since 2015), bot-filtered | **Free** | AQS REST Pageviews API; hourly dumps at `dumps.wikimedia.org`; Wikimedia Enterprise (paid) for bulk | [wikitech — Pageviews](https://wikitech.wikimedia.org/wiki/Data_Platform/AQS/Pageviews) |
| **Google Trends** | Relative search interest (índice de busca) | Free (normalized) | Unofficial endpoints / `pytrends`; new official Trends API in beta | [trends.google.com](https://trends.google.com/) |
| **Common Crawl CC-NEWS** | Daily WARC archive of global news pages | **Free** (AWS egress cost) | `s3://commoncrawl/crawl-data/CC-NEWS/`; `news-please` extractor | [commoncrawl.org/blog/news-dataset-available](https://commoncrawl.org/blog/news-dataset-available) |
| **RavenPack / Bigdata.com** | Investment-grade news analytics & sentiment, 40,000+ sources, 300,000+ entities; new LLM platform **Bigdata.com** | Paid | REST/SFTP feeds; **WRDS** access for academics | [ravenpack.com](https://www.ravenpack.com/) |
| **LSEG/Bloomberg news analytics** | Real-time machine-readable news + sentiment scores | Paid | Data Platform / `blpapi` MRN feeds | [lseg.com](https://www.lseg.com/) |

**How they feed models**: GDELT tone and event counts → geopolitical-risk and conflict nowcasts; Wikipedia/Google Trends → demand and attention factors (e.g., ticker/brand interest as earnings predictor); RavenPack event sentiment → cross-sectional equity sentiment factor and earnings-surprise drift. **Caveats**: GDELT/Trends carry **methodology revisions** and Trends is *renormalized* on each query (not point-in-time stable); media-volume signals are confounded by coverage intensity; commercial feeds require **timestamped, as-published** delivery to avoid look-ahead.

---

## 4. Satellite & geospatial alternative data (dados de satélite)

| Provider | Signal | Free? | API / how | Link |
|---|---|---|---|---|
| **Planet Labs** | Daily high-frequency imagery; Insights Platform (APIs, GIS) | Paid (science/research tier) | **Data API**, Insights Platform | [planet.com](https://www.planet.com/) |
| **Orbital Insight** | Geospatial analytics (foot traffic, fill levels, geolocation) | Paid | Dashboards/feeds (firm distressed 2023-24 — verify status) | [en.wikipedia.org/wiki/Orbital_Insight](https://en.wikipedia.org/wiki/Orbital_Insight) |
| **RS Metrics** | Metals/industrials, parking-lot retail traffic, ESG asset signals | Paid | Data feeds/dashboards | [rsmetrics.com](https://www.rsmetrics.com/) |
| **SpaceKnow** | Economic activity indices from imagery (China mfg, etc.) | Paid | API/dashboards | [spaceknow.com](https://www.spaceknow.com/) |
| **Kpler (incl. MarineTraffic/FleetMon)** | Commodity flows + AIS vessel tracking | Paid (enterprise) | API; MarineTraffic now enterprise-only | [kpler.com](https://www.kpler.com/) |

**Use cases**: parking-lot car counts → retail same-store sales nowcast; oil-tank shadow/float → crude inventory estimates; nighttime lights and thermal → industrial-activity and energy proxies; agricultural NDVI → crop-yield/soft-commodity signals. **Open/free baselines** worth pairing: Sentinel-2 / Landsat (Copernicus, USGS) and the Sentinel Hub API for self-built indices. **Caveats**: cloud cover and revisit gaps create missing-data; vendor model changes break time series; tiny-sample alpha decays fast once a dataset is widely licensed.

---

## 5. Web / consumer alternative data (consumo e web)

| Source | Signal | Free? | API / how | Link |
|---|---|---|---|---|
| **Similarweb** | Website visits, sources, engagement → online-sales nowcast | Paid (limited free) | Digital Data API (visits, duration, geo) | [similarweb.com](https://www.similarweb.com/) |
| **Sensor Tower (acq. data.ai)** | App downloads, revenue, MAU/DAU; **data.ai merged into Sensor Tower (2024)** | Paid | Store Intelligence API | [sensortower.com](https://sensortower.com/) |
| **Earnest Analytics** | US card-panel consumer spend; **Orion** card data + consumer index | Paid | Datafeeds/dashboards | [earnestanalytics.com](https://www.earnestanalytics.com/) |
| **Facteus** | 40M+ active cards, ~1-day lag transaction panel | Paid | API/feeds; Snowflake Marketplace | [facteus.com](https://facteus.com/) |
| **Revelio Labs** | Workforce: 1.1B+ profiles, headcount, attrition, layoffs, sentiment | Paid | API/feed/dashboard; **WRDS** for academics | [reveliolabs.com](https://www.reveliolabs.com/) |
| **LinkUp** | Direct-from-employer job postings since 2007 (clean, deduped) | Paid | Data feed; Snowflake (w/ Revelio) | [linkup.com](https://www.linkup.com/) |
| **MarineTraffic** | AIS vessel positions (now under Kpler) | Paid | API (enterprise) | [marinetraffic.com](https://www.marinetraffic.com/) |

**How they feed models**: card-spend panels → pre-announcement revenue estimates (Earnest reports ~90% earnings-surprise accuracy on its panels); Similarweb traffic → e-commerce quarterly sales; job-postings velocity (LinkUp/Revelio) → hiring/expansion and sector-rotation signals; app downloads → consumer-tech revenue. **Caveats**: **panel bias** (card panels skew by issuer/demographic), **coverage drift** over time, and **survivorship** in company mappings; most providers gate history behind enterprise contracts, so academics should route via **WRDS**/Snowflake Marketplace where available.

**Brazil-relevant consumer signals**: most US card/web panels under-cover Brazilian issuers and tickers, so for B3 names pair global vendors with local proxies — **Pix** transaction volumes (Banco Central do Brasil open statistics, `dadosabertos.bcb.gov.br`), Google Trends BR, and Similarweb country breakdowns — to nowcast domestic retail and fintech revenue. BDRs of US consumer names let local funds trade on the global alt-data signal directly on B3.

---

## 6. Crypto / on-chain alternative data (dados on-chain)

| Platform | Focus | Free? | API / how | Link |
|---|---|---|---|---|
| **Glassnode** | On-chain fundamentals (BTC/ETH), entity-adjusted metrics, cohorts | Freemium (Advanced/Pro paid) | REST timeseries API + MCP + CLI | [glassnode.com](https://glassnode.com/) |
| **Nansen** | Wallet labels ("smart money"), flows; 500M+ labeled addresses | Paid (some free) | API + dashboards | [nansen.ai](https://www.nansen.ai/) |
| **Dune** | SQL over decoded chain data; community dashboards | Freemium | Dune API (query execution, results) | [dune.com](https://dune.com/) |
| **Santiment** | On-chain + social sentiment, dev activity | Freemium | GraphQL `Sanbase`/`sanpy` | [santiment.net](https://santiment.net/) |
| **Token Terminal** | Protocol financials (revenue, P/S, fees) | Paid | Data API | [tokenterminal.com](https://tokenterminal.com/) |

**How they feed models**: exchange in/outflows and stablecoin supply → liquidity/risk-on signals; "smart-money" wallet flows → momentum/rotation; protocol revenue multiples (Token Terminal) → fundamental crypto factors; social volume (Santiment) → sentiment-extreme reversals. **Caveats**: **address-clustering/labeling is heuristic** (false labels), chains/methods change, and on-chain metrics are **not point-in-time** unless the vendor snapshots vintages — re-derivations silently restate history. Pair with raw-node or CoinAPI/`The Graph` infrastructure for reproducibility.

---

## 7. Cross-cutting caveats: point-in-time, look-ahead & licensing

- **Point-in-time (PIT) / vintages**: use data *as it was known on the decision date*. ESG restatements, fundamental backfills, and on-chain re-labeling all introduce **look-ahead bias** if you use the latest revision in a backtest ([Deloitte — alternative data](https://www.deloitte.com/us/en/insights/industry/financial-services/alternative-data-for-investors-from-discovery-to-institutionalization.html)).
- **Survivorship (viés de sobrevivência)**: include delisted/dead entities; survivorship alone can overstate returns ~2%/yr (more for small/levered) and understate drawdowns by ~14pp ([JFQA — Survival, Look-Ahead Bias](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/survival-lookahead-bias-and-persistence-in-hedge-fund-performance/309BF3E4C6A6EF54604E325F234ED3A5)).
- **Short history**: most alt-data starts ≤2015 — limited regimes, fragile signal estimation.
- **Licensing & compliance**: commercial feeds (RavenPack, Similarweb, card panels) restrict redistribution and backtesting scope; verify **MNPI/data-privacy** (card/geolocation data anonymization) and territorial use before research-to-production. Free sources (GDELT, EDGAR, Climate TRACE, Common Crawl, Wikimedia) are open but carry attribution and AWS-egress/compute costs.
- **Alpha decay**: once a dataset is broadly licensed, edge erodes — combine 2-3 complementary sources rather than one.

---

**Sources:** [Polymarket docs](https://docs.polymarket.com/) · [Polymarket/ICE — Fortune](https://fortune.com/crypto/2025/10/07/polymarket-2-billion-intercontinental-exchange-new-york-stock-exchange-9-billion/) · [Kalshi docs](https://docs.kalshi.com/) · [Kalshi/Polymarket — TechCrunch](https://techcrunch.com/2025/10/10/kalshi-hits-5b-valuation-days-after-rival-polymarket-gets-2b-nyse-backing-at-8b/) · [Metaculus API](https://www.metaculus.com/api/) · [Manifold API](https://docs.manifold.markets/api) · [awesome-prediction-markets](https://github.com/spfunctions/awesome-prediction-markets) · [CRS — prediction markets](https://www.congress.gov/crs-product/IF13187) · [MSCI APIs](https://developer.msci.com/apis) · [Sustainalytics ESG](https://www.sustainalytics.com/esg-data) · [CDP](https://www.cdp.net/) · [IFRS/ISSB-TCFD](https://www.ifrs.org/sustainability/tcfd/) · [EDGAR](https://edgar.jrc.ec.europa.eu/) · [Climate TRACE](https://climatetrace.org/) · [Climate Policy Radar](https://www.climatepolicyradar.org/) · [GDELT](https://www.gdeltproject.org/) · [Wikimedia Pageviews](https://wikitech.wikimedia.org/wiki/Data_Platform/AQS/Pageviews) · [Common Crawl CC-NEWS](https://commoncrawl.org/blog/news-dataset-available) · [RavenPack](https://www.ravenpack.com/) · [Planet](https://www.planet.com/) · [Orbital Insight](https://en.wikipedia.org/wiki/Orbital_Insight) · [Kpler/MarineTraffic](https://www.kpler.com/) · [Similarweb](https://www.similarweb.com/) · [Earnest](https://www.earnestanalytics.com/) · [Facteus](https://facteus.com/) · [Revelio Labs](https://www.reveliolabs.com/) · [Glassnode](https://glassnode.com/) · [Nansen](https://www.nansen.ai/) · [Dune](https://dune.com/) · [Santiment](https://santiment.net/) · [Token Terminal](https://tokenterminal.com/) · [Deloitte alt-data](https://www.deloitte.com/us/en/insights/industry/financial-services/alternative-data-for-investors-from-discovery-to-institutionalization.html) · [JFQA look-ahead bias](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/survival-lookahead-bias-and-persistence-in-hedge-fund-performance/309BF3E4C6A6EF54604E325F234ED3A5)

**Keywords:** alternative data, dados alternativos, prediction markets, mercados de previsão, Polymarket, Kalshi, Metaculus, event probabilities, ESG data, dados ESG, climate finance, finanças climáticas, EDGAR, Climate TRACE, GDELT, news sentiment, análise de sentimento, satellite data, dados de satélite, geospatial, card spend, gasto com cartão, web traffic, tráfego web, on-chain analytics, Glassnode, Nansen, Dune, point-in-time, viés de antecipação, survivorship bias, viés de sobrevivência, nowcasting, RavenPack, alpha decay
